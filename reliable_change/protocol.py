"""Halo inference, shared sensor stresses and independent-event statistics."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def tiled_predict(network, batch, tile_size=512, halo=64):
    """Stitch nonoverlapping crop centers using reflected halo observations."""
    x = batch['x']
    height, width = x.shape[-2:]
    if max(height, width) <= tile_size:
        return network(batch)
    core = tile_size-2*halo
    if core <= 0 or tile_size % 32:
        raise ValueError('tile_size must be divisible by32 and exceed twice halo')
    output = {}
    def reflected(start, length, limit):
        index = torch.arange(start, start+length, device=x.device)
        if limit == 1:
            return index*0
        index = index.remainder(2*(limit-1))
        return torch.minimum(index, 2*(limit-1)-index)
    for top in range(0, height, core):
        rows = reflected(top-halo, tile_size, height)
        for left in range(0, width, core):
            cols = reflected(left-halo, tile_size, width)
            patch = dict(batch, x=x.index_select(-2, rows).index_select(-1, cols))
            prediction = network(patch)
            h, w = min(core, height-top), min(core, width-left)
            for key, value in prediction.items():
                if isinstance(value, torch.Tensor) and value.ndim == 4 and value.shape[-2:] == (tile_size, tile_size):
                    if key not in output:
                        output[key] = value.new_empty((len(x), value.shape[1], height, width))
                    output[key][..., top:top+h, left:left+w] = value[..., halo:halo+h, halo:halo+w]
    return output


def stress(data, state, kind, level, seed=13, modality='both', cloud_masks=None):
    """Apply fixed observable corruption; labels never define its realization."""
    rng = np.random.default_rng(seed)
    result = {k: v.copy() for k, v in data.items()}
    x = result['x']
    n, _, height, width = x.shape
    quality = result.get('quality', np.ones((n, 2, height, width), np.float32)).copy()
    if kind == 'cloud':
        if not 0 <= level <= 1:
            raise ValueError('Cloud fraction must be between0 and1')
        count = round(level*height*width)
        for i in range(n):
            field = gaussian_filter(rng.normal(size=(height, width)), sigma=max(1, min(height, width)/16)) if cloud_masks is None else cloud_masks[i % len(cloud_masks)]
            if field.shape != (height, width):
                raise ValueError('Cloud masks must match image resolution')
            mask = np.zeros(height*width, bool)
            if count:
                mask[np.argsort(np.asarray(field).ravel(), kind='stable')[-count:]] = True
            mask = mask.reshape(height, width)
            x[i, 3:6, mask] = np.asarray(state['optical_mean'])
            quality[i, 0, mask] = 0
    elif kind == 'speckle':
        if level <= 0:
            raise ValueError('Gamma equivalent looks must be positive')
        linear = np.maximum(np.expm1(x[:, 6:7]*state['sar_std']+state['sar_mean']), 0)
        gamma = rng.gamma(level, 1/level, size=linear.shape)
        log_sar = np.clip(np.log1p(linear*gamma), state['sar_lower'], state['sar_upper'])
        x[:, 6:7] = (log_sar-state['sar_mean'])/state['sar_std']
        quality[:, 1] *= level/(level+1)
    elif kind == 'shift':
        pixels = int(level)
        if pixels < 0 or pixels != level:
            raise ValueError('Shift must be a nonnegative integer pixel count')
        if pixels:
            for channels, q in [((3,6),0),((6,7),1)]:
                if modality not in {'both', 'optical' if q == 0 else 'sar'}:
                    continue
                lo, hi = channels
                padded = np.pad(x[:, lo:hi], ((0,0),(0,0),(pixels,pixels),(pixels,pixels)), mode='reflect')
                x[:, lo:hi] = padded[..., pixels:pixels+height, :width]
                quality[:, q] *= max(0., 1-pixels/8)
    else:
        raise ValueError('Unknown corruption: '+kind)
    x[data['available'][:, 0] == 0, 3:6] = 0
    x[data['available'][:, 1] == 0, 6:7] = 0
    result['x'], result['quality'] = x.astype(np.float32), quality.astype(np.float32)
    return result


def event_statistics(frame, reference, metric='f1', draws=10000, seed=13):
    """Average seeds within event, then paired event bootstrap and sign-flip/Holm."""
    if frame.duplicated(['method','event','seed']).any():
        raise ValueError('Each method/event/seed must have one score for one condition')
    if not np.isfinite(frame[metric]).all():
        raise ValueError('Nonfinite metric values')
    event = frame.groupby(['method','event'])[metric].agg(['mean','std','count']).reset_index()
    wide = event.pivot(index='event', columns='method', values='mean')
    if reference not in wide:
        raise ValueError('Missing reference method')
    rng = np.random.default_rng(seed)
    summaries = {}
    for name, values in wide.items():
        values = values.dropna()
        summaries[name] = dict(mean=float(values.mean()), event_sd=float(values.std(ddof=1)) if len(values)>1 else None, events=len(values))
    comparisons = []
    for name in wide:
        if name == reference:
            continue
        paired = wide[[reference,name]].dropna()
        diff = (paired[reference]-paired[name]).to_numpy()
        if len(diff)<2:
            raise ValueError('Paired event inference needs at least two common events')
        boot = rng.choice(diff, (draws,len(diff)), replace=True).mean(1)
        signs = np.asarray(list(itertools.product([-1,1], repeat=len(diff)))) if len(diff)<=16 else rng.choice([-1,1], (draws,len(diff)))
        null = (signs*diff).mean(1)
        p = float(np.mean(abs(null)>=abs(diff.mean())-1e-12))
        low,high = np.quantile(boot,[.025,.975])
        comparisons.append(dict(method=name, reference_minus_method=float(diff.mean()), ci_low=float(low),ci_high=float(high),p_raw=p,
                                paired_events=paired.index.astype(str).tolist()))
    running = 0.
    for i,row in enumerate(sorted(comparisons,key=lambda row:row['p_raw'])):
        running = max(running,min(1.,(len(comparisons)-i)*row['p_raw']))
        row['p_holm'] = running
    return dict(metric=metric, reference=reference, seed=seed, bootstrap_draws=draws,
                methods=summaries,event_seed_summary=event.replace({np.nan:None}).to_dict('records'),comparisons=comparisons)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['stress','statistics'])
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--preprocessing')
    p.add_argument('--kind',choices=['cloud','speckle','shift'])
    p.add_argument('--levels',nargs='+',type=float)
    p.add_argument('--modality',choices=['optical','sar','both'],default='both')
    p.add_argument('--cloud-masks')
    p.add_argument('--seed',type=int,default=13)
    p.add_argument('--draws',type=int,default=10000)
    p.add_argument('--reference',default='full')
    p.add_argument('--metric',default='f1')
    a=p.parse_args()
    out=Path(a.output)
    if a.command=='statistics':
        import pandas as pd
        result=event_statistics(pd.read_csv(a.input),a.reference,a.metric,a.draws,a.seed)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    else:
        from .cli import read
        if not a.kind or not a.levels or not a.preprocessing:
            p.error('stress needs --kind --levels --preprocessing')
        d=read(a.input)
        state=json.loads(Path(a.preprocessing).read_text(encoding='utf-8'))
        clouds=np.load(a.cloud_masks,allow_pickle=False) if a.cloud_masks else None
        out.mkdir(parents=True,exist_ok=True)
        for level in a.levels:
            np.savez_compressed(out/f'{a.kind}_{level:g}.npz',**stress(d,state,a.kind,level,a.seed,a.modality,clouds))
        (out/'stress.json').write_text(json.dumps(vars(a),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
