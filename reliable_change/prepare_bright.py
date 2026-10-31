"""Prepare event-disjoint BRIGHT NPZ tiles with training-only statistics."""
import argparse
import json
from pathlib import Path

import numpy as np

EVENTS = ['Bata', 'Beirut', 'Hawaii', 'Libya', 'Noto']


def prepare(source, output, fold):
    with np.load(source, allow_pickle=False) as archive:
        d = {key: archive[key] for key in archive.files}
    x = d['x'].astype(np.float32)
    if x.ndim != 4 or x.shape[1] != 7 or 'event' not in d or 'group' not in d:
        raise ValueError('x[N,7,H,W], y, available, event names and globally unique source group IDs required')
    event = d['event'].astype(str)
    if not np.isin(event, EVENTS).all():
        raise ValueError('Expected the five named matched-triplet BRIGHT events')
    masks = dict(test=event == EVENTS[fold], validation=event == EVENTS[(fold+1) % 5])
    masks['train'] = ~(masks['test'] | masks['validation'])
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f'Empty {name} event fold')
        for other, second in masks.items():
            if other != name and np.intersect1d(d['group'][mask], d['group'][second]).size:
                raise ValueError('A source tile group crosses independent event partitions')
    optical = np.concatenate([x[masks['train'], :3].transpose(0, 2, 3, 1).reshape(-1, 3),
                              x[masks['train'] & (d['available'][:, 0] > 0), 3:6].transpose(0, 2, 3, 1).reshape(-1, 3)])
    lower, upper = np.percentile(optical, [1, 99], axis=0)
    optical_mean = np.clip((optical-lower)/np.maximum(upper-lower, 1e-6), 0, 1).mean(0)
    for start in [0, 3]:
        x[:, start:start+3] = np.clip((x[:, start:start+3]-lower[None, :, None, None])/
                                      np.maximum(upper-lower, 1e-6)[None, :, None, None], 0, 1)
    if (x[:, 6:7] < 0).any():
        raise ValueError('SAR input must be calibrated nonnegative linear intensity')
    sar = np.log1p(x[:, 6:7])
    train_sar = sar[masks['train'] & (d['available'][:, 1] > 0)]
    if not train_sar.size:
        raise ValueError('Training fold needs acquired SAR for preprocessing fit')
    sar_low, sar_high = np.percentile(train_sar, [1, 99])
    sar = np.clip(sar, sar_low, sar_high)
    sar_mean, sar_std = float(np.clip(train_sar, sar_low, sar_high).mean()), max(float(np.clip(train_sar, sar_low, sar_high).std()), 1e-6)
    x[:, 6:7] = (sar-sar_mean)/sar_std
    x[d['available'][:, 0] == 0, 3:6] = 0
    x[d['available'][:, 1] == 0, 6:7] = 0
    d['x'] = x.astype(np.float32)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        np.savez_compressed(output/f'{name}.npz', **{key: value[mask] if value.ndim and len(value) == len(x) else value for key, value in d.items()})
    state = dict(fold=fold, test_event=EVENTS[fold], validation_event=EVENTS[(fold+1) % 5],
                 optical_lower=lower.tolist(), optical_upper=upper.tolist(),
                 optical_mean=optical_mean.tolist(),
                 sar_lower=float(sar_low), sar_upper=float(sar_high), sar_mean=sar_mean, sar_std=sar_std)
    (output/'preprocessing.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    a = p.parse_args()
    prepare(a.input, a.output, a.fold)


if __name__ == '__main__':
    main()
