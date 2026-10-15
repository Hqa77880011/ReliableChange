import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--metric',choices=['lpips','saliency'],required=True)
    args=parser.parse_args()
    with np.load(args.predictions,allow_pickle=False) as data:
        prediction=data['prediction'] if 'prediction' in data.files else data['probability']
    with np.load(args.reference,allow_pickle=False) as data:
        target=data['y']
    if prediction.shape!=target.shape:
        raise ValueError('Prediction and reference image arrays must match')
    if args.metric=='lpips':
        import lpips
        import torch
        model=lpips.LPIPS(net='alex').eval()
        values=[]
        for first,second in zip(prediction,target):
            a=torch.from_numpy(first[None]).float()
            b=torch.from_numpy(second[None]).float()
            if a.shape[1]==1:
                a=a.repeat(1,3,1,1)
                b=b.repeat(1,3,1,1)
            with torch.no_grad():values.append(float(model(a*2-1,b*2-1).mean()))
        result={'lpips':float(np.mean(values))}
    else:
        from py_sod_metrics import Emeasure,Fmeasure,Smeasure,WeightedFmeasure
        fm,em,sm,wfm=Fmeasure(),Emeasure(),Smeasure(),WeightedFmeasure()
        for first,second in zip(prediction,target):
            first=np.rint(np.clip(first.squeeze(),0,1)*255).astype(np.uint8)
            second=np.rint(np.clip(second.squeeze(),0,1)*255).astype(np.uint8)
            for metric in [fm,em,sm,wfm]:metric.step(pred=first,gt=second)
        f=fm.get_results()['fm'];e=em.get_results()['em']
        result={'adaptive_fmeasure':float(f['adp']),'max_fmeasure':float(np.max(f['curve'])),'mean_fmeasure':float(np.mean(f['curve'])),'adaptive_emeasure':float(e['adp']),'max_emeasure':float(np.max(e['curve'])),'smeasure':float(sm.get_results()['sm']),'weighted_fmeasure':float(wfm.get_results()['wfm'])}
    Path(args.output).write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
