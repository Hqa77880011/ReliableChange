"""Small execution checks for stitching, shared stresses and event statistics."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from reliable_change.baselines import build_model
from reliable_change.protocol import event_statistics, stress, tiled_predict


def main():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    class PixelModel(nn.Module):
        def forward(self,b):
            return {'probability':b['x'][:,:1].sigmoid()}
    large={'x':torch.randn(1,7,65,73),'available':torch.ones(1,2)}
    tiled=tiled_predict(PixelModel(),large,64,16)['probability']
    torch.testing.assert_close(tiled,large['x'][:,:1].sigmoid(),atol=0,rtol=0)
    for architecture in ['unet']:
        c=dict(architecture=architecture,variant='full',widths=[4,8,12,16])
        model=build_model(c)
        b=dict(x=torch.rand(2,7,32,32),y=torch.rand(2,1,32,32).round(),available=torch.ones(2,2))
        before=model.head.weight.detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        loss=model.objective(b,model(b))
        assert torch.isfinite(loss)
        loss.backward()
        assert model.encoder[0][0].weight.grad.abs().sum()>0
        optimizer.step()
        assert not torch.equal(before,model.head.weight)
        model.eval()
        expected=model(b)['probability'].detach()
        buffer=io.BytesIO()
        torch.save(model.state_dict(),buffer)
        buffer.seek(0)
        restored=build_model(c).eval()
        restored.load_state_dict(torch.load(buffer,weights_only=True))
        torch.testing.assert_close(expected,restored(b)['probability'],atol=1e-6,rtol=1e-6)
    rng=np.random.default_rng(13)
    data=dict(x=rng.uniform(0,1,(2,7,32,32)).astype(np.float32),y=np.zeros((2,1,32,32),np.float32),available=np.ones((2,2),np.float32))
    state=dict(optical_mean=[.2,.3,.4],sar_std=1.,sar_mean=1.,sar_lower=0.,sar_upper=3.)
    cloud=stress(data,state,'cloud',.2)
    assert (cloud['quality'][:,0]==0).sum()==2*round(.2*32*32)
    other=stress(dict(data,y=np.ones_like(data['y'])),state,'cloud',.2)
    np.testing.assert_array_equal(cloud['x'],other['x'])
    np.testing.assert_array_equal(stress(data,state,'shift',0)['x'],data['x'])
    assert np.isfinite(stress(data,state,'speckle',4)['x']).all()
    for kind,level in [('cloud',.4),('shift',6),('speckle',2)]:
        altered=stress(data,state,kind,level)
        np.testing.assert_array_equal(altered['x'],stress(data,state,kind,level)['x'])
    frame=pd.DataFrame([dict(method=method,event=event,seed=seed,f1=.7 if method=='full' else .5)
                        for method in ['full','baseline'] for event in range(5) for seed in range(5)])
    statistics=event_statistics(frame,'full',draws=1000)
    result=statistics['comparisons'][0]
    assert np.isclose(result['reference_minus_method'],.2) and np.isclose(result['ci_low'],.2) and np.isclose(result['ci_high'],.2)
    assert np.isclose(result['p_raw'],2/32)
    out=Path('runs/protocol_check')
    out.mkdir(parents=True,exist_ok=True)
    (out/'statistics.json').write_text(json.dumps(statistics,allow_nan=False,indent=2),encoding='utf-8')
    print('PASS: halo centers cover odd-sized scene exactly, U-Net backward/update/reload, fixed stresses independent of labels, five-event/five-seed bootstrap and exact p=2/32')


if __name__=='__main__':
    main()
