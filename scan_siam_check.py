"""Check official FC-Siam and ChangeMamba with explicit reference scan kernels."""
import argparse
import gc
import io
import json

import torch

from reliable_change.baselines import build_model
from reliable_change.reference_scan import selective_scan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fc-siam-root',default='.venv/FC-Siam')
    p.add_argument('--changemamba-root',default='.venv/MambaCD')
    p.add_argument('--glcd-da-root',default='.venv/GLCD-DA')
    p.add_argument('--architectures',nargs='+',default=['fc_siam_diff','fc_siam_conc','changemamba','glcd_da_cd'])
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(13)
    u=torch.ones(1,1,2,requires_grad=True)
    result=selective_scan(u,torch.ones_like(u),-torch.log(torch.tensor([[2.]])),torch.ones(1,1,1,2),torch.ones(1,1,1,2),delta_softplus=False)
    torch.testing.assert_close(result,torch.tensor([[[1.,1.5]]]))
    result.sum().backward()
    torch.testing.assert_close(u.grad,torch.tensor([[[1.5,1.]]]))
    for name,settings in [('fc_siam_diff',dict(fc_siam_root=a.fc_siam_root)),
                          ('fc_siam_conc',dict(fc_siam_root=a.fc_siam_root)),
                          ('changemamba',dict(changemamba_root=a.changemamba_root,changemamba_backend='reference')),
                          ('glcd_da_cd',dict(glcd_da_root=a.glcd_da_root))]:
        if name not in a.architectures:
            continue
        config=dict(architecture=name,**settings)
        model=build_model(config)
        size=64 if name=='glcd_da_cd' else 32
        batch=dict(x=torch.rand(2,7,size,size),y=torch.rand(2,1,size,size).round(),available=torch.tensor([[1.,1.],[0.,1.]]))
        parameter=next(model.network.parameters())
        before=parameter.detach().clone()
        optimizer=torch.optim.SGD(model.parameters(),lr=.001)
        loss=model.objective(batch,model(batch))
        loss.backward()
        assert torch.isfinite(loss) and parameter.grad.abs().sum()>0
        optimizer.step()
        assert not torch.equal(parameter,before)
        model.eval()
        with torch.no_grad():
            expected=model(batch)['probability']
        buffer=io.BytesIO()
        torch.save(model.state_dict(),buffer)
        buffer.seek(0)
        state=torch.load(buffer,weights_only=True)
        loss_value=loss.item()
        del model,optimizer,parameter,before,loss
        gc.collect()
        restored=build_model(config).eval()
        restored.load_state_dict(state)
        with torch.no_grad():
            torch.testing.assert_close(restored(batch)['probability'],expected,atol=1e-6,rtol=1e-6)
        print(json.dumps(dict(architecture=name,loss=loss_value,parameters=sum(v.numel() for v in restored.parameters()),gradient_update_reload='PASS')),flush=True)
        del restored,state,buffer
        gc.collect()


if __name__=='__main__':
    main()
