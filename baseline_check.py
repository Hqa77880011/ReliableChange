"""Exercise the exact external ChangeFormer encoder/decoder and shared trainer."""
import argparse
import io
import json
import sys
from pathlib import Path

import torch

from reliable_change.baselines import build_model
from reliable_change.cli import main as cli_main


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--changeformer-root',required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(13)
    c=dict(architecture='changeformer',changeformer_root=a.changeformer_root,variant='full',augment=False)
    m=build_model(c)
    b=dict(x=torch.rand(2,7,64,64),y=torch.rand(2,1,64,64).round(),available=torch.ones(2,2))
    before=m.network.Tenc_x2.patch_embed1.proj.weight.detach().clone()
    optimizer=torch.optim.AdamW(m.parameters(),lr=.00006)
    loss=m.objective(b,m(b))
    assert torch.isfinite(loss)
    loss.backward()
    assert m.network.Tenc_x2.patch_embed1.proj.weight.grad.abs().sum()>0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for name,parameter in m.named_parameters() if 'make_pred' in name)
    optimizer.step()
    assert not torch.equal(before,m.network.Tenc_x2.patch_embed1.proj.weight)
    m.eval()
    expected=m(b)['probability'].detach()
    buffer=io.BytesIO()
    torch.save(m.state_dict(),buffer)
    buffer.seek(0)
    restored=build_model(c).eval()
    restored.load_state_dict(torch.load(buffer,weights_only=True))
    torch.testing.assert_close(expected,restored(b)['probability'],atol=1e-6,rtol=1e-6)
    out=Path('runs/baseline_check')
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(dict(architecture='unet',variant='full',widths=[4,8,12,16])),encoding='utf-8')
    for argv in [
        ['train','--input','runs/smoke/prepared','--output',str(out/'unet'),'--config',str(out/'config.json'),'--epochs','1','--batch-size','2'],
        ['evaluate','--input','runs/smoke/prepared/test.npz','--output',str(out/'unet/evaluation'),'--model',str(out/'unet/model.pt')],
        ['infer','--input','runs/smoke/unlabeled.npz','--output',str(out/'unet/inference'),'--model',str(out/'unet/model.pt')],
    ]:
        sys.argv=['reliable-change']+argv
        cli_main()
    print('PASS official ChangeFormerV6 nine-channel projections/all deep-supervision heads/backward/update/reload and shared U-Net training/eval/infer')


if __name__=='__main__':
    main()
