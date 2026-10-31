"""Dense baselines sharing observed channels, availability masks and region loss."""
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .method import block, lovasz, Model


class EarlyFusionUNet(nn.Module):
    """Four-level U-Net over seven observed channels and two availability planes."""
    def __init__(self,c):
        super().__init__()
        self.c=c
        widths=c.get('widths',[64,128,256,512])
        self.encoder=nn.ModuleList([block(9 if i==0 else widths[i-1],w) for i,w in enumerate(widths)])
        self.decoder=nn.ModuleList([block(widths[i]+widths[i+1],widths[i]) for i in range(3)])
        self.head=nn.Conv2d(widths[0],1,1)

    def forward(self,b):
        x=b['x'].float().clone()
        available=sample_availability(b['available'],self.training)
        x[:,3:6]*=available[:,0,None,None,None]
        x[:,6:7]*=available[:,1,None,None,None]
        z=torch.cat([x,available[:,:,None,None].expand(-1,-1,*x.shape[-2:])],1)
        skips=[]
        for i,encoder in enumerate(self.encoder):
            z=encoder(F.max_pool2d(z,2) if i else z)
            skips.append(z)
        for i in [2,1,0]:
            z=self.decoder[i](torch.cat([F.interpolate(z,skips[i].shape[-2:],mode='bilinear',align_corners=False),skips[i]],1))
        return {'probability':self.head(z).sigmoid()}

    def objective(self,b,p):
        return region_objective(b,p,self.c)


def sample_availability(available,training):
    result=available.float().clone()
    if training:
        full=result.sum(1)==2
        r=torch.rand(len(result),device=result.device)
        result[full & (r>=.5) & (r<.75),0]=0
        result[full & (r>=.75),1]=0
    return result


def region_objective(batch,prediction,c):
    target=batch['y'].float()
    probability=prediction['probability'].clamp(1e-6,1-1e-6)
    weight=torch.where(target>.5,c.get('positive_weight',1.),1.)
    return F.binary_cross_entropy(probability,target,weight=weight)+lovasz(probability,target)


class ChangeFormer(nn.Module):
    """Use the published V6 temporal/decoder blocks with a nine-channel projection."""
    def __init__(self,c):
        super().__init__()
        self.c=c
        root=Path(c['changeformer_root']).resolve()
        if not (root/'models/ChangeFormer.py').is_file():
            raise FileNotFoundError('changeformer_root must be the official wgcban/ChangeFormer checkout')
        sys.path.insert(0,str(root))
        from models.ChangeFormer import ChangeFormerV6
        self.network=ChangeFormerV6(input_nc=9,output_nc=2,embed_dim=256)

    def forward(self,b):
        x=b['x'].float()
        available=sample_availability(b['available'],self.training)
        planes=available[:,:,None,None].expand(-1,-1,*x.shape[-2:])
        first=torch.cat([x[:,:3],torch.zeros_like(x[:,3:7]),planes],1)
        second=torch.cat([torch.zeros_like(x[:,:3]),x[:,3:6]*available[:,0,None,None,None],x[:,6:7]*available[:,1,None,None,None],planes],1)
        result=self.network(first,second)
        logits=result[-1] if isinstance(result,(list,tuple)) else result
        logits=F.interpolate(logits,x.shape[-2:],mode='bilinear',align_corners=False)
        prediction={'probability':logits.softmax(1)[:,1:2]}
        if isinstance(result,(list,tuple)):
            prediction['auxiliary_probabilities']=[F.interpolate(value,x.shape[-2:],mode='bilinear',align_corners=False).softmax(1)[:,1:2] for value in result[:-1]]
        return prediction

    def objective(self,b,p):
        loss=region_objective(b,p,self.c)
        if self.c.get('deep_supervision',True):
            for probability in p.get('auxiliary_probabilities',[]):
                loss=loss+region_objective(b,{'probability':probability},self.c)
        return loss


class ChangeMamba(ChangeFormer):
    """Published ChangeMamba-BCD temporal blocks and decoder with observed inputs."""
    def __init__(self,c):
        nn.Module.__init__(self)
        self.c=c
        root=Path(c['changemamba_root']).resolve()
        if not (root/'changedetection/models/ChangeMambaBCD.py').is_file():
            raise FileNotFoundError('changemamba_root must point to ChenHongruixuan/MambaCD')
        sys.path.insert(0,str(root))
        from changedetection.configs.config import get_config
        from changedetection.script.script_utils import get_vssm_kwargs
        from changedetection.models.ChangeMambaBCD import ChangeMambaBCD
        path=c.get('changemamba_config',str(root/'changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml'))
        config=get_config(SimpleNamespace(cfg=path,opts=None))
        kwargs=get_vssm_kwargs(config)
        kwargs['in_chans']=9
        self.network=ChangeMambaBCD(pretrained=c.get('pretrained_encoder'),**kwargs)


def build_model(c):
    """Select a concrete architecture; never substitute a missing baseline."""
    name=c.get('architecture','reliable_change')
    if name=='reliable_change':
        return Model(c)
    if name=='unet':
        return EarlyFusionUNet(c)
    if name=='changeformer':
        return ChangeFormer(c)
    if name=='changemamba':
        return ChangeMamba(c)
    raise ValueError('Unknown architecture: '+name)
