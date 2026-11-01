"""Dense baselines sharing observed channels, availability masks and region loss."""
import sys
import importlib.util
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .method import block, lovasz, Model


@contextmanager
def external_imports(root):
    """Isolate author projects which both publish a top-level models package."""
    prefixes=('models','Unet')
    saved={k:v for k,v in sys.modules.items() if k.split('.')[0] in prefixes}
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0,str(root))
    try:
        yield
    finally:
        for key in list(sys.modules):
            if key.split('.')[0] in prefixes:
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path.remove(str(root))


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
        with external_imports(root):
            from models.ChangeFormer import ChangeFormerV6
            self.network=ChangeFormerV6(input_nc=9,output_nc=2,embed_dim=256)

    def forward(self,b):
        x=b['x'].float()
        available=sample_availability(b['available'],self.training)
        planes=available[:,:,None,None].expand(-1,-1,*x.shape[-2:])
        first=torch.cat([x[:,:3],torch.zeros_like(x[:,3:7]),planes],1)
        second=torch.cat([torch.zeros_like(x[:,:3]),x[:,3:6]*available[:,0,None,None,None],x[:,6:7]*available[:,1,None,None,None],planes],1)
        if hasattr(self,'input_projection'):
            first,second=self.input_projection(first),self.input_projection(second)
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
        backend=c.get('changemamba_backend','native')
        if backend=='reference':
            from .reference_scan import install
            install()
        elif backend!='native':
            raise ValueError('changemamba_backend must be native or reference')
        elif getattr(sys.modules.get('changedetection.models.vmamba'),'reliable_reference',False):
            raise RuntimeError('Use a fresh process for the native ChangeMamba backend')
        from changedetection.configs.config import get_config
        from changedetection.script.script_utils import get_vssm_kwargs
        from changedetection.models.ChangeMambaBCD import ChangeMambaBCD
        path=c.get('changemamba_config',str(root/'changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml'))
        config=get_config(SimpleNamespace(cfg=path,opts=None))
        kwargs=get_vssm_kwargs(config)
        kwargs['in_chans']=9
        self.network=ChangeMambaBCD(pretrained=c.get('pretrained_encoder'),**kwargs)


class FCSiam(ChangeFormer):
    """Original FC-Siam difference/concatenation U-Net on observed streams."""
    def __init__(self,c):
        nn.Module.__init__(self)
        self.c=c
        root=Path(c['fc_siam_root']).resolve()
        variant=c['architecture'].removeprefix('fc_siam_')
        filename='siamunet_'+variant
        path=root/(filename+'.py')
        if not path.is_file():
            raise FileNotFoundError('fc_siam_root must point to rcdaudt/fully_convolutional_change_detection')
        spec=importlib.util.spec_from_file_location('_reliable_'+filename,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.network=getattr(module,'SiamUnet_'+variant)(9,2)


class GLCDDetection(ChangeFormer):
    """Author GLCD-DA CD stage with an explicit observed-input projection."""
    def __init__(self,c):
        nn.Module.__init__(self)
        self.c=c
        root=Path(c['glcd_da_root']).resolve()
        if not (root/'models/networks.py').is_file():
            raise FileNotFoundError('glcd_da_root must point to WHU-SGG-RS-Pro-Group/ChangeDetection-GLCD-DA')
        self.input_projection=nn.Conv2d(9,3,1)
        with external_imports(root):
            from models.networks import BASE_Transformer
            from models import laplace
            # Author kernels unconditionally call .cuda() before .to(img.device).
            # Keep the identical coefficients while permitting CPU and nondefault GPUs.
            laplace.gauss_kernel.__defaults__=(3,False)
            self.network=BASE_Transformer(input_nc=3,output_nc=2,with_pos='learned',resnet_stages_num=4)


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
    if name in {'fc_siam_diff','fc_siam_conc'}:
        return FCSiam(c)
    if name=='glcd_da_cd':
        return GLCDDetection(c)
    raise ValueError('Unknown architecture: '+name)
