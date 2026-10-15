import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder,boundary,image_demo


def lovasz(probability,target):
    errors=(target-probability).abs().flatten();order=errors.argsort(descending=True);truth=target.flatten()[order]
    total=truth.sum();intersection=total-truth.cumsum(0);union=total+(1-truth).cumsum(0)
    gradient=1-intersection/union.clamp_min(1);gradient=torch.cat([gradient[:1],gradient[1:]-gradient[:-1]])
    return errors[order]@gradient


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',24)
        self.optical=Encoder(3,w);self.sar=Encoder(1,w);self.difference=nn.Conv2d(4*w,w,1)
        self.quality=nn.Conv2d(4*w,1,1);self.head=nn.Conv2d(w,1,1);self.edge=nn.Conv2d(w,1,1)
    def align(self,reference,post):
        if self.c['variant'] in {'baseline','no_alignment'}:return post
        radius=self.c.get('alignment_radius',1);k=2*radius+1
        patches=F.unfold(post,k,padding=radius).reshape(post.shape[0],post.shape[1],k*k,*post.shape[-2:])
        correlation=(F.normalize(reference,dim=1)[:,:,None]*F.normalize(patches,dim=1)).sum(1)
        return (correlation.softmax(1)[:,None]*patches).sum(2)
    def fused(self,b,available):
        x=b['x'].float();reference=self.optical(x[:,:3]);branches=[self.optical(x[:,3:6]),self.sar(x[:,6:7])]
        difference=[];quality=[]
        for post in branches:
            post=self.align(reference,post);d=torch.cat([reference,post,post-reference,abs(post-reference)],1)
            difference.append(self.difference(d));quality.append(self.quality(d).sigmoid())
        quality=torch.stack(quality,1);valid=available[:,:,None,None,None].float()
        if self.c['variant'] in {'baseline','no_quality'}:quality=torch.ones_like(quality)
        weights=valid*(quality+1e-6);weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-6)
        fused=(weights*quality*torch.stack(difference,1)).sum(1)
        return self.head(fused).sigmoid(),self.edge(fused).sigmoid(),quality.squeeze(2),fused
    def forward(self,b):
        availability=b['available'];p,edge,q,feature=self.fused(b,availability)
        masked=availability.clone();both=(masked.sum(1)>1);masked[both,1]=0
        pm,_,_,_=self.fused(b,masked)
        return {'probability':p,'boundary':edge,'quality':q,'feature':feature,'masked_probability':pm}
    def objective(self,b,p):
        y=b['y'].float();prob=p['probability'];loss=F.binary_cross_entropy(prob,y)+lovasz(prob,y)
        if self.c['variant']=='baseline':return loss
        loss=loss+.2*F.binary_cross_entropy(p['boundary'],boundary(y,1))
        distance=p['feature'].abs().mean(1,keepdim=True)
        loss=loss+.3*((1-y)*distance+y*F.relu(1-distance)).mean()
        if self.c['variant']!='no_mask_consistency':loss=loss+.2*F.binary_cross_entropy(p['masked_probability'],prob.detach())
        if 'quality' in b:loss=loss+.1*((p['quality']-b['quality']).square()*b['available'][:,:,None,None]).mean()
        return loss


def demo(seed,n,c):
    d=image_demo(seed,n,7);d['available']=np.ones((n,2),np.float32);d['quality']=np.ones((n,2,32,32),np.float32);return d
