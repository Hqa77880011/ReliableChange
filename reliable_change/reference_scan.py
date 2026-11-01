"""Differentiable real-valued VMamba recurrence for explicit reference execution."""
import importlib
import sys
from types import ModuleType

import torch
from torch.nn import functional as F


def selective_scan(u,delta,a,b,c,d=None,delta_bias=None,delta_softplus=True):
    """Evaluate the grouped selective state recurrence without CUDA extensions."""
    dtype=u.dtype
    u,delta,a,b,c=[value.float() for value in [u,delta,a,b,c]]
    if delta_bias is not None:
        delta=delta+delta_bias.float()[None,:,None]
    if delta_softplus:
        delta=F.softplus(delta)
    if b.ndim==3:
        b=b[:,None]
    if c.ndim==3:
        c=c[:,None]
    channels=u.shape[1]
    b=b.repeat_interleave(channels//b.shape[1],dim=1)
    c=c.repeat_interleave(channels//c.shape[1],dim=1)
    state=u.new_zeros(u.shape[0],channels,a.shape[-1])
    result=[]
    for t in range(u.shape[-1]):
        step=delta[:,:,t,None]
        state=(step*a[None]).exp()*state+step*b[:,:,:,t]*u[:,:,t,None]
        value=(state*c[:,:,:,t]).sum(-1)
        if d is not None:
            value=value+d.float()[None]*u[:,:,t]
        result.append(value)
    return torch.stack(result,-1).to(dtype)


class ReferenceSelectiveScan:
    @staticmethod
    def apply(u,delta,a,b,c,d,delta_bias,delta_softplus,*kernel_options):
        return selective_scan(u,delta,a,b,c,d,delta_bias,delta_softplus)


class UnsupportedTritonVariant:
    @staticmethod
    def apply(*args):
        raise ValueError('Reference execution supports the official v2/v3 scan; select a matching configuration')


def install():
    """Load the unchanged official architecture with explicit reference kernels."""
    name='changedetection.models.vmamba'
    if name in sys.modules:
        module=sys.modules[name]
        if not getattr(module,'reliable_reference',False):
            raise RuntimeError('Select native/reference ChangeMamba backends in separate Python processes')
        return module
    # Upstream imports Triton even for its own pure-Torch CrossScan path.
    shim=ModuleType('changedetection.models.csm_triton')
    for attribute in ['CrossScanTriton','CrossMergeTriton','CrossScanTriton1b1']:
        setattr(shim,attribute,UnsupportedTritonVariant)
    sys.modules[shim.__name__]=shim
    module=importlib.import_module(name)
    for attribute in ['SelectiveScanMamba','SelectiveScanCore','SelectiveScanOflex']:
        setattr(module,attribute,ReferenceSelectiveScan)
    module.reliable_reference=True
    return module
