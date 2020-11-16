from functools import lru_cache
import collections.abc
import numpy as np

model_config=None
def getk(i):
    k=i
    nblock_per_stage=(model_config['depth']-2)//6
    i=(k-1)//(nblock_per_stage*2)
    return model_config["stage%d"%(i+1)]['k%ds'%((k+1)%2+1)][((k-1)%(nblock_per_stage*2))//2]

def gets(i):
    k=i
    if k%2==1:
        return 1
    nblock_per_stage=(model_config['depth']-2)//6
    i=(k-1)//(nblock_per_stage*2)
    if (((k-1)%(nblock_per_stage*2))//2 + 1) in set(model_config["stage%d"%(i+1)]['maxpool']):
        return 2
    return 1
@lru_cache(maxsize=None)
def get_maxrf(i):
    if i==0:
        return 2,5 # starting RF
    s,rf=get_maxrf(i-1)
    s=s*gets(i)
    rf= rf+ (getk(i)-1)*s
    return s,rf



def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def count_parameters(model, trainable=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == trainable)

def count_non_zero_params(model):
    sum_params = 0
    for p in model.parameters():
        if p.requires_grad:
            sum_params += p[p != 0].numel()
    return sum_params


def linear_rampup(current, rampup_length=16):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def customsigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    current = current

    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-1 * (phase ** 3)))


