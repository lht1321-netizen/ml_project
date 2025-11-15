from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import copy
import inspect
import functools
from fvcore.common.config import CfgNode as _CfgNode

# @title ShapeSpec (pixeldecoder_tem_fpn.ipynb)
class ShapeSpec:
    """
    (backbone과 pixeldecoder에서 모두 사용)
    """
    def __init__(self, channels=None, height=None, width=None, stride=None):
        self.channels = channels
        self.height = height
        self.width = width
        self.stride = stride

    def __str__(self) -> str:
        return f"ShapeSpec(C={self.channels}, H={self.height}, W={self.width}, S={self.stride})"

    __repr__ = __str__

# @title _get_clones (pixeldecoder_tem_fpn.ipynb)
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# @title _get_activation_fn (pixeldecoder_tem_fpn.ipynb)
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

# @title get_norm (pixeldecoder_tem_fpn.ipynb)
def get_norm(norm, out_channels):
    if norm is None:
        return None
    if isinstance(norm, str):
        if norm == "" or norm.lower() == "none":
            return None
        if norm == "BN":
            return nn.BatchNorm2d(out_channels)
        if norm == "SyncBN":
            return nn.SyncBatchNorm(out_channels)
        if norm == "GN":
            groups = 32 if out_channels % 32 == 0 else max(1, min(32, out_channels))
            return nn.GroupNorm(groups, out_channels)
        if norm == "LN":
            class _ChannelsFirstLayerNorm(nn.Module):
                def __init__(self, num_channels, eps=1e-6):
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(num_channels))
                    self.bias = nn.Parameter(torch.zeros(num_channels))
                    self.eps = eps
                def forward(self, x):
                    mean = x.mean(dim=1, keepdim=True)
                    var = (x - mean).pow(2).mean(dim=1, keepdim=True)
                    x = (x - mean) / torch.sqrt(var + self.eps)
                    return x * self.weight[:, None, None] + self.bias[:, None, None]
            return _ChannelsFirstLayerNorm(out_channels)
        raise ValueError(...)
    if callable(norm):
        return norm(out_channels)
    if isinstance(norm, nn.Module):
        return norm
    raise TypeError(...)

# @title Conv2d (pixeldecoder_tem_fpn.ipynb)
class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        norm=None,
        activation=None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias if norm is None else False,
        )
        self.norm = norm
        self.activation = activation

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x) if callable(self.activation) else self.activation(x)
        return x

# @title configurable (pixeldecoder_tem_fpn.ipynb)
# fvcore의 configurable 데코레이터를 위한 헬퍼 함수들
def configurable(init_func=None, *, from_config=None):
    if init_func is not None:
        assert (
            inspect.isfunction(init_func)
            and from_config is None
            and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable. Check API documentation for examples."

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                ) from e
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a 'from_config' classmethod.")

            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)

        return wrapped
    else:
        if from_config is None:
            return configurable
        assert inspect.isfunction(
            from_config
        ), "from_config argument of configurable must be a function!"

        def wrapper(orig_func):
            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)

            wrapped.from_config = from_config
            return wrapped

        return wrapper

def _get_args_from_config(from_config_func, *args, **kwargs):
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "cfg":
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take 'cfg' as the first argument!")
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg:
        ret = from_config_func(*args, **kwargs)
    else:
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        ret.update(extra_kwargs)
    return ret


def _called_with_cfg(*args, **kwargs):
    # omegaconf는 설치가 필요할 수 있습니다.
    try:
        from omegaconf import DictConfig
    except ImportError:
        # 설치되지 않았을 경우를 대비한 임시 클래스
        class DictConfig:
            pass

    if len(args) and isinstance(args[0], (_CfgNode, DictConfig)):
        return True
    if isinstance(kwargs.pop("cfg", None), (_CfgNode, DictConfig)):
        return True
    return False