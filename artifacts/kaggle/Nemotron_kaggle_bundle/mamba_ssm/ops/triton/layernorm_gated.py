# Pure-PyTorch fallback for mamba_ssm layernorm_gated symbols.
#
# The real mamba_ssm package implements these via Triton fused kernels which
# are ABI-incompatible with the Kaggle environment.  This module provides
# numerically correct but un-fused PyTorch equivalents so that
# modeling_nemotron_h.py can complete a forward pass without the Triton kernel.
#
# Signature matches mamba_ssm 2.2.x:
#   rmsnorm_fn / rms_norm_fn / layer_norm_fn
#   RMSNorm / LayerNorm
#
# Key semantics preserved:
#   - residual addition before normalisation (when residual is not None)
#   - prenorm=True returns (normed_output, residual_for_next_layer)
#   - residual_in_fp32 upcasts the residual accumulator to float32
#   - return_dropout_mask returns None mask (dropout always 0 at inference)

import torch
import torch.nn.functional as F


def _rms_norm(x, weight, bias, eps):
    orig_dtype = x.dtype
    x_f = x.float()
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_normed = (x_f * torch.rsqrt(variance + eps)).to(orig_dtype)
    out = x_normed * weight
    if bias is not None:
        out = out + bias
    return out


def _layer_norm(x, weight, bias, eps):
    return F.layer_norm(x, x.shape[-1:], weight, bias, eps)


def rmsnorm_fn(
    x,
    weight,
    bias=None,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    layerscale=None,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
    **kwargs,
):
    if residual is not None:
        x = x + residual.to(x.dtype)
    residual_out = x.float() if residual_in_fp32 else x

    out = _rms_norm(x, weight, bias, eps)

    if prenorm and return_dropout_mask:
        return out, residual_out, None
    if prenorm:
        return out, residual_out
    if return_dropout_mask:
        return out, None
    return out


# mamba_ssm exports both spellings; some model files import one, some the other
rms_norm_fn = rmsnorm_fn


def layer_norm_fn(
    x,
    weight,
    bias=None,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    layerscale=None,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
    **kwargs,
):
    if residual is not None:
        x = x + residual.to(x.dtype)
    residual_out = x.float() if residual_in_fp32 else x

    out = _layer_norm(x, weight, bias, eps)

    if prenorm and return_dropout_mask:
        return out, residual_out, None
    if prenorm:
        return out, residual_out
    if return_dropout_mask:
        return out, None
    return out


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )
        self.variance_epsilon = eps

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rmsnorm_fn(
            x,
            self.weight,
            bias=None,
            residual=residual,
            eps=self.variance_epsilon,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(hidden_size, device=device, dtype=dtype)
        )
        self.variance_epsilon = eps

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_fn(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.variance_epsilon,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )
