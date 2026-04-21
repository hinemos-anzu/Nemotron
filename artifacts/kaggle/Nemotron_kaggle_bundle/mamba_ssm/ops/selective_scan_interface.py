# Stub: selective_scan_cuda is ABI-incompatible on Kaggle.
# These symbols are imported at class-definition time in some transformers
# model files.  Providing no-op stubs prevents ImportError while the real
# CUDA kernel is never called (model uses use_mamba_kernels=False path).


def selective_scan_fn(*args, **kwargs):
    raise NotImplementedError("selective_scan_fn: stub only — should not be called")


def mamba_inner_fn(*args, **kwargs):
    raise NotImplementedError("mamba_inner_fn: stub only — should not be called")


def selective_state_update(*args, **kwargs):
    raise NotImplementedError("selective_state_update: stub only — should not be called")
