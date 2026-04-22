# Stub: triton layernorm_gated kernel — not called when use_mamba_kernels=False.
# All symbols that modeling_nemotron_h.py imports at module level must exist
# here so that 'from mamba_ssm.ops.triton.layernorm_gated import ...' succeeds.
# The actual functions are never invoked because the mamba2 code paths are
# disabled via the is_mamba_2_ssm_available() -> False patch.


def _stub(*args, **kwargs):
    raise NotImplementedError("mamba_ssm stub — should not be called")


# Function names that appear in nemotron/mamba model files
rmsnorm_fn = _stub        # reported missing: ImportError fix
rms_norm_fn = _stub       # alias variant
layer_norm_fn = _stub


class RMSNorm:
    """Stub class — not instantiated when mamba kernels are disabled."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RMSNorm stub — should not be instantiated")


class LayerNorm:
    """Stub class — not instantiated when mamba kernels are disabled."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LayerNorm stub — should not be instantiated")
