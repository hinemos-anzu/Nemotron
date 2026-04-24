# Stub: selective_scan_cuda symbols — Mamba-1 API, not called by NemotronH.
#
# NemotronH (modeling_nemotron_h.py) is a Mamba-2 based model. It uses:
#   ops.triton.selective_state_update.selective_state_update
#   ops.triton.ssd_combined.mamba_chunk_scan_combined
# These are loaded only when config.use_mamba_kernels=True, which is forced
# False by the is_mamba_2_ssm_available = lambda: False patch in the kaggle
# script. With that patch, is_fast_path_available=False and cuda_kernels_forward
# is never entered — so these Mamba-1 ops are unreachable during inference.
#
# Stubs raise RuntimeError on call so that static validation passes while
# still failing loudly if somehow invoked at runtime.


def selective_scan_fn(*args, **kwargs):
    raise RuntimeError(
        "selective_scan_fn called — Mamba-1 CUDA kernel stub. "
        "This should be unreachable: is_mamba_2_ssm_available is patched False "
        "so use_mamba_kernels=False and is_fast_path_available=False. "
        "Check that the is_mamba_2_ssm_available patch was applied before model load."
    )


def mamba_inner_fn(*args, **kwargs):
    raise RuntimeError(
        "mamba_inner_fn called — Mamba-1 CUDA kernel stub. "
        "This should be unreachable: is_mamba_2_ssm_available is patched False "
        "so use_mamba_kernels=False and is_fast_path_available=False. "
        "Check that the is_mamba_2_ssm_available patch was applied before model load."
    )


def selective_state_update(*args, **kwargs):
    raise RuntimeError(
        "selective_state_update called — Mamba-1 CUDA kernel stub. "
        "This should be unreachable: is_mamba_2_ssm_available is patched False "
        "so use_mamba_kernels=False and is_fast_path_available=False. "
        "Check that the is_mamba_2_ssm_available patch was applied before model load."
    )
