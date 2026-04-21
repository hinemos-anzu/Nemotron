# Local stub: replaces broken Kaggle system mamba_ssm (ABI-incompatible selective_scan_cuda).
# is_mamba_2_ssm_available() is patched to False before any model import, so the
# mamba2 code paths in modeling_nemotron_h.py are never entered.  This stub only
# needs to be importable without raising; no real CUDA ops are exercised.

__version__ = "2.2.2.stub"
