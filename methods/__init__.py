"""
Class-Incremental Learning Methods for FCIL on CIC-AndMal-2020.
"""

from config import ILConfig
from methods.base import BaseILMethod
from methods.finetune import FineTuneMethod, JointCumulativeMethod
from methods.ewc import EWCMethod
from methods.lwf import LwFMethod
from methods.replay import ReplayHerdingMethod
from methods.spcil import SPCILMethod
from methods.malfscil import MalFSCILMethod
from methods.malfsil import MALFSILMethod


def build_il_method(config: ILConfig) -> BaseILMethod:
    """Factory builder for Continual Learning algorithms."""
    m_name = config.method_name.lower()
    if m_name == "finetune":
        return FineTuneMethod()
    elif m_name == "joint":
        return JointCumulativeMethod()
    elif m_name == "ewc":
        return EWCMethod(ewc_lambda=config.ewc_lambda)
    elif m_name == "lwf":
        return LwFMethod(temperature=config.lwf_temperature, alpha=config.lwf_alpha)
    elif m_name == "replay":
        return ReplayHerdingMethod(buffer_size_per_class=config.replay_buffer_size_per_class, use_herding=config.herding)
    elif m_name == "spcil":
        return SPCILMethod(lambda_init=config.spcil_lambda_init, lambda_step=config.spcil_lambda_step)
    elif m_name in {"malfscil", "malfsil"}:
        return MalFSCILMethod(
            vae_weight=config.malfscil_vae_weight,
            kl_weight=config.malfscil_kl_weight,
            arc_weight=config.malfscil_arc_weight,
            arc_scale=config.malfscil_arc_scale,
            arc_margin=config.malfscil_arc_margin,
            graph_attention_dim=config.malfscil_graph_attention_dim,
        )
    else:
        raise ValueError(f"Unknown IL method name: {config.method_name}")


__all__ = [
    "BaseILMethod",
    "FineTuneMethod",
    "JointCumulativeMethod",
    "EWCMethod",
    "LwFMethod",
    "ReplayHerdingMethod",
    "SPCILMethod",
    "MalFSCILMethod",
    "MALFSILMethod",
    "build_il_method",
]
