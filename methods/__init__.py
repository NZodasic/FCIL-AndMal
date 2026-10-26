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
from methods.falcon import FalconMethod


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
    elif m_name == "malfscil":
        return MalFSCILMethod(
            vae_weight=config.malfscil_vae_weight,
            kl_weight=config.malfscil_kl_weight,
            arc_weight=config.malfscil_arc_weight,
            arc_scale=config.malfscil_arc_scale,
            arc_margin=config.malfscil_arc_margin,
            graph_attention_dim=config.malfscil_graph_attention_dim,
        )
    elif m_name == "malfsil":
        return MALFSILMethod(
            buffer_size_per_class=config.replay_buffer_size_per_class,
            distill_weight=getattr(config, "malfsil_distill_weight", 1.0),
            proto_weight=getattr(config, "malfsil_proto_weight", 0.5),
            temperature=getattr(config, "lwf_temperature", 2.0),
        )
    elif m_name == "falcon":
        return FalconMethod(
            buffer_size_per_class=getattr(config, "replay_buffer_size_per_class", 20),
            replay_ratio=getattr(config, "falcon_replay_ratio", 0.2),
            alpha_cls=getattr(config, "falcon_alpha", 1.0),
            beta_old=getattr(config, "falcon_beta", 1.0),
            gamma_new=getattr(config, "falcon_gamma", 0.5),
            temperature=getattr(config, "falcon_temperature", 2.0),
            focal_gamma=getattr(config, "focal_gamma", 2.0),
            focal_alpha=getattr(config, "focal_alpha", 1.0),
            use_herding=getattr(config, "herding", True),
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
    "FalconMethod",
    "build_il_method",
]
