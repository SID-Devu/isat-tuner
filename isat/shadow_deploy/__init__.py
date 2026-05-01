"""Shadow deployment: run new models alongside production, compare quality, auto-promote with statistical rigor."""

from isat.shadow_deploy.shadow import (
    AutoPromoter,
    ComparisonMetrics,
    DeploymentReport,
    ModelVersion,
    QualityTracker,
    ShadowDeployment,
    ShadowResult,
    ShadowRunner,
    shadow_deploy,
)

__all__ = [
    "AutoPromoter",
    "ComparisonMetrics",
    "DeploymentReport",
    "ModelVersion",
    "QualityTracker",
    "ShadowDeployment",
    "ShadowResult",
    "ShadowRunner",
    "shadow_deploy",
]
