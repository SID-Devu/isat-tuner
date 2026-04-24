from isat.analysis.outliers import detect_outliers, remove_outliers
from isat.analysis.significance import compare_configs, WelchResult
from isat.analysis.pareto import ParetoFrontier
from isat.analysis.regression import RegressionDetector

__all__ = [
    "detect_outliers", "remove_outliers",
    "compare_configs", "WelchResult",
    "ParetoFrontier",
    "RegressionDetector",
]
