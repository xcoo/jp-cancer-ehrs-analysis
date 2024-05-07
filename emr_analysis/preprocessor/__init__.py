from typing import Any, Dict, Mapping, Type

from emr_analysis.abc.preprocessor import Preprocessor
from emr_analysis.preprocessor.convert_newline import ConvertNewlinePreprocessor
from emr_analysis.preprocessor.strip import StripPreprocessor
from emr_analysis.preprocessor.trim_header import TrimHeaderPreprocessor
from emr_analysis.preprocessor.zenhan import ZenhanPreprocessor


__all__ = [
    "ConvertNewlinePreprocessor",
    "StripPreprocessor",
    "TrimHeaderPreprocessor",
    "ZenhanPreprocessor"
]


mapping: Mapping[str, Type[Preprocessor]] = {
    "convert-newline": ConvertNewlinePreprocessor,
    "strip": StripPreprocessor,
    "trim-header": TrimHeaderPreprocessor,
    "zenhan": ZenhanPreprocessor
}


def create_preprocessor(name: str, options: Dict[str, Any]) -> Preprocessor:
    return mapping[name](options)
