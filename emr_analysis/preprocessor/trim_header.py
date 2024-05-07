from typing import List

from emr_analysis.abc.preprocessor import Preprocessor


class TrimHeaderPreprocessor(Preprocessor):
    default_options = {"header_lines": 2}

    def preprocess(self, rows: List[List[str]]) -> List[List[str]]:
        return rows[self.options["header_lines"]:]
