from typing import List

from emr_analysis.abc.preprocessor import Preprocessor


class StripPreprocessor(Preprocessor):
    def preprocess(self, rows: List[List[str]]) -> List[List[str]]:
        return [[s.strip() for s in row] for row in rows]
