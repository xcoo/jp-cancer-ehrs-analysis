from typing import List

from emr_analysis.abc.preprocessor import Preprocessor


class ConvertNewlinePreprocessor(Preprocessor):
    def preprocess(self, rows: List[List[str]]) -> List[List[str]]:
        return [[s.replace("\n", "\\n") for s in row] for row in rows]
