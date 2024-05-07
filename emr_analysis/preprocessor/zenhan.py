from typing import List

import mojimoji

from emr_analysis.abc.preprocessor import Preprocessor


class ZenhanPreprocessor(Preprocessor):
    default_options = {"kana": True, "digit": False, "ascii": False}

    def preprocess(self, rows: List[List[str]]) -> List[List[str]]:
        han_to_zen_options = self.options
        rows = [[mojimoji.han_to_zen(s, **han_to_zen_options) for s in row] for row in rows]

        zen_to_han_options = {k: not v for k, v in self.options.items()}
        rows = [[mojimoji.zen_to_han(s, **zen_to_han_options) for s in row] for row in rows]

        return rows
