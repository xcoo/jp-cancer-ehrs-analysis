import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from emr_analysis.data import Effect  # noqa: F401
from emr_analysis.tokenizer import StanzaTokenizer, simple_tokenize


class ReportDataset(Dataset):
    def __init__(self, data, config, patient_ids, report_size=None, token_size=None,
                 max_reports=10, max_tokens=500,
                 effect_excludes={Effect.UNKNOWN, Effect.UNDETECTED}):
        self.emr_radiation_report = data.emr_radiation_report
        self.fm_treatment_record = data.fm_treatment_record
        self.word_embedding = data.word_embedding
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(self.word_embedding.vectors))

        self.report_size = report_size
        self.token_size = token_size

        self.effect_excludes = effect_excludes
        effects = [e for e in Effect if e not in effect_excludes]
        self.int2effect_dict = dict(enumerate(effects))
        self.effect2int_dict = {v: k for k, v in self.int2effect_dict.items()}
        self.effect_size = len(effects)

        self.tokenizer = StanzaTokenizer()

        self.dataset = self._create_training_dataset(patient_ids, max_reports=max_reports,
                                                     max_tokens=max_tokens)
        self.unk_index = self.word_embedding.get_index("<unk>")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        x = self.embedding(torch.tensor([[self.word_embedding.get_index(t, self.unk_index)
                                          for t in ts]
                                         for ts in item["tokens"]]))
        y0 = torch.tensor(self.effect2int(item["effect"]))
        y1 = torch.tensor([ReportDataset.pd2int(b) for b in item["pd_confirms"]])

        return x, y0, y1

    def effect2int(self, effect):
        return self.effect2int_dict[effect]

    def int2effect(self, x):
        return self.int2effect_dict[x]

    @staticmethod
    def pd2int(x):
        if x is None:
            return 0
        elif x:
            return 1
        else:
            return 2

    @staticmethod
    def int2pd(x):
        if x == 0:
            return None
        elif x == 1:
            return True
        elif x == 2:
            return False

    @staticmethod
    def fetch_reports(report_df, patient_id, start_dt, end_dt):
        return report_df.query("PatientNo == @patient_id") \
            .query(f"'{start_dt}' <= ExaminDateTime and "
                   f"ExaminDateTime < '{end_dt + datetime.timedelta(days=1)}'")

    @staticmethod
    def thinout_reports(reports, size, pd_date=None):
        reports = reports.sort_values(by="ExaminDateTime")
        if len(reports) <= size:
            return reports
        if pd_date is not None:
            pd_date = pd_date.date()
        df = pd.DataFrame(columns=reports.columns)
        i = 0
        step = (len(reports) - 1) / size
        for j, (_, report) in enumerate(reports.iterrows()):
            if len(df) >= size - 1:
                break
            if report["ExaminDateTime"].date() == pd_date or j == int(i):
                df.loc[len(df)] = report
                i += step
        df.loc[len(df)] = reports.iloc[len(reports) - 1]
        return df

    @staticmethod
    def report_to_text(report) -> str:
        return " ".join([report["Findings"], report["Impression"]])

    @staticmethod
    def fill_padding(tokens: list[str], size: int, side: str = "left") -> list[str]:
        if len(tokens) > size:
            return tokens[:size]
        else:
            if side == "left":
                return ["<pad>"] * (size - len(tokens)) + tokens
            elif side == "right":
                return tokens + ["<pad>"] * (size - len(tokens))
            else:
                raise ValueError("irregal side")

    def _fill_padding(self, dataset) -> None:
        if self.report_size is None:
            self.report_size = max([len(x["tokens"]) for x in dataset])
        if self.token_size is None:
            self.token_size = max([max([len(t) for t in x["tokens"]]) for x in dataset])
        for x in dataset:
            if len(x["texts"]) > self.report_size:
                x["texts"] = x["texts"][:self.report_size]
                x["tokens"] = x["tokens"][:self.report_size]
                x["pd_confirms"] = x["pd_confirms"][:self.report_size]
                x["report_dates"] = x["report_dates"][:self.report_size]
            else:
                x["texts"] = [""] * (self.report_size - len(x["texts"])) + x["texts"]
                x["tokens"] = [[]] * (self.report_size - len(x["tokens"])) + x["tokens"]
                x["pd_confirms"] = [None] * (self.report_size - len(x["pd_confirms"])) \
                    + x["pd_confirms"]
                x["report_dates"] = [None] * (self.report_size - len(x["report_dates"])) \
                    + x["report_dates"]
            x["tokens"] = [ReportDataset.fill_padding(t, self.token_size)
                           for t in x["tokens"]]

    def _pd_report_exists(self, record):
        patient_id = record["患者ID"]  # noqa: F841
        pd_dt = record["PD確定日"]
        if pd.isnull(pd_dt):
            return True
        delta = datetime.timedelta(days=1)
        return not self.emr_radiation_report.query("PatientNo == @patient_id") \
            .query(f"'{pd_dt - delta}' <= ExaminDateTime and " +
                   f"ExaminDateTime < '{pd_dt + delta}'").empty

    def _create_training_dataset(self, patient_ids, max_reports=10, max_tokens=500):
        dataset = []
        for patient_id in patient_ids:
            records = self.fm_treatment_record.query("患者ID == @patient_id") \
                .query("治療内容 == '薬物療法'")
            for i in range(len(records) - 1):
                start_dt = records.iloc[i]["治療開始日"]
                next_start_dt = records.iloc[i + 1]["治療開始日"]
                pd_dt = records.iloc[i]["PD確定日"]
                effect = records.iloc[i]["効果"]

                reports = ReportDataset.fetch_reports(self.emr_radiation_report, patient_id,
                                                      start_dt, next_start_dt)

                if pd.isnull(start_dt) or \
                   not self._pd_report_exists(records.iloc[i]) or \
                   effect in self.effect_excludes or \
                   reports.empty:
                    continue

                if pd.isnull(pd_dt):
                    if effect == Effect.PD:
                        continue
                    pd_dt = datetime.datetime(1970, 1, 1)
                reports = ReportDataset.thinout_reports(reports, max_reports, pd_dt)
                texts = [ReportDataset.report_to_text(report)
                         for _, report in reports.iterrows()]
                tokens = [simple_tokenize(self.tokenizer, text)[:max_tokens] for text in texts]
                delta = datetime.timedelta(days=1)
                pd_confirms = [report["ExaminDateTime"] - delta <= pd_dt and
                               pd_dt <= report["ExaminDateTime"] + delta
                               for _, report in reports.iterrows()]
                dataset.append({
                    "patient_id": patient_id,
                    "start_date": start_dt,
                    "report_dates": list(reports["ExaminDateTime"]),
                    "texts": texts,
                    "tokens": tokens,
                    "effect": effect,
                    "pd_date": pd_dt,
                    "pd_confirms": pd_confirms
                })

        self._fill_padding(dataset)

        return dataset
