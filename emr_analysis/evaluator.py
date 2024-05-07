import datetime

import pandas as pd

from emr_analysis.treatment_period import Regimen
import emr_analysis.util as util


class OutOfEvaluationError(Exception):
    def __init__(self, message=None):
        self.message = message


class Evaluator(object):
    def __init__(self, data, config):
        self.data = data
        self.config = config

        self.emr_radiation_report_group = data.emr_radiation_report.groupby("PatientNo")

        regimen = Regimen(config)
        fm_treatment_record = data.fm_treatment_record
        fm_treatment_record["治療詳細"] = \
            data.fm_treatment_record["治療詳細"].map(regimen.normalize)
        self.fm_treatment_record_group = data.fm_treatment_record.groupby("患者ID")

    def corresponding_record_by_period(self, patient_id, start_dt, end_dt, medication=None,
                                       margin_days=(200, 200), regimen_margin_days=(200, 200)):
        if patient_id not in self.fm_treatment_record_group.groups:
            return KeyError(f"{patient_id} does not exist")

        if isinstance(margin_days, int):
            margin_days = (margin_days, margin_days)

        p_fm_treatment_record = self.fm_treatment_record_group.get_group(patient_id)
        p_fm_treatment_record = p_fm_treatment_record.query("治療内容 == '薬物療法'") \
            .query("治療終了日 != 'NaT'")
        p_fm_treatment_record = p_fm_treatment_record.sort_values(by="治療開始日")

        # Merge the same medications separated by another treatment
        df = pd.DataFrame(columns=p_fm_treatment_record.columns)
        prev_row = None
        for _, row in p_fm_treatment_record.iterrows():
            if prev_row is not None and \
               row["治療詳細"] == prev_row["治療詳細"] and \
               row["プロトコールの有無"] == prev_row["プロトコールの有無"]:
                df.loc[df.index[-1]] = row
                df.at[df.index[-1], "治療開始日"] = prev_row["治療開始日"]
            else:
                df.loc[len(df)] = row
                prev_row = row
        p_fm_treatment_record = df

        if p_fm_treatment_record.empty:
            raise OutOfEvaluationError("Out of treatment period")

        start_date = util.date_to_datetime(start_dt.date())
        end_date = util.date_to_datetime(end_dt.date())

        last = p_fm_treatment_record.iloc[-1]
        if end_date < p_fm_treatment_record.iloc[0]["治療開始日"] or \
           last["治療終了日"] < start_date:
            raise OutOfEvaluationError("Out of treatment period")

        delta1 = datetime.timedelta(days=margin_days[0])
        delta2 = datetime.timedelta(days=margin_days[1])

        rdelta1 = datetime.timedelta(days=regimen_margin_days[0])
        rdelta2 = datetime.timedelta(days=regimen_margin_days[1])

        return p_fm_treatment_record.query(
            f"('{start_date - delta1}' <= 治療開始日 and " +
            f"治療開始日 <= '{start_date + delta1}' and " +
            f"'{end_date - delta2}' <= 治療終了日 and 治療終了日 <= '{end_date + delta2}') or" +
            f"(治療詳細 == @medication and"
            f"'{start_date - rdelta1}' <= 治療開始日 and " +
            f"治療開始日 <= '{start_date + rdelta1}' and " +
            f"'{end_date - rdelta2}' <= 治療終了日 and 治療終了日 <= '{end_date + rdelta2}')")
