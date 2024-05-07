import datetime
import re

import pandas as pd


class Medication(object):
    def __init__(self, config):
        norm_dict = config["treatment_period"]["normalization"].copy()
        for k in norm_dict:
            if type(norm_dict[k]) is not list:
                norm_dict[k] = [norm_dict[k]]
        self.norm_dict = norm_dict

    def normalize(self, s):
        s = re.sub(r"Cap|\d+(mg|mL)", "", s).strip()
        m = re.match(r"治験 *([-0-9A-Za-z\u30A0-\u30FF]+)", s)
        if m:
            s = m.group(1)
        for k, v in self.norm_dict.items():
            for w in v:
                if w in s:
                    return k
        return s


class Regimen(object):
    def __init__(self, config):
        regimen_dict = config["treatment_period"]["regimens"].copy()
        for k in regimen_dict:
            if type(regimen_dict[k]) is not list:
                regimen_dict[k] = [regimen_dict[k]]
        self.regimen_dict = regimen_dict

    def normalize(self, s):
        s = re.sub(r"・NETU|・OLZ|weekly", "", s).strip()
        s = re.sub(r"±", "+/-", s)
        m = re.match(r"治験 *([-0-9A-Za-z]+)", s)
        if m:
            s = m.group(1)
        for k, v in self.regimen_dict.items():
            if s in v:
                return k
        return s


class TreatmentPeriod(object):
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.medication = Medication(self.config)
        self.regimen = Regimen(self.config)

        self.emr_injection = data.emr_injection.copy()
        self.emr_injection["MedicationName"] = \
            self.emr_injection["MedicationName"].map(self.medication.normalize)

        self.emr_prescription = data.emr_prescription.copy()
        self.emr_prescription["MedicationName"] = \
            self.emr_prescription["MedicationName"].map(self.medication.normalize)

    @staticmethod
    def normalize_pathname(s):
        return re.sub(r"\(short\)", "", s).strip()

    def __is_same_regimen(self, regimen_df, date1, date2):
        pathname1 = None
        if date1:
            df = regimen_df.query("StartDate <= @date1 and @date1 <= EndDate")
            if not df.empty:
                pathname1 = df.iloc[0]["PATHNAME"]

        pathname2 = None
        if date2:
            df = regimen_df.query("StartDate <= @date2 and @date2 <= EndDate")
            if not df.empty:
                pathname2 = df.iloc[0]["PATHNAME"]

        if pathname1 is None and pathname2 is None:
            return False
        return pathname1 == pathname2

    def __treatment_periods_from_df(self, df):
        df = df.sort_values(by="DateTime")

        df1 = pd.DataFrame(columns=df.columns)
        d = None
        ms = {}
        for index, row in df.iterrows():
            date = row["DateTime"].date()
            medication_name = row["MedicationName"]
            if date == d:
                ms.add(medication_name)
            else:
                if len(ms):
                    df1.at[df1.index[-1], "MedicationName"] = "+".join(sorted(ms))
                df1.loc[len(df1)] = row
                ms = {medication_name}
                d = date
        if len(ms):
            df1.at[df1.index[-1], "MedicationName"] = "+".join(sorted(ms))

        df2 = pd.DataFrame(columns=df1.columns)
        df2["LastDateTime"] = None
        d = None
        m = None
        for index, row in df1.iterrows():
            medication_name = self.regimen.normalize(row["MedicationName"])
            date = row["DateTime"]
            if m == medication_name:
                if not pd.isnull(row["DosageQuantity"]) and \
                   "1日" in row["Usage"] and \
                   row["DosageQuantityUnit"] == "日分":
                    date += datetime.timedelta(days=row["DosageQuantity"])
                df2.at[df2.index[-1], "LastDateTime"] = date
            else:
                row["LastDateTime"] = row["DateTime"]
                df2.loc[len(df2)] = row
                d = row["DateTime"]
                m = medication_name
        return df2

    def __injection_periods(self, patient_id):
        includes = "|".join(self.config["treatment_period"]["includes"])
        excludes = "|".join(self.config["treatment_period"]["excludes"])

        if "AntiCancer" in self.emr_injection:
            p_injection = self.emr_injection.query("PatientNo == @patient_id") \
                .query("EndDateTime != 'NULL'") \
                .query(f"AntiCancer == True or MedicationName.str.match('.*({includes}).*')") \
                .query(f"not MedicationName.str.match('.*({excludes}).*')")
        else:
            p_injection = self.emr_injection.query("PatientNo == @patient_id and ") \
                .query("EndDateTime != 'NULL'") \
                .query(f"(MedicationName.str.match('.*({includes}).*') or "
                       "DocumentTitle.str.match('.*抗がん剤注射.*') and ") \
                .query(f"not MedicationName.str.match('.*({excludes}).*'))")

        return p_injection

    def __prescription_periods(self, patient_id):
        includes = "|".join(self.config["treatment_period"]["includes"])
        excludes = "|".join(self.config["treatment_period"]["excludes"])

        if "AntiCancer" in self.emr_prescription:
            p_prescription = self.emr_prescription.query("PatientNo == @patient_id") \
                .query(f"AntiCancer == True or MedicationName.str.match('.*({includes}).*')") \
                .query(f"not MedicationName.str.match('.*({excludes}).*')") \
                .query("DocumentTitle != '中止処方'")
        else:
            p_prescription = self.emr_prescription.query("PatientNo == @patient_id") \
                .query(f"MedicationName.str.match('.*({includes}).*')") \
                .query(f"not MedicationName.str.match('.*({excludes}).*')") \
                .query("DocumentTitle != '中止処方'")

        return p_prescription

    def treatment_periods(self, patient_id):
        df1 = self.__injection_periods(patient_id)
        df2 = self.__prescription_periods(patient_id)
        df = pd.concat([df1, df2]).sort_values(by="DateTime")

        df = self.__treatment_periods_from_df(df)
        df["MedicationName"] = df["MedicationName"].map(self.regimen.normalize)

        return df
