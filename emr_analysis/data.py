import csv
from enum import Enum
from pathlib import Path
import pickle
import re

from gensim.models import KeyedVectors
import numpy as np
import pandas as pd

from emr_analysis.preprocessor import ConvertNewlinePreprocessor, StripPreprocessor, \
    TrimHeaderPreprocessor, ZenhanPreprocessor
import emr_analysis.util as util


class Effect(Enum):
    CR = 0
    PR = 1
    PD = 2
    SD = 3
    NE = 4
    UNKNOWN = 5
    UNDETECTED = 6

    @staticmethod
    def from_str(s):
        if s == "CR":
            return Effect.CR
        elif s == "PR":
            return Effect.PR
        elif s == "PD":
            return Effect.PD
        elif s == "SD":
            return Effect.SD
        elif s == "NE":
            return Effect.NE
        else:
            return Effect.UNKNOWN

    @staticmethod
    def names():
        return [Effect.CR.name,
                Effect.PR.name,
                Effect.PD.name,
                Effect.SD.name,
                Effect.NE.name,
                Effect.UNKNOWN.name,
                Effect.UNDETECTED.name]


class Data(object):
    DEFAULT_WORD_EMBEDDING_FILE = "word-embedding/fasttext.stanza.ja.300.vec.gz"

    def __init__(self, data_dir="./data/", word_embedding_file=None, preprocess=False):
        self.data_dir = Path(data_dir)
        self.preprocess = preprocess

        self.emr_injection = None
        self.emr_prescription = None
        self.emr_radiation_report = None

        self.fm_treatment_record = None

        self.patient_ids = []

        if word_embedding_file is None:
            self.word_embedding_file = self.data_dir / Data.DEFAULT_WORD_EMBEDDING_FILE
        else:
            self.word_embedding_file = Path(word_embedding_file)
        self.word_embedding: KeyedVectors = None

    def _load_data(self, f, strip=False, convert_newline=False, zenhan=False, trim_header=1):
        with open(self.data_dir / f, newline="") as in_csv:
            rdr = csv.reader(in_csv)
            rows = list(rdr)

        if strip:
            rows = StripPreprocessor().preprocess(rows)

        if convert_newline:
            rows = ConvertNewlinePreprocessor().preprocess(rows)

        if zenhan:
            rows = ZenhanPreprocessor().preprocess(rows)

        header = [re.sub("[・＿]", "_", s) for s in rows[0]]
        contents = TrimHeaderPreprocessor({"header_lines": trim_header}).preprocess(rows)

        return pd.DataFrame(contents, columns=header)

    def _load_emr_injection(self, **opts):
        df = self._load_data("emr/injection.csv", **opts)
        df["PatientNo"] = df["PatientNo"].map(util.normalize_patient_id)
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df["EndDateTime"] = pd.to_datetime(df["EndDateTime"], errors="coerce")
        df["OrderDateTime"] = pd.to_datetime(df["OrderDateTime"])
        if "AntiCancer" in df:
            df = df.astype({"AntiCancer": int})
            df["AntiCancer"] = df["AntiCancer"].map(bool)
        df = df.astype({"PatientNo": "category"})
        return df.sort_values(by=["PatientNo", "DateTime"])

    def _load_emr_prescription(self, **opts):
        df = self._load_data("emr/prescription.csv", **opts)
        df["PatientNo"] = df["PatientNo"].map(util.normalize_patient_id)
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df["OrderDateTime"] = pd.to_datetime(df["OrderDateTime"])
        df["DosageQuantity"] = pd.to_numeric(df["DosageQuantity"], errors="coerce")
        if "AntiCancer" in df:
            df = df.astype({"AntiCancer": int})
            df["AntiCancer"] = df["AntiCancer"].map(bool)
        df = df.astype({"PatientNo": "category"})
        return df.sort_values(by=["PatientNo", "DateTime"])

    def _load_emr_radiation_report(self, **opts):
        df = self._load_data("emr/radiation_report.csv", **opts)
        df["PatientNo"] = df["PatientNo"].map(util.normalize_patient_id)
        df["ExaminDateTime"] = pd.to_datetime(df["ExaminDateTime"])
        df = df.astype({"PatientNo": "category"})
        return df.sort_values(by=["PatientNo", "ExaminDateTime"])

    def _load_emr_regimen_calendar(self, **opts):
        df = self._load_data("emr/regimen_calendar.csv", **opts)
        df["PatientNo"] = df["PatientNo"].map(util.normalize_patient_id)
        df["StartDate"] = pd.to_datetime(df["StartDate"])
        df["EndDate"] = pd.to_datetime(df["EndDate"])
        df = df.astype({"PatientNo": "category"})
        return df.sort_values(by=["PatientNo", "StartDate"])

    def _load_emr_regimen_injection(self, **opts):
        df = self._load_data("emr/regimen_injection.csv", **opts)
        df["PatientNo"] = df["PatientNo"].map(util.normalize_patient_id)
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df["EndDateTime"] = pd.to_datetime(df["EndDateTime"], errors="coerce")
        df["OrderDateTime"] = pd.to_datetime(df["OrderDateTime"])
        if "AntiCancer" in df:
            df = df.astype({"AntiCancer": int})
            df["AntiCancer"] = df["AntiCancer"].map(bool)
        df = df.astype({"PatientNo": "category"})
        return df.sort_values(by=["PatientNo", "DateTime"])

    def _load_fm_treatment_record(self, **opts):
        df = self._load_data("structured/treatment_record.csv", **opts)
        df["患者ID"] = df["患者ID"].map(util.normalize_patient_id)
        df["治療開始日"] = pd.to_datetime(df["治療開始日"])
        df["治療終了日"] = pd.to_datetime(df["治療終了日"])
        df["効果"] = df["効果"].map(Effect.from_str)
        df["PD確定日"] = pd.to_datetime(df["PD確定日"])
        df = df.astype({"患者ID": "category"})
        return df.sort_values(by=["患者ID", "治療開始日"])

    def _filter_patients(self):
        emr_patient_set = set(self.emr_injection["PatientNo"]) | \
            set(self.emr_prescription["PatientNo"]) | \
            set(self.emr_radiation_report["PatientNo"])
        fm_patient_set = set(self.fm_treatment_record["患者ID"])
        patient_set = emr_patient_set & fm_patient_set
        self.patient_ids = sorted(list(patient_set))

        self.emr_injection.query("PatientNo in @patient_set", inplace=True)
        self.emr_prescription.query("PatientNo in @patient_set", inplace=True)
        self.emr_radiation_report.query("PatientNo in @patient_set", inplace=True)

        self.fm_treatment_record.query("患者ID in @patient_set", inplace=True)

    def _load_word_embedding(self) -> KeyedVectors:
        pkl_path = self.word_embedding_file.parent / (self.word_embedding_file.name + ".pkl")

        if pkl_path.exists():
            with open(pkl_path, mode="rb") as fp:
                word_embedding = pickle.load(fp)
        else:
            word_embedding = KeyedVectors.load_word2vec_format(self.word_embedding_file,
                                                               binary=False)
            with open(pkl_path, mode="wb") as fp:
                pickle.dump(word_embedding, fp)
            print(f"Pickled word embedding model: {pkl_path}")

        word_embedding.add_vectors(["<pad>", "<unk>"],
                                   [np.zeros(word_embedding.vector_size)] * 2)
        return word_embedding

    def load_data(self):
        emr_opts = {}
        if self.preprocess:
            emr_opts = {
                "strip": True,
                "convert_newline": True,
                "zenhan": True,
                "trim_header": 1
            }
        else:
            emr_opts = {"trim_header": 1}

        print("Loading EMR data...")
        self.emr_injection = self._load_emr_injection(**emr_opts)
        self.emr_prescription = self._load_emr_prescription(**emr_opts)
        self.emr_radiation_report = self._load_emr_radiation_report(**emr_opts)
        print("Loaded EMR data")

        fm_opts = {}
        if self.preprocess:
            fm_opts = {
                "strip": True,
                "convert_newline": True,
                "zenhan": True,
                "trim_header": 1
            }
        else:
            fm_opts = {"trim_header": 1}

        print("Loading structured data...")
        self.fm_treatment_record = self._load_fm_treatment_record(**fm_opts)
        print("Loaded structured data")

        self._filter_patients()

        print("Loading word embedding model...")
        self.word_embedding = self._load_word_embedding()
        print("Loaded word embedding model")
