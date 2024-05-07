import datetime
import re


def normalize_patient_id(patient_id):
    m = re.match(r"0*([1-9]\d+)", patient_id)
    if m:
        return m.group(1)
    else:
        return patient_id


def normalize_medication(s, regexps=[]):
    s = re.sub(r"Cap|\d+(mg|mL)", "", s).strip()
    for r in regexps:
        m = re.search(r, s)
        if m:
            return m.group(0)
    return s


def date_to_datetime(date):
    return datetime.datetime.combine(date, datetime.time())
