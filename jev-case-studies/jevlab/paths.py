import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PREPARED = DATA / "prepared"
CACHE = DATA / "cache"
# Pipeline-test (mock) runs write to a separate tree so they can never be mistaken for findings.
_MOCK = os.environ.get("JEV_MOCK") == "1"
RESULTS = ROOT / "results" / "mock" if _MOCK else ROOT / "results"
FIGURES = RESULTS / "figures"
REPORTS = ROOT / "reports" / "mock" if _MOCK else ROOT / "reports"

for _p in (PREPARED, CACHE, RESULTS, FIGURES, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)
