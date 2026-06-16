"""Append-only audit trail for the falsification battery (results/falsification_log.csv).

One row per test. The battery halts and reports on any FAIL — callers must check.
"""
import os
import pandas as pd
from config import OUTPUT_DIR

LOG_PATH = os.path.join(OUTPUT_DIR, "falsification_log.csv")

COLUMNS = [
    "Test_ID", "Test_Name", "World_Eliminated", "Readout_Metric",
    "Decision_Rule", "Observed_Value", "Verdict", "Sweep_Refs", "Notes",
]


def log_verdict(Test_ID, Test_Name, World_Eliminated, Readout_Metric,
                Decision_Rule, Observed_Value, Verdict, Sweep_Refs="", Notes=""):
    """Append (or replace) a verdict row. Re-running a test overwrites its row
    by Test_ID so the log reflects the latest evidence, but never silently drops
    other tests."""
    assert Verdict in {"PASS", "FAIL", "INCONCLUSIVE", "PENDING"}, Verdict
    row = {
        "Test_ID": Test_ID, "Test_Name": Test_Name, "World_Eliminated": World_Eliminated,
        "Readout_Metric": Readout_Metric, "Decision_Rule": Decision_Rule,
        "Observed_Value": Observed_Value, "Verdict": Verdict,
        "Sweep_Refs": Sweep_Refs, "Notes": Notes,
    }
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        df = df[df.Test_ID != Test_ID]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=COLUMNS)
    df.to_csv(LOG_PATH, index=False)
    print(f"[falsification_log] {Test_ID} {Verdict}: {Test_Name} (obs={Observed_Value})")
    return df


def has_pending_or_fail() -> bool:
    if not os.path.exists(LOG_PATH):
        return True
    df = pd.read_csv(LOG_PATH)
    return bool((df.Verdict.isin(["PENDING", "FAIL"])).any())
