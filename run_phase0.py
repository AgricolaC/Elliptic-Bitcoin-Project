"""Phase 0 runner: materialize ground truth and write F0a-F0e verdicts.

Code-level confound fixes (F0b self-conditioning, F0c ε-fallback, F0d label
masking, F0e pagerank audit) are enforced by the pytest suite; this script
records their verdicts and generates results/snapshot_topology.csv (F0a).
"""
import sys, os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "source")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

import numpy as np
import pandas as pd
from config import OUTPUT_DIR
from data.load_dataset import download_and_load_data
from data.snapshot_topology import build_snapshot_topology
from evaluation.falsification_log import log_verdict


def _pytest(node) -> bool:
    """Return True if the given pytest node id passes."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", node, "-q"],
        cwd=HERE, capture_output=True, text=True
    )
    print(r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])
    return r.returncode == 0


def main():
    print("Loading raw dataset...")
    df, df_edge, _, _ = download_and_load_data()

    # ── F0a: snapshot_topology.csv + τ=43 illicit-collapse assertion ──────────
    topo = build_snapshot_topology(df, df_edge)
    topo_path = os.path.join(OUTPUT_DIR, "snapshot_topology.csv")
    topo.to_csv(topo_path, index=False)
    print(f"Wrote {topo_path} ({len(topo)} rows)")

    n42 = int(topo.loc[topo.Tau == 42, "N_illicit"].values[0])
    n43 = int(topo.loc[topo.Tau == 43, "N_illicit"].values[0])
    ratio = n43 / max(n42, 1)
    collapse = n43 < n42 / 5
    log_verdict(
        "F0a", "Snapshot ground truth + τ=43 illicit collapse",
        World_Eliminated="none (ground truth)",
        Readout_Metric="N_illicit[43] / N_illicit[42]",
        Decision_Rule="N_illicit[43] < N_illicit[42] / 5",
        Observed_Value=round(ratio, 4),
        Verdict="PASS" if collapse else "FAIL",
        Notes=f"N_illicit: τ42={n42}, τ43={n43}",
    )

    # ── F0b-F0e: code-level confound fixes, enforced by pytest ────────────────
    checks = [
        ("F0b", "Self-conditioning fixed (one-step-ahead, exclude τ)",
         "World C-confound", "test passes",
         "inference state excludes τ; calib state excludes τ-1",
         "tests/test_remediation.py::TestOneStepAheadBlocks"),
        ("F0c", "ε-fallback threshold under prevalence collapse",
         "enables World-C isolation", "test passes",
         "fallback fires when calib positives < ε=10",
         "tests/test_remediation.py::TestEpsilonFallback"),
        ("F0d", "Label masking: y=-1 excluded from loss",
         "measurement validity", "test passes",
         "perturbing masked labels leaves trained head identical",
         "tests/test_remediation.py::TestTemporalModels::test_temporal_loss_respects_labeled_mask"),
        ("F0e", "PageRank feature alive under early injection",
         "feeds F4a interpretation", "test passes",
         "PageRank column std > 0.01 across nodes",
         "tests/test_remediation.py::TestPagerankAudit"),
    ]
    for tid, name, world, metric, rule, node in checks:
        ok = _pytest(node)
        log_verdict(tid, name, World_Eliminated=world, Readout_Metric=metric,
                    Decision_Rule=rule, Observed_Value=1.0 if ok else 0.0,
                    Verdict="PASS" if ok else "FAIL", Notes=node)

    print("\nPhase 0 complete. falsification_log.csv:")
    print(pd.read_csv(os.path.join(OUTPUT_DIR, "falsification_log.csv")).to_string(index=False))


if __name__ == "__main__":
    main()
