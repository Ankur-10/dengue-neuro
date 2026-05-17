"""
train_model.py
──────────────
Trains the two-stage Random Forest pipeline for Dengue Neurological
Complication prediction and saves all artefacts to the models/ directory.

Stage 1 : CNS vs PNS  (binary classifier)
Stage 2a: CNS-specific diagnosis  (E / En / SD / STROKE / TM)
Stage 2b: PNS-specific diagnosis  (GBS / HPP / MYO)

Usage
-----
    python train_model.py --data data/Newdata_dengue_neuro.xlsx

Output (saved to models/)
--------------------------
    rf_step1.pkl        – Stage-1 Random Forest
    rf_cns.pkl          – Stage-2a Random Forest (CNS)
    rf_pns.pkl          – Stage-2b Random Forest (PNS)
    le_cns.pkl          – LabelEncoder for CNS diagnoses
    le_pns.pkl          – LabelEncoder for PNS diagnoses
    feature_config.pkl  – Feature lists used at inference time
    training_report.txt – Cross-validation metrics summary
"""

import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

DROP_COLS = [
    "name", "add",                          # identifiers / free text
    "PAST EPI",                             # 97.8 % null
    "POWER",                                # 91.1 % null
    "adm_GBS", "F/U_GBS",                  # >50 % null (CPK & NCS_pattern kept as sparse optional features)
    "F/U_MBI", "F/U_MRS",                  # follow-up (not available at admission)
    "KFT", "LFT",                           # binary flags – redundant with CREAT/UREA/SGOT etc.
]

SHARED_FEATURES = [
    "age", "sex",
    "fever ", "rash", "bleeding", "vitals",
    "GCS",
    "Hb", "TLC", "neutrophls", "lymphocytes", "Hct", "platelet",
    "Na", "K", "RBS",
    "CREAT", "UREA",
    "SGOT", "SGPT", "Billirubin",
    "protein", "albumin",
]

CNS_EXTRA_FEATURES = [
    "headache", "alteredsensorium", "siezure",
    "CN involv", "B/B invol", "CN deficit",
    "adm _MBI", "adm_MRS",
    "CSF_TLC", "CSF_P", "CSF_L", "CSF_protein", "CSF_sugar",
]

PNS_EXTRA_FEATURES = [
    "weakness", "sensory",
    "CPK",          # creatine phosphokinase – elevated in myopathy / rhabdomyolysis
    "NCS_pattern",  # nerve conduction study pattern (0 = normal/not done, 1 = abnormal)
]

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

CV_FOLDS = 5


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.dropna(subset=["CNS/PNS", "diagnosis"])

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 0)

    df["sex"] = df["sex"].astype(int)
    return df


def resolve_features(wanted: list[str], available: list[str]) -> list[str]:
    return [f for f in dict.fromkeys(wanted) if f in available]


def train_and_evaluate(model, X, y, label: str, cv: int = CV_FOLDS) -> dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    acc_scores  = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    f1_scores   = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted")
    rec_scores  = cross_val_score(model, X, y, cv=skf, scoring="recall_weighted")
    prec_scores = cross_val_score(model, X, y, cv=skf, scoring="precision_weighted")

    results = {
        "label":     label,
        "accuracy":  acc_scores,
        "f1":        f1_scores,
        "recall":    rec_scores,
        "precision": prec_scores,
    }

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc_scores.mean():.3f}  ± {acc_scores.std():.3f}")
    print(f"  F1 (wtd)  : {f1_scores.mean():.3f}  ± {f1_scores.std():.3f}")
    print(f"  Recall    : {rec_scores.mean():.3f}  ± {rec_scores.std():.3f}")
    print(f"  Precision : {prec_scores.mean():.3f}  ± {prec_scores.std():.3f}")

    # Full-data fit for final classification report
    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"\n  Classification report (train):\n")
    report = classification_report(y, y_pred, zero_division=0)
    print("\n".join("    " + line for line in report.splitlines()))
    print(f"  Confusion matrix (train):\n{confusion_matrix(y, y_pred)}")

    return results


def save_pkl(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")


def write_report(results_list: list[dict], path: str):
    lines = ["TRAINING REPORT – Dengue Neuro Diagnosis Model",
             "=" * 55, ""]
    for r in results_list:
        lines += [
            f"Model : {r['label']}",
            f"  CV Accuracy  : {r['accuracy'].mean():.3f} ± {r['accuracy'].std():.3f}",
            f"  CV F1 (wtd)  : {r['f1'].mean():.3f} ± {r['f1'].std():.3f}",
            f"  CV Recall    : {r['recall'].mean():.3f} ± {r['recall'].std():.3f}",
            f"  CV Precision : {r['precision'].mean():.3f} ± {r['precision'].std():.3f}",
            "",
        ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))
    print(f"\n  Report saved → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(data_path: str, output_dir: str = "models"):
    print(f"\nLoading data from: {data_path}")
    df = load_and_clean(data_path)
    print(f"Clean dataset shape: {df.shape}")
    print(f"CNS: {(df['CNS/PNS']=='CNS').sum()}  |  PNS: {(df['CNS/PNS']=='PNS').sum()}")

    available = df.columns.tolist()

    # ── Feature lists ──────────────────────────────────────────────────────
    all_feats  = resolve_features(SHARED_FEATURES + CNS_EXTRA_FEATURES + PNS_EXTRA_FEATURES, available)
    cns_feats  = resolve_features(SHARED_FEATURES + CNS_EXTRA_FEATURES + PNS_EXTRA_FEATURES, available)
    pns_feats  = resolve_features(SHARED_FEATURES + PNS_EXTRA_FEATURES + CNS_EXTRA_FEATURES, available)

    print(f"\nFeature counts – Step1: {len(all_feats)}  | CNS: {len(cns_feats)}  | PNS: {len(pns_feats)}")

    # ── Stage 1 : CNS vs PNS ───────────────────────────────────────────────
    X_all = df[all_feats]
    y_step1 = (df["CNS/PNS"] == "CNS").astype(int)

    rf_step1 = RandomForestClassifier(**RF_PARAMS)
    r1 = train_and_evaluate(rf_step1, X_all, y_step1, "Stage-1  |  CNS vs PNS")

    # ── Stage 2a : CNS diagnosis ───────────────────────────────────────────
    cns_mask = df["CNS/PNS"] == "CNS"
    X_cns    = df[cns_mask][cns_feats]
    le_cns   = LabelEncoder()
    y_cns    = le_cns.fit_transform(df[cns_mask]["diagnosis"])

    rf_cns = RandomForestClassifier(**RF_PARAMS)
    r2 = train_and_evaluate(rf_cns, X_cns, y_cns,
                             f"Stage-2a | CNS Diagnosis  {list(le_cns.classes_)}")

    # ── Stage 2b : PNS diagnosis ───────────────────────────────────────────
    pns_mask = df["CNS/PNS"] == "PNS"
    X_pns    = df[pns_mask][pns_feats]
    le_pns   = LabelEncoder()
    y_pns    = le_pns.fit_transform(df[pns_mask]["diagnosis"])

    rf_pns = RandomForestClassifier(**RF_PARAMS)
    r3 = train_and_evaluate(rf_pns, X_pns, y_pns,
                             f"Stage-2b | PNS Diagnosis  {list(le_pns.classes_)}")

    # ── Feature importances ────────────────────────────────────────────────
    print("\n\n── Top-10 Feature Importances ──────────────────────────────")
    for name, mdl, feats in [
        ("Stage-1 (CNS/PNS)", rf_step1, all_feats),
        ("Stage-2a (CNS diag)", rf_cns, cns_feats),
        ("Stage-2b (PNS diag)", rf_pns, pns_feats),
    ]:
        imp = pd.Series(mdl.feature_importances_, index=feats).nlargest(10)
        print(f"\n  {name}:")
        for feat, val in imp.items():
            print(f"    {feat:<25} {val:.4f}")

    # ── Save artefacts ─────────────────────────────────────────────────────
    print("\n\nSaving model artefacts …")
    save_pkl(rf_step1,  f"{output_dir}/rf_step1.pkl")
    save_pkl(rf_cns,    f"{output_dir}/rf_cns.pkl")
    save_pkl(rf_pns,    f"{output_dir}/rf_pns.pkl")
    save_pkl(le_cns,    f"{output_dir}/le_cns.pkl")
    save_pkl(le_pns,    f"{output_dir}/le_pns.pkl")
    save_pkl(
        {"all_feats": all_feats, "cns_feats": cns_feats, "pns_feats": pns_feats},
        f"{output_dir}/feature_config.pkl",
    )
    write_report([r1, r2, r3], f"{output_dir}/training_report.txt")

    print("\n✅  Training complete.  All artefacts saved to:", output_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dengue Neuro RF models")
    parser.add_argument(
        "--data",
        default="data/Newdata_dengue_neuro.xlsx",
        help="Path to the input Excel file",
    )
    parser.add_argument(
        "--output_dir",
        default="models",
        help="Directory to save model artefacts (default: models/)",
    )
    args = parser.parse_args()
    main(args.data, args.output_dir)
