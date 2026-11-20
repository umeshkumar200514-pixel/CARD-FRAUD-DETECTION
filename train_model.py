"""
Train the card-fraud model from the terminal (same steps as card_fraud_detection.ipynb).

    python train_model.py                  # full data (slow, best results), balancing = under-sampling + SMOTE
    python train_model.py --nrows 200000   # quick trial on the first 200,000 rows of each file
    python train_model.py --balance auto   # let validation PR-AUC choose the balancing method

Needs data/fraudTrain.csv and data/fraudTest.csv (Sparkov dataset from Kaggle).
Creates models/best_model.pkl, models/demo_examples.json (real test-set transactions for the app's demo)
and the charts/tables in reports/ that app.py shows (including samples of the synthetic SMOTE fraud rows).
"""
import argparse
import json
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")                     # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from fraud_utils import (RAW_REQUIRED, build_single_row, engineer_batch, form_values_from_row, fraud_class_values,
                         predict_scores, single_input_from_form)          # SAME feature code the app uses

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
NUM_COLS = ["amt", "hour", "day_of_week", "month", "is_weekend", "is_night",
            "age", "city_pop", "distance_km", "dist_from_prev", "speed_kmh",
            "time_since_last", "amt_diff_prev", "amt_ratio",
            "txn_1h", "txn_24h", "amt_sum_24h"]
CAT_COLS = ["category", "gender", "state"]
FEATURES = NUM_COLS + CAT_COLS
KNN_MAX_TRAIN_ROWS = 200_000              # KNN becomes extremely slow above this
BALANCE_METHODS = ["none", "class_weight", "under", "under+smote"]


def step(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def best_threshold(y_true, prob):
    prec, rec, thr = precision_recall_curve(y_true, prob)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    return float(thr[np.argmax(f1)])


def scores(y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    return {"precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "pr_auc": average_precision_score(y_true, prob),
            "roc_auc": roc_auc_score(y_true, prob)}


def make_xgb(spw=1.0):
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
                         tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)


def balance(method, X, y, cat_idx):
    """Returns (X_balanced, y_balanced, n_real). Rows after position n_real are SYNTHETIC (made by SMOTENC)."""
    if method in ("none", "class_weight"):
        return X, y, len(y)
    under = RandomUnderSampler(sampling_strategy=0.1, random_state=RANDOM_STATE)   # 1 fraud : 10 genuine
    if method == "under":
        Xu, yu = under.fit_resample(X, y)
        return Xu, yu, len(yu)
    if method == "under+smote":
        Xu, yu = under.fit_resample(X, y)
        Xs, ys = SMOTENC(categorical_features=cat_idx, sampling_strategy=0.5,         # 1 fraud : 2 genuine
                         random_state=RANDOM_STATE).fit_resample(Xu, yu)
        return Xs, ys, len(yu)                     # imblearn appends the new synthetic rows at the end
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser(description="Train the card fraud detection model")
    ap.add_argument("--data-dir", default="data", help="folder with fraudTrain.csv and fraudTest.csv")
    ap.add_argument("--nrows", type=int, default=None, help="use only the first N rows of each file (quick trial)")
    ap.add_argument("--no-knn", action="store_true", help="skip the KNN model")
    ap.add_argument("--balance", default="under+smote", choices=["auto"] + BALANCE_METHODS,
                    help="balancing used for the final models (default under+smote; auto = best validation PR-AUC)")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # ------------------------------------------------------------------ 1. load
    step("Loading data")
    train_file = os.path.join(args.data_dir, "fraudTrain.csv")
    test_file = os.path.join(args.data_dir, "fraudTest.csv")
    for f in (train_file, test_file):
        if not os.path.exists(f):
            raise SystemExit(f"File not found: {f}\nDownload the Sparkov dataset from Kaggle "
                             "(kartik2112/fraud-detection) and put both CSV files in the data folder.")
    needed = set(RAW_REQUIRED + ["is_fraud"])                    # read only what the model needs (saves memory)
    train = pd.read_csv(train_file, usecols=lambda c: c in needed, nrows=args.nrows)
    test = pd.read_csv(test_file, usecols=lambda c: c in needed, nrows=args.nrows)
    print(f"Train: {train.shape}, fraud {train['is_fraud'].mean() * 100:.3f}%")
    print(f"Test : {test.shape}, fraud {test['is_fraud'].mean() * 100:.3f}%")

    # ------------------------------------------------------------------ 2. EDA chart
    step("Saving EDA chart -> reports/eda_overview.png")
    eda = train[["trans_date_trans_time", "category", "amt", "is_fraud"]].copy()
    eda["hour"] = pd.to_datetime(eda["trans_date_trans_time"]).dt.hour
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    eda["is_fraud"].value_counts().plot(kind="bar", ax=ax[0, 0], color=["#4C78A8", "#E45756"])
    ax[0, 0].set_title("Class imbalance (0 = genuine, 1 = fraud)")
    eda.groupby("hour")["is_fraud"].mean().mul(100).plot(ax=ax[0, 1], marker="o")
    ax[0, 1].set_title("Fraud rate (%) by hour")
    eda.groupby("category")["is_fraud"].mean().mul(100).sort_values().plot(kind="barh", ax=ax[1, 0])
    ax[1, 0].set_title("Fraud rate (%) by category")
    sns.boxplot(data=eda[eda["amt"] < eda["amt"].quantile(0.99)], x="is_fraud", y="amt", ax=ax[1, 1])
    ax[1, 1].set_title("Amount: genuine vs fraud (below 99th percentile)")
    plt.tight_layout(); plt.savefig("reports/eda_overview.png", dpi=150); plt.close()
    del eda

    # ------------------------------------------------------------------ 3. features (shared with the app)
    step("Building features (train + test joined so test cards see their history; past rows only)")
    train["split"], test["split"] = "train", "test"
    full = engineer_batch(pd.concat([train, test], ignore_index=True))
    del train, test

    # ------------------------------------------------------------------ 4. split, encode, scale
    step("Encoding and scaling (fit on train only)")
    train_df = full[full["split"] == "train"].sort_values("trans_date_trans_time").reset_index(drop=True)
    test_df = full[full["split"] == "test"].sort_values("trans_date_trans_time").reset_index(drop=True)
    del full
    scaler = StandardScaler().fit(train_df[NUM_COLS])
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(train_df[CAT_COLS])

    def prepare(df):
        X = df[FEATURES].copy()
        X[NUM_COLS] = scaler.transform(X[NUM_COLS])
        X[CAT_COLS] = encoder.transform(X[CAT_COLS])
        return X

    X_train_full, y_train_full = prepare(train_df), train_df["is_fraud"].values
    X_test, y_test = prepare(test_df), test_df["is_fraud"].values
    cut = int(len(train_df) * 0.8)                 # time-based: oldest 80% fit, newest 20% validation
    X_tr, y_tr = X_train_full.iloc[:cut], y_train_full[:cut]
    X_val, y_val = X_train_full.iloc[cut:], y_train_full[cut:]
    print(f"Fit part  : {X_tr.shape}, fraud = {int(y_tr.sum())}")
    print(f"Validation: {X_val.shape}, fraud = {int(y_val.sum())}")
    print(f"Test      : {X_test.shape}, fraud = {int(y_test.sum())}")
    if y_tr.sum() < 10 or y_val.sum() < 2:
        raise SystemExit("Too few fraud rows. Use more data (increase --nrows or remove it).")

    # ------------------------------------------------------------------ 5. balancing comparison (validation)
    step("Comparing balancing methods with XGBoost (validation PR-AUC)")
    cat_idx = [FEATURES.index(c) for c in CAT_COLS]
    rows, balanced_sets = [], {}
    for method in BALANCE_METHODS:
        Xb, yb, n_real = balance(method, X_tr, y_tr, cat_idx)
        balanced_sets[method] = (Xb, yb, n_real)
        spw = (yb == 0).sum() / max((yb == 1).sum(), 1) if method == "class_weight" else 1.0
        prob = make_xgb(spw).fit(Xb, yb).predict_proba(X_val)[:, 1]
        thr = best_threshold(y_val, prob)
        rows.append({"method": method, "train_fraud": int(yb.sum()), "synthetic_fraud": len(yb) - n_real,
                     "train_rows": len(yb),
                     "threshold": round(thr, 3), **scores(y_val, prob, thr)})
        print(f"  {method:12s} val PR-AUC = {rows[-1]['pr_auc']:.4f}")
    balance_table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).round(4)
    balance_table.to_csv("reports/balancing_comparison_validation.csv", index=False)
    print("Best balancing method on validation:", balance_table.iloc[0]["method"])
    best_method = balance_table.iloc[0]["method"] if args.balance == "auto" else args.balance
    X_bal, y_bal, n_real = balanced_sets[best_method]
    spw_best = (y_bal == 0).sum() / (y_bal == 1).sum() if best_method == "class_weight" else 1.0
    print("Balancing used for the final models:", best_method)

    # How many rows were real / synthetic, plus a readable sample of the synthetic fraud rows
    n_syn = len(y_bal) - n_real
    summary = {"method": best_method, "original_fit_rows": int(len(y_tr)), "original_fit_fraud": int(y_tr.sum()),
               "balanced_rows": int(len(y_bal)), "balanced_genuine": int((y_bal == 0).sum()),
               "balanced_fraud": int((y_bal == 1).sum()), "synthetic_fraud": int(n_syn)}
    with open("reports/balancing_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  genuine {summary['balanced_genuine']:,} | fraud {summary['balanced_fraud']:,} "
          f"(of which {n_syn:,} synthetic SMOTE rows)")
    if n_syn > 0:
        syn = X_bal.iloc[n_real:].sample(min(50, n_syn), random_state=RANDOM_STATE).copy()
        syn[NUM_COLS] = scaler.inverse_transform(syn[NUM_COLS])          # back to real units
        syn[CAT_COLS] = encoder.inverse_transform(syn[CAT_COLS])
        syn[NUM_COLS] = syn[NUM_COLS].round(2)
        syn.insert(0, "is_fraud", 1)
        syn.to_csv("reports/smote_synthetic_examples.csv", index=False)
        print("  saved reports/smote_synthetic_examples.csv (50 synthetic fraud rows in real units)")

    # ------------------------------------------------------------------ 6. model comparison
    step("Training and comparing models (chosen by VALIDATION PR-AUC, test only reported)")
    cw = "balanced" if best_method == "class_weight" else None
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=cw),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE, class_weight=cw),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=14, n_jobs=-1,
                                                random_state=RANDOM_STATE, class_weight=cw),
        "XGBoost": make_xgb(spw_best),
    }
    if args.no_knn:
        print("  KNN skipped (--no-knn)")
    elif len(y_bal) > KNN_MAX_TRAIN_ROWS:
        print(f"  KNN skipped: {len(y_bal):,} training rows would make it extremely slow")
    else:
        models["KNN"] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    results, fitted, thresholds = [], {}, {}
    for name, mdl in models.items():
        t0 = time.time()
        mdl.fit(X_bal, y_bal)
        val_prob = mdl.predict_proba(X_val)[:, 1]
        thr = best_threshold(y_val, val_prob)                     # threshold from validation
        test_prob = mdl.predict_proba(X_test)[:, 1]               # test only for reporting
        results.append({"model": name, "threshold": round(thr, 3),
                        "val_pr_auc": average_precision_score(y_val, val_prob),
                        **scores(y_test, test_prob, thr)})
        fitted[name], thresholds[name] = mdl, thr
        print(f"  {name:20s} val PR-AUC = {results[-1]['val_pr_auc']:.4f} | test PR-AUC = "
              f"{results[-1]['pr_auc']:.4f} | test F1 = {results[-1]['f1']:.4f}  ({time.time() - t0:.0f}s)")

    comparison = pd.DataFrame(results).sort_values("val_pr_auc", ascending=False).round(4)
    comparison.to_csv("reports/model_comparison_test.csv", index=False)
    ax = comparison.set_index("model")[["precision", "recall", "f1", "pr_auc"]].plot(kind="bar", figsize=(11, 5))
    ax.set_title("Model comparison on the original test set"); ax.set_ylim(0, 1.05)
    plt.xticks(rotation=20); plt.tight_layout(); plt.savefig("reports/model_comparison.png", dpi=150); plt.close()

    # ------------------------------------------------------------------ 7. final model + save
    best_name = comparison.iloc[0]["model"]
    final_model, final_thr = fitted[best_name], thresholds[best_name]
    step(f"Final model: {best_name} | balancing: {best_method} | threshold: {final_thr:.3f}")
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= final_thr).astype(int)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    print("Confusion matrix (test):\n", cm)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix (test)")
    plt.tight_layout(); plt.savefig("reports/confusion_matrix.png", dpi=150); plt.close()

    if hasattr(final_model, "feature_importances_"):
        pd.Series(final_model.feature_importances_, index=FEATURES).sort_values().plot(
            kind="barh", figsize=(7, 7), title="Feature importance")
        plt.tight_layout(); plt.savefig("reports/feature_importance.png", dpi=150); plt.close()

    joblib.dump({"model": final_model, "threshold": final_thr, "scaler": scaler, "encoder": encoder,
                 "features": FEATURES, "num_cols": NUM_COLS, "cat_cols": CAT_COLS,
                 "model_name": best_name, "balancing": best_method},
                "models/best_model.pkl")
    print("Saved models/best_model.pkl")

    # ------------------------------------------------------------------ 8. SHAP charts
    step("SHAP explainability charts")
    try:
        import shap
        explainer = shap.TreeExplainer(final_model)
        sample = X_test.sample(min(2000, len(X_test)), random_state=RANDOM_STATE)
        sv = fraud_class_values(explainer.shap_values(sample))
        plt.figure(); shap.summary_plot(sv, sample, feature_names=FEATURES, show=False)
        plt.tight_layout(); plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight"); plt.close("all")
        plt.figure(); shap.summary_plot(sv, sample, feature_names=FEATURES, plot_type="bar", show=False)
        plt.tight_layout(); plt.savefig("reports/shap_bar.png", dpi=150, bbox_inches="tight"); plt.close("all")
        print("Saved reports/shap_summary.png and reports/shap_bar.png")
    except Exception as e:
        print("SHAP skipped:", e)

    # ------------------------------------------------------------------ 9. demo examples for the app
    step("Saving real test-set examples for the app demo -> models/demo_examples.json")
    bundle = joblib.load("models/best_model.pkl")
    examples = []

    def add_examples(candidates, label, want, flagged):
        n = 0
        for _, r in candidates.iterrows():
            f = form_values_from_row(r)
            s = float(predict_scores(bundle, build_single_row(single_input_from_form(f)))[0])   # exactly what the app computes
            if (s >= final_thr) == flagged:
                n += 1
                examples.append({"name": f"Real {label} #{n} from test set (card ...{str(r['cc_num'])[-4:]}, "
                                         f"{r['category']}, ${f['amt']:,.2f})",
                                 "actual": label, "score": round(s, 4), "values": f})
            if n == want:
                break
        return n

    order = np.argsort(-test_prob)
    ranked = test_df.iloc[order].assign(_score=test_prob[order])
    n_f = add_examples(ranked[(ranked["is_fraud"] == 1) & (ranked["_score"] >= final_thr)].head(200), "FRAUD", 6, True)
    gen = test_df[(test_df["is_fraud"] == 0) & (test_prob < final_thr * 0.3)]
    n_g = add_examples(gen.sample(min(200, len(gen)), random_state=RANDOM_STATE), "GENUINE", 3, False)
    with open("models/demo_examples.json", "w") as fh:
        json.dump({"threshold": final_thr, "examples": examples}, fh, indent=2)
    print(f"Saved {n_f} real fraud and {n_g} real genuine examples (each re-checked through the app's own code)")

    # A small CSV for the Batch page: all test transactions of a few cards that had caught fraud
    caught_cards = ranked[(ranked["is_fraud"] == 1) & (ranked["_score"] >= final_thr)]["cc_num"].unique()[:12]
    demo = pd.concat(chunk[chunk["cc_num"].isin(caught_cards)]          # read in pieces to save memory
                     for chunk in pd.read_csv(test_file, index_col=0, nrows=args.nrows, chunksize=100_000))
    demo.to_csv("reports/demo_batch.csv", index=False)
    print(f"Saved reports/demo_batch.csv ({len(demo):,} rows, {int(demo['is_fraud'].sum())} real fraud) for the Batch page")

    step("Done! Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
