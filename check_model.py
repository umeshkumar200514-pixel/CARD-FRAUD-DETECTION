"""
Check that the trained model really catches fraud, and show REAL fraud transactions it caught,
so you can copy their values into the app's "Check one transaction" form.

    python check_model.py
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from fraud_utils import engineer_batch, load_bundle, predict_scores

N_ROWS = 100_000

bundle = load_bundle()
thr = float(bundle["threshold"])
print(f"Model: {bundle['model_name']} | balancing: {bundle['balancing']} | threshold: {thr:.3f} ({thr * 100:.1f}%)")

raw = pd.read_csv("data/fraudTest.csv", index_col=0, nrows=N_ROWS)
feats = engineer_batch(raw)
scores = predict_scores(bundle, feats)
y, pred = raw["is_fraud"].values, (scores >= thr).astype(int)

print(f"\nFirst {N_ROWS:,} rows of fraudTest.csv")
print(f"Real fraud rows    : {y.sum()}")
print(f"Caught by the model: {int(((y == 1) & (pred == 1)).sum())}")
print(f"Recall {recall_score(y, pred, zero_division=0):.3f} | Precision {precision_score(y, pred, zero_division=0):.3f}")
print("Confusion matrix [[genuine ok, false alarm], [missed fraud, caught fraud]]:")
print(confusion_matrix(y, pred, labels=[0, 1]))

feats["fraud_score_%"] = (scores * 100).round(1)
caught = feats[(y == 1) & (pred == 1)].sort_values("fraud_score_%", ascending=False)
cols = ["fraud_score_%", "amt", "category", "trans_date_trans_time", "gender", "state", "age", "city_pop",
        "lat", "long", "merch_lat", "merch_long", "time_since_last", "prev_amt", "card_avg_amt",
        "amt_sum_24h", "txn_1h", "txn_24h", "prev_merch_lat", "prev_merch_long"]
pd.set_option("display.width", 250, "display.max_columns", 30)
print("\nTop 3 REAL fraud transactions the model caught (copy these values into the app form):")
for _, r in caught[cols].head(3).iterrows():
    print("-" * 60)
    first = pd.isna(r["prev_amt"])
    print(f"  first transaction of card  : {'YES -> tick the box, history fields are ignored' if first else 'no -> leave the box unticked'}")
    for c in cols:
        v = r[c]
        if c == "time_since_last":
            print(f"  hours_since_last (form)    : {v / 3600:.2f}")
        else:
            print(f"  {c:27s}: {round(v, 4) if isinstance(v, float) else v}")
