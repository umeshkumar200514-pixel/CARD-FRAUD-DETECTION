"""
Helper functions for the Streamlit app.
The feature code here is the SAME logic as the training notebook (card_fraud_detection.ipynb),
so the app sees exactly the features the model was trained on.
"""
import joblib
import numpy as np
import pandas as pd

# Columns a CSV must have for batch prediction (raw Sparkov columns)
RAW_REQUIRED = ["trans_date_trans_time", "cc_num", "category", "amt", "gender", "state",
                "lat", "long", "city_pop", "dob", "merch_lat", "merch_long"]


def load_bundle(path="models/best_model.pkl"):
    return joblib.load(path)


def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two latitude/longitude points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- batch (CSV) features
def engineer_batch(df):
    """Build all model features from raw Sparkov-style columns. History features use past rows of the SAME card
    inside the uploaded file only."""
    df = df.copy()
    df["_row"] = np.arange(len(df))                       # to restore the original order later
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    t = df["trans_date_trans_time"]

    df["hour"] = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = df["hour"].isin([23, 0, 1, 2, 3, 4]).astype(int)
    df["age"] = (t - df["dob"]).dt.days / 365.25
    df["distance_km"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    g = df.groupby("cc_num")
    df["prev_merch_lat"] = g["merch_lat"].shift(1)
    df["prev_merch_long"] = g["merch_long"].shift(1)
    df["dist_from_prev"] = haversine(df["prev_merch_lat"], df["prev_merch_long"], df["merch_lat"], df["merch_long"])

    df["time_since_last"] = g["trans_date_trans_time"].diff().dt.total_seconds()
    hours = (df["time_since_last"] / 3600).clip(lower=1 / 60)
    df["speed_kmh"] = df["dist_from_prev"] / hours

    df["prev_amt"] = g["amt"].shift(1)
    df["amt_diff_prev"] = df["amt"] - df["prev_amt"]
    df["card_avg_amt"] = g["amt"].transform(lambda s: s.shift(1).expanding().mean())
    df["amt_ratio"] = df["amt"] / df["card_avg_amt"]

    parts = []
    for _, grp in df.groupby("cc_num"):
        s = grp.set_index("trans_date_trans_time")["amt"]
        r1, r24 = s.rolling("1h", closed="left"), s.rolling("24h", closed="left")
        parts.append(pd.DataFrame({"txn_1h": r1.count().values, "txn_24h": r24.count().values,
                                   "amt_sum_24h": r24.sum().values}, index=grp.index))
    roll = pd.concat(parts).loc[df.index]
    df[["txn_1h", "txn_24h", "amt_sum_24h"]] = roll.fillna(0).values

    df = df.replace([np.inf, -np.inf], np.nan)
    for c in ["time_since_last", "dist_from_prev", "speed_kmh", "amt_diff_prev"]:
        df[c] = df[c].fillna(0)
    df["amt_ratio"] = df["amt_ratio"].fillna(1)

    return df.sort_values("_row").reset_index(drop=True).drop(columns="_row")


# --------------------------------------------------------------------------- single transaction features
def build_single_row(i):
    """i = dict of values typed in the form. Returns a one-row DataFrame with the model features (unscaled)."""
    distance_km = float(haversine(i["lat"], i["long"], i["merch_lat"], i["merch_long"]))
    when = pd.Timestamp(i["when"])
    hour = when.hour

    row = {
        "amt": i["amt"], "hour": hour, "day_of_week": when.dayofweek, "month": when.month,
        "is_weekend": int(when.dayofweek >= 5), "is_night": int(hour in (23, 0, 1, 2, 3, 4)),
        "age": i["age"], "city_pop": i["city_pop"], "distance_km": distance_km,
        "category": i["category"], "gender": i["gender"], "state": i["state"],
    }

    if i["first_txn"]:                                   # same defaults the notebook uses for a card's first transaction
        row.update(dist_from_prev=0.0, speed_kmh=0.0, time_since_last=0.0, amt_diff_prev=0.0,
                   amt_ratio=1.0, txn_1h=0, txn_24h=0, amt_sum_24h=0.0)
    else:
        dist_prev = float(haversine(i["prev_merch_lat"], i["prev_merch_long"], i["merch_lat"], i["merch_long"]))
        secs = float(i["hours_since_last"]) * 3600
        hours = max(secs / 3600, 1 / 60)
        row.update(
            dist_from_prev=dist_prev, speed_kmh=dist_prev / hours, time_since_last=secs,
            amt_diff_prev=i["amt"] - i["prev_amt"],
            amt_ratio=i["amt"] / i["card_avg_amt"] if i["card_avg_amt"] > 0 else 1.0,
            txn_1h=i["txn_1h"], txn_24h=i["txn_24h"], amt_sum_24h=i["amt_sum_24h"],
        )
    return pd.DataFrame([row])


# --------------------------------------------------------------------------- prediction
def prepare_matrix(bundle, feat_df):
    """Scale + encode exactly like training. Returns the matrix the model really sees."""
    X = feat_df[bundle["features"]].copy()
    X[bundle["num_cols"]] = bundle["scaler"].transform(X[bundle["num_cols"]])
    X[bundle["cat_cols"]] = bundle["encoder"].transform(X[bundle["cat_cols"]])
    return X


def predict_scores(bundle, feat_df):
    """Return the model's fraud score for each row."""
    return bundle["model"].predict_proba(prepare_matrix(bundle, feat_df))[:, 1]


def risk_label(score, threshold):
    if score >= threshold:
        return "High"
    if score >= threshold * 0.5:
        return "Medium"
    return "Low"


def warning_signs(r):
    """Simple rule-based hints for humans. These are NOT the model's internal reasoning."""
    out = []
    if r["is_night"]:
        out.append("Transaction made at night (11 pm - 4 am)")
    if r["distance_km"] > 100:
        out.append(f"Merchant is {r['distance_km']:.0f} km away from the customer's home location")
    if r["dist_from_prev"] > 50 and r["speed_kmh"] > 800:
        out.append(f"Impossible travel: about {r['speed_kmh']:.0f} km/h from the previous transaction location")
    if r["amt_ratio"] > 5:
        out.append(f"Amount is {r['amt_ratio']:.1f}x this card's usual average")
    if r["txn_1h"] >= 3:
        out.append(f"{int(r['txn_1h'])} transactions on this card in the last hour")
    if r["txn_24h"] >= 8:
        out.append(f"{int(r['txn_24h'])} transactions on this card in the last 24 hours")
    if str(r["category"]).endswith("_net"):
        out.append("Online purchase category")
    return out


# --------------------------------------------------------------------------- SHAP (explainable AI)
def fraud_class_values(sv):
    """SHAP returns different shapes for different models. Always give back a 2-D array (rows x features)
    for the FRAUD class (class 1)."""
    if isinstance(sv, list):                       # older shap + sklearn trees: [class0, class1]
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:                               # newer shap: (rows, features, classes)
        sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
    return sv


def get_explainer(bundle):
    """TreeExplainer works for XGBoost / Random Forest / Decision Tree. Returns None for other models."""
    try:
        import shap
        return shap.TreeExplainer(bundle["model"])
    except Exception:
        return None


def shap_table(bundle, explainer, feat_df):
    """One row per transaction: SHAP value for every feature (positive = pushes towards FRAUD)."""
    X = prepare_matrix(bundle, feat_df)
    vals = fraud_class_values(explainer.shap_values(X))
    return pd.DataFrame(vals, columns=bundle["features"], index=feat_df.index)


def top_shap_reasons(shap_row, feat_row, n=6):
    """Top-n features by absolute SHAP impact, with the real (unscaled) feature value."""
    order = shap_row.abs().sort_values(ascending=False).index[:n]
    out = []
    for f in order:
        v = feat_row[f]
        out.append({"feature": f,
                    "value": round(float(v), 2) if isinstance(v, (int, float, np.integer, np.floating)) else str(v),
                    "impact": round(float(shap_row[f]), 4),
                    "direction": "towards FRAUD" if shap_row[f] > 0 else "towards GENUINE"})
    return out


# --------------------------------------------------------------------------- context for the chatbot
def build_chat_context(bundle, last):
    """Plain-text description of the project and the last checked transaction, sent to the chatbot."""
    txt = [
        "PROJECT: Debit/Credit card fraud detection using machine learning (B.Tech final-year academic project).",
        "Data: Sparkov simulated card transactions (is_fraud 1 = fraud, 0 = genuine; fraud is about 0.5% of rows).",
        f"Final model: {bundle['model_name']}. Imbalance handling: {bundle['balancing']}. "
        f"Decision threshold on the model score: {bundle['threshold']:.3f}.",
        "Model features: " + ", ".join(bundle["features"]) + ".",
        "Feature meanings: distance_km = customer home to merchant; dist_from_prev / speed_kmh = distance and speed from the "
        "previous transaction of the same card; amt_ratio = amount divided by the card's average past amount; "
        "txn_1h / txn_24h = number of transactions on the card in the last 1 / 24 hours; is_night = 11 pm to 4 am.",
        "SHAP: positive SHAP value pushes the score towards fraud, negative pushes towards genuine.",
    ]
    if last:
        txt += ["", "LAST CHECKED TRANSACTION:",
                f"- Result: {last['prediction']} (score {last['score'] * 100:.1f}%, risk level {last['risk']}, "
                f"threshold {bundle['threshold'] * 100:.1f}%)",
                "- Input values: " + ", ".join(f"{k}={v}" for k, v in last["features"].items()),
                "- Rule-based warning signs: " + ("; ".join(last["signs"]) if last["signs"] else "none")]
        if last.get("shap"):
            txt.append("- Top SHAP contributions: " + "; ".join(
                f"{r['feature']}={r['value']} ({r['direction']}, impact {r['impact']})" for r in last["shap"]))
    else:
        txt += ["", "No transaction has been checked yet in this session."]
    return "\n".join(txt)


# --------------------------------------------------------------------------- demo examples (form <-> real rows)
FORM_DEFAULTS = dict(
    amt=42.5, category="grocery_pos", date="2020-08-15", time="14:30", gender="F", age=38.0, state="NY",
    city_pop=120000, lat=40.71, long=-74.00, merch_lat=40.72, merch_long=-74.01, first_txn=False,
    hours_since_last=20.0, prev_merch_lat=40.71, prev_merch_long=-74.00, prev_amt=35.0, card_avg_amt=48.0,
    txn_1h=0, txn_24h=2, amt_sum_24h=80.0)


def form_values_from_row(r):
    """Turn one engineered transaction row into the values of the app's form (rounded like the form shows them)."""
    when = pd.Timestamp(r["trans_date_trans_time"])
    first = pd.isna(r.get("prev_amt"))
    f = dict(
        amt=round(float(r["amt"]), 2), category=str(r["category"]),
        date=when.strftime("%Y-%m-%d"), time=when.strftime("%H:%M"),
        gender=str(r["gender"]), age=round(float(r["age"]), 1), state=str(r["state"]),
        city_pop=int(r["city_pop"]), lat=round(float(r["lat"]), 4), long=round(float(r["long"]), 4),
        merch_lat=round(float(r["merch_lat"]), 4), merch_long=round(float(r["merch_long"]), 4),
        first_txn=bool(first))
    if first:
        f.update({k: FORM_DEFAULTS[k] for k in ["hours_since_last", "prev_merch_lat", "prev_merch_long",
                                                 "prev_amt", "card_avg_amt", "txn_1h", "txn_24h", "amt_sum_24h"]})
    else:
        f.update(hours_since_last=round(float(r["time_since_last"]) / 3600, 4),
                 prev_merch_lat=round(float(r["prev_merch_lat"]), 4), prev_merch_long=round(float(r["prev_merch_long"]), 4),
                 prev_amt=round(float(r["prev_amt"]), 2), card_avg_amt=round(float(r["card_avg_amt"]), 2),
                 txn_1h=int(r["txn_1h"]), txn_24h=int(r["txn_24h"]), amt_sum_24h=round(float(r["amt_sum_24h"]), 2))
    return f


def single_input_from_form(f):
    """Form values -> the input dict build_single_row() expects."""
    i = {k: v for k, v in f.items() if k not in ("date", "time")}
    i["when"] = pd.Timestamp(f"{f['date']} {f['time']}")
    return i
