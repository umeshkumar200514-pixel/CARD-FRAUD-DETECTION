import datetime as dt
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from chatbot import DEFAULT_MODEL, GEMINI_MODEL, offline_reply, stream_reply, stream_reply_gemini
from fraud_utils import (FORM_DEFAULTS, RAW_REQUIRED, build_chat_context, build_single_row, engineer_batch, get_explainer,
                         load_bundle, predict_scores, risk_label, shap_table, single_input_from_form,
                         top_shap_reasons, warning_signs)

MODEL_PATH = "models/best_model.pkl"
DEMO_PATH = "models/demo_examples.json"
st.set_page_config(page_title="Card Fraud Detection", page_icon="💳", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle(MODEL_PATH)


@st.cache_resource
def get_shap(_bundle):
    return get_explainer(_bundle)


if not os.path.exists(MODEL_PATH):
    st.error("models/best_model.pkl not found. Run the training notebook first and keep the 'models' folder next to app.py.")
    st.stop()

bundle = get_bundle()
THR = float(bundle["threshold"])

# ------------------------------------------------------------------ sidebar
st.sidebar.title("💳 Card Fraud Detection")
page = st.sidebar.radio("Page", ["Check one transaction", "Batch check (CSV)", "AI assistant", "Model & dashboard"])
st.sidebar.markdown("---")
st.sidebar.write(f"**Model:** {bundle['model_name']}")
st.sidebar.write(f"**Balancing:** {bundle['balancing']}")
st.sidebar.write(f"**Decision threshold:** {THR:.3f}")
st.sidebar.caption("A transaction is flagged as fraud when its score is at or above the threshold.")

enc_categories = dict(zip(bundle["cat_cols"], bundle["encoder"].categories_))
CATEGORIES = list(enc_categories["category"])
GENDERS = list(enc_categories["gender"])
STATES = list(enc_categories["state"])


def pick(options, wanted):
    return options.index(wanted) if wanted in options else 0


# ------------------------------------------------------------------ examples for the demo
EXAMPLES = {
    "Custom (start from a normal purchase)": {"values": dict(FORM_DEFAULTS)},
    "Normal daytime purchase": {"values": dict(FORM_DEFAULTS)},
    "Suspicious: night, far away, big amount": {"values": dict(
        FORM_DEFAULTS, amt=1150.0, category="shopping_net", time="02:30", merch_lat=34.05, merch_long=-118.24,
        hours_since_last=0.3, prev_amt=40.0, card_avg_amt=45.0, txn_1h=3, txn_24h=6, amt_sum_24h=2300.0)},
}
if os.path.exists(DEMO_PATH):                      # REAL test-set transactions saved by train_model.py
    with open(DEMO_PATH) as fh:
        for e in json.load(fh)["examples"]:
            EXAMPLES[e["name"]] = e

# ====================================================================== PAGE 1
if page == "Check one transaction":
    st.title("Check one transaction")
    st.caption("Fill in the transaction details, or pick an example. The model decides on its own.")
    if not os.path.exists(DEMO_PATH):
        st.info("Run `python train_model.py` again to add REAL fraud examples from the test set to this list.")

    example_name = st.selectbox("Quick example", list(EXAMPLES.keys()))
    chosen = EXAMPLES[example_name]
    ex = dict(FORM_DEFAULTS, **chosen["values"])
    if chosen.get("actual"):
        st.caption(f"This is a real transaction from fraudTest.csv that the model never saw during training. "
                   f"Its true label in the dataset is **{chosen['actual']}**.")
    k = str(abs(hash(example_name)))                   # new widget keys when the example changes

    with st.form("txn_form"):
        st.subheader("Transaction")
        c1, c2, c3 = st.columns(3)
        card_type = c1.selectbox("Card type", ["Credit", "Debit"], key=f"ct_{k}",
                                 help="For display only: the dataset has no card-type column, so the model does not use it.")
        amt = c2.number_input("Amount", min_value=0.0, value=float(ex["amt"]), step=1.0, format="%.2f", key=f"amt_{k}")
        category = c3.selectbox("Merchant category", CATEGORIES, index=pick(CATEGORIES, ex["category"]), key=f"cat_{k}")

        c1, c2 = st.columns(2)
        date = c1.date_input("Date", value=dt.date.fromisoformat(ex["date"]),
                             min_value=dt.date(2018, 1, 1), max_value=dt.date(2030, 12, 31), key=f"d_{k}")
        time = c2.time_input("Time", value=dt.time.fromisoformat(ex["time"]), step=60, key=f"t_{k}")

        st.subheader("Customer")
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", GENDERS, index=pick(GENDERS, ex["gender"]), key=f"g_{k}")
        age = c2.number_input("Age", min_value=10.0, max_value=110.0, value=float(ex["age"]), step=1.0, format="%.1f", key=f"age_{k}")
        state = c3.selectbox("State", STATES, index=pick(STATES, ex["state"]), key=f"st_{k}")
        city_pop = c4.number_input("City population", min_value=0, value=int(ex["city_pop"]), step=1000, key=f"cp_{k}")

        st.subheader("Location")
        c1, c2, c3, c4 = st.columns(4)
        lat = c1.number_input("Customer latitude", value=float(ex["lat"]), format="%.4f", key=f"la_{k}")
        long = c2.number_input("Customer longitude", value=float(ex["long"]), format="%.4f", key=f"lo_{k}")
        merch_lat = c3.number_input("Merchant latitude", value=float(ex["merch_lat"]), format="%.4f", key=f"ml_{k}")
        merch_long = c4.number_input("Merchant longitude", value=float(ex["merch_long"]), format="%.4f", key=f"mo_{k}")

        st.subheader("This card's recent history")
        first_txn = st.checkbox("This is the first transaction of this card (no history)", value=bool(ex["first_txn"]), key=f"ft_{k}")
        c1, c2, c3, c4 = st.columns(4)
        hours_since_last = c1.number_input("Hours since previous transaction", min_value=0.0, value=float(ex["hours_since_last"]), format="%.4f", key=f"hs_{k}")
        prev_amt = c2.number_input("Previous transaction amount", min_value=0.0, value=float(ex["prev_amt"]), format="%.2f", key=f"pa_{k}")
        card_avg_amt = c3.number_input("Card's average amount", min_value=0.0, value=float(ex["card_avg_amt"]), format="%.2f", key=f"av_{k}")
        amt_sum_24h = c4.number_input("Total spent in last 24 h", min_value=0.0, value=float(ex["amt_sum_24h"]), format="%.2f", key=f"s24_{k}")
        c1, c2, c3, c4 = st.columns(4)
        txn_1h = c1.number_input("Transactions in last 1 h", min_value=0, value=int(ex["txn_1h"]), key=f"t1_{k}")
        txn_24h = c2.number_input("Transactions in last 24 h", min_value=0, value=int(ex["txn_24h"]), key=f"t24_{k}")
        prev_merch_lat = c3.number_input("Previous merchant latitude", value=float(ex["prev_merch_lat"]), format="%.4f", key=f"pl_{k}")
        prev_merch_long = c4.number_input("Previous merchant longitude", value=float(ex["prev_merch_long"]), format="%.4f", key=f"po_{k}")

        submitted = st.form_submit_button("Check transaction", type="primary")

    if submitted:
        form = dict(amt=amt, category=category, date=date.isoformat(), time=time.strftime("%H:%M"), gender=gender,
                    age=age, state=state, city_pop=city_pop, lat=lat, long=long, merch_lat=merch_lat,
                    merch_long=merch_long, first_txn=first_txn, hours_since_last=hours_since_last,
                    prev_merch_lat=prev_merch_lat, prev_merch_long=prev_merch_long, prev_amt=prev_amt,
                    card_avg_amt=card_avg_amt, txn_1h=txn_1h, txn_24h=txn_24h, amt_sum_24h=amt_sum_24h)
        feats = build_single_row(single_input_from_form(form))
        score = float(predict_scores(bundle, feats)[0])
        risk = risk_label(score, THR)
        signs = warning_signs(feats.iloc[0])

        st.markdown("---")
        left, right = st.columns([1, 2])
        with left:
            st.metric("Fraud score", f"{score * 100:.1f}%")
            st.progress(min(max(score, 0.0), 1.0))
            st.write(f"**Risk level:** {risk}")
            st.caption(f"{card_type} card. Score is the model's output, not a guaranteed real-world probability. "
                       f"Flagged when score >= {THR * 100:.1f}%.")
        with right:
            if score >= THR:
                st.error("⚠ FRAUDULENT transaction (flagged)")
            else:
                st.success("✅ GENUINE transaction")
            if chosen.get("actual"):
                ok = (chosen["actual"] == "FRAUD") == (score >= THR)
                st.write(f"**True label in the dataset:** {chosen['actual']}  "
                         f"{'✔ model is correct' if ok else '✘ model is wrong here'}")
            st.write("**Warning signs found (simple rules, not the model's exact reasoning):**")
            if signs:
                for s in signs:
                    st.write(f"- {s}")
            else:
                st.write("- No obvious warning signs from the simple rules")

        # ---------------- SHAP explanation ----------------
        st.subheader("Why did the model give this score? (SHAP)")
        shap_info = None
        explainer = get_shap(bundle)
        if explainer is None:
            st.info("SHAP explanations need the `shap` package and a tree model (XGBoost / Random Forest / Decision Tree).")
        else:
            try:
                sh = shap_table(bundle, explainer, feats).iloc[0]
                shap_info = top_shap_reasons(sh, feats.iloc[0], n=6)
                top = sh.reindex(sh.abs().sort_values(ascending=False).index[:10])[::-1]
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh([f"{f} = {feats.iloc[0][f]:.2f}" if isinstance(feats.iloc[0][f], (int, float, np.number))
                         else f"{f} = {feats.iloc[0][f]}" for f in top.index],
                        top.values, color=["#E45756" if v > 0 else "#4C78A8" for v in top.values])
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Impact on fraud score  (red = towards FRAUD, blue = towards GENUINE)")
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("Bars show how much each feature pushed this transaction's score up (red) or down (blue). "
                           "Units depend on the model (log-odds for XGBoost, probability for tree ensembles).")
            except Exception as e:
                st.warning(f"Could not compute SHAP values: {e}")

        # remembered for the AI assistant page
        st.session_state["last"] = dict(
            prediction="Fraud" if score >= THR else "Genuine", score=score, risk=risk, signs=signs, shap=shap_info,
            features={k: (round(float(v), 2) if isinstance(v, (int, float, np.number)) else str(v))
                      for k, v in feats.iloc[0].items()})
        st.info("Want a plain-language explanation? Open the **AI assistant** page and ask 'why was this flagged?'")

        with st.expander("Features sent to the model"):
            st.dataframe(feats.T.rename(columns={0: "value"}).astype(str))

# ====================================================================== PAGE 2
elif page == "Batch check (CSV)":
    st.title("Batch check (CSV)")
    st.write("Upload a CSV with the Sparkov columns (for example a slice of `fraudTest.csv`).")
    st.caption("Required columns: " + ", ".join(RAW_REQUIRED) + ". Optional: `is_fraud` (to see accuracy metrics).")
    st.info("History features (transactions in the last hour, distance from previous transaction, etc.) are calculated "
            "from the rows inside your file, so a file with many rows per card gives the most reliable results.")

    up = st.file_uploader("Upload CSV", type="csv")
    use_demo = False
    if up is None and os.path.exists("reports/demo_batch.csv"):
        use_demo = st.button("Or use the demo file (real test transactions of cards that had fraud)")
    if up is not None or use_demo:
        raw = pd.read_csv(up if up is not None else "reports/demo_batch.csv")
        if "Unnamed: 0" in raw.columns:
            raw = raw.drop(columns="Unnamed: 0")
        missing = [c for c in RAW_REQUIRED if c not in raw.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
            st.stop()

        with st.spinner("Building features and predicting..."):
            feats = engineer_batch(raw)
            scores = predict_scores(bundle, feats)

        out = raw.copy()
        out["fraud_score"] = np.round(scores, 4)
        out["prediction"] = np.where(scores >= THR, "Fraud", "Genuine")
        out["risk"] = [risk_label(s, THR) for s in scores]

        n_fraud = int((scores >= THR).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions checked", f"{len(out):,}")
        c2.metric("Flagged as fraud", f"{n_fraud:,}")
        c3.metric("Flagged %", f"{n_fraud / len(out) * 100:.2f}%")

        if "is_fraud" in raw.columns:
            y, pred = raw["is_fraud"].values, (scores >= THR).astype(int)
            st.subheader("Check against the real labels in your file")
            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{precision_score(y, pred, zero_division=0):.3f}")
            m2.metric("Recall", f"{recall_score(y, pred, zero_division=0):.3f}")
            m3.metric("F1", f"{f1_score(y, pred, zero_division=0):.3f}")
            cm = confusion_matrix(y, pred, labels=[0, 1])
            st.dataframe(pd.DataFrame(cm, index=["Actual genuine", "Actual fraud"],
                                      columns=["Predicted genuine", "Predicted fraud"]))

        st.subheader("Highest-risk transactions")
        st.dataframe(out.sort_values("fraud_score", ascending=False).head(50))
        st.download_button("Download all predictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="fraud_predictions.csv", mime="text/csv")

# ====================================================================== PAGE 3 (AI assistant)
elif page == "AI assistant":
    st.title("🤖 AI assistant")
    st.sidebar.markdown("---")
    st.sidebar.subheader("AI assistant settings")
    provider = st.sidebar.radio("AI provider", ["Gemini (free tier)", "Claude"])
    if provider.startswith("Gemini"):
        api_key = st.sidebar.text_input("Gemini API key (optional)", type="password",
                                        value=os.environ.get("GEMINI_API_KEY", ""),
                                        help="Free key at aistudio.google.com. Without a key, a small offline helper answers.")
        chat_model = st.sidebar.text_input("Gemini model", value=GEMINI_MODEL)
        streamer = stream_reply_gemini
    else:
        api_key = st.sidebar.text_input("Anthropic API key (optional)", type="password",
                                        value=os.environ.get("ANTHROPIC_API_KEY", ""),
                                        help="Get one at console.anthropic.com. Without a key, a small offline helper answers.")
        chat_model = st.sidebar.text_input("Claude model", value=DEFAULT_MODEL)
        streamer = stream_reply

    last = st.session_state.get("last")
    if last:
        st.info(f"The assistant knows your last checked transaction: **{last['prediction']}** "
                f"(score {last['score'] * 100:.1f}%, risk {last['risk']}).")
    else:
        st.info("Tip: check a transaction first, then ask 'why was this flagged?'. You can also ask about the model, SHAP, SMOTE or fraud in general.")
    if not api_key:
        st.caption("No API key: running in offline mode (limited answers).")

    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask about a result, the model, SHAP or card fraud...")
    if question:
        st.session_state["chat"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            if api_key:
                try:
                    reply = st.write_stream(streamer(api_key, chat_model, build_chat_context(bundle, last),
                                                     st.session_state["chat"]))
                except Exception as e:
                    st.error(f"Could not reach the {provider} API: {e}")
                    st.session_state["chat"].pop()          # keep the history valid
                    reply = None
            else:
                reply = offline_reply(question, last, bundle)
                st.markdown(reply)
        if reply:
            st.session_state["chat"].append({"role": "assistant", "content": reply})

    if st.session_state["chat"] and st.button("Clear chat"):
        st.session_state["chat"] = []
        st.rerun()

# ====================================================================== PAGE 4
else:
    st.title("Model & dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Final model", bundle["model_name"])
    c2.metric("Balancing method", bundle["balancing"])
    c3.metric("Threshold", f"{THR:.3f}")

    st.subheader("Model comparison (test set)")
    if os.path.exists("reports/model_comparison_test.csv"):
        st.dataframe(pd.read_csv("reports/model_comparison_test.csv"))
    else:
        st.info("reports/model_comparison_test.csv not found.")
    if os.path.exists("reports/model_comparison.png"):
        st.image("reports/model_comparison.png")

    st.subheader("Balancing methods compared (validation set)")
    if os.path.exists("reports/balancing_comparison_validation.csv"):
        st.dataframe(pd.read_csv("reports/balancing_comparison_validation.csv"))

    cols = st.columns(2)
    for col, (title, path) in zip(cols, [("Confusion matrix (test)", "reports/confusion_matrix.png"),
                                         ("Feature importance", "reports/feature_importance.png")]):
        with col:
            st.subheader(title)
            if os.path.exists(path):
                st.image(path)
            else:
                st.info(f"{path} not found.")

    if os.path.exists("reports/shap_summary.png") or os.path.exists("reports/shap_bar.png"):
        st.subheader("SHAP: which features matter most (whole test sample)")
        cols = st.columns(2)
        for col, path in zip(cols, ["reports/shap_summary.png", "reports/shap_bar.png"]):
            if os.path.exists(path):
                col.image(path)

    if os.path.exists("reports/eda_overview.png"):
        st.subheader("Data overview (EDA)")
        st.image("reports/eda_overview.png")

    st.subheader("Data balancing (SMOTE)")
    if os.path.exists("reports/balancing_summary.json"):
        with open("reports/balancing_summary.json") as fh:
            bs = json.load(fh)
        st.write(f"Method used for the final model: **{bs['method']}**")
        before = pd.DataFrame({"Genuine": [bs["original_fit_rows"] - bs["original_fit_fraud"], bs["balanced_genuine"]],
                               "Real fraud": [bs["original_fit_fraud"], bs["balanced_fraud"] - bs["synthetic_fraud"]],
                               "Synthetic fraud (SMOTE)": [0, bs["synthetic_fraud"]]},
                              index=["Before balancing", "After balancing"])
        st.dataframe(before)
        st.bar_chart(before)
        st.caption("Balancing is applied to the training part only. Validation and test keep the real fraud rate.")
    if os.path.exists("reports/smote_synthetic_examples.csv"):
        with st.expander("Examples of synthetic fraud transactions created by SMOTENC (real units)"):
            st.dataframe(pd.read_csv("reports/smote_synthetic_examples.csv"))
            st.caption("These rows do not exist in the dataset. SMOTENC created them between real fraud transactions "
                       "to give the model more fraud examples to learn from. They are used for training only.")

    if os.path.exists(DEMO_PATH):
        st.subheader("Real test-set transactions used in the demo")
        with open(DEMO_PATH) as fh:
            demo = json.load(fh)["examples"]
        st.dataframe(pd.DataFrame([dict(example=e["name"], true_label=e["actual"],
                                         model_score=f"{e['score'] * 100:.1f}%", **e["values"]) for e in demo]).astype(str))

    with st.expander("Features used by the model"):
        st.write(", ".join(bundle["features"]))
