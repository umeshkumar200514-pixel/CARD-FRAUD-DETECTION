"""
AI assistant for the fraud-detection app.
- With a Google Gemini API key -> answers in free text using the Gemini API (free tier available).
- With an Anthropic API key     -> answers in free text using the Claude API.
- Without a key                 -> a small rule-based helper that still explains the last result.
Gemini key: https://aistudio.google.com  (sidebar box or GEMINI_API_KEY environment variable)
Claude key: https://console.anthropic.com (sidebar box or ANTHROPIC_API_KEY environment variable)
"""

DEFAULT_MODEL = "claude-haiku-4-5-20251001"      # fast and cheap; you can also try "claude-sonnet-5-5"
GEMINI_MODEL = "gemini-2.5-flash-lite"           # check aistudio.google.com for the models currently on the free tier

SYSTEM_PROMPT = """You are the assistant inside a student machine-learning project that detects debit/credit card fraud.
Help the user understand the model, its predictions and fraud in general.

Rules:
- Use ONLY the project context below for facts about the model, the data and the last checked transaction. If something is not in the context, say you do not know.
- Never invent numbers, accuracy values or transaction details.
- Explain simply (the user is a student), in short answers. If the user writes in Hinglish or Hindi, reply in the same style.
- The model is an academic project trained on simulated data. A flag is a warning for review, not proof of fraud, and a 'genuine' result is not a guarantee.
- If the user asks what a cardholder should do after a flagged transaction, give practical safe steps (contact the bank, block the card, check recent transactions, change passwords). Do not ask for card numbers, PINs, OTPs or passwords.

PROJECT CONTEXT:
{context}
"""


def stream_reply(api_key, model, context, history):
    """Generator that yields the reply text piece by piece. history = [{'role': 'user'|'assistant', 'content': str}, ...]"""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(model=model, max_tokens=800,
                                system=SYSTEM_PROMPT.format(context=context),
                                messages=history) as stream:
        for text in stream.text_stream:
            yield text


def stream_reply_gemini(api_key, model, context, history):
    """Same as stream_reply, but uses the Google Gemini API (free tier available)."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    contents = [types.Content(role="model" if m["role"] == "assistant" else "user",
                              parts=[types.Part(text=m["content"])]) for m in history]
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT.format(context=context),
                                         max_output_tokens=800)
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        if chunk.text:
            yield chunk.text


def offline_reply(question, last, bundle):
    """Very small rule-based fallback used when no API key is given."""
    q = question.lower()
    if any(w in q for w in ["why", "reason", "flag", "kyun", "kyon", "explain"]):
        if not last:
            return "Pehle 'Check one transaction' page par ek transaction check karo, phir main reason bata paunga."
        lines = [f"Result: **{last['prediction']}** (score {last['score'] * 100:.1f}%, risk {last['risk']})."]
        if last.get("shap"):
            lines.append("Model ke hisab se sabse zyada asar wale features:")
            lines += [f"- {r['feature']} = {r['value']} -> {r['direction']}" for r in last["shap"][:4]]
        if last["signs"]:
            lines.append("Simple rule-based warning signs: " + "; ".join(last["signs"]))
        return "\n".join(lines)
    if "threshold" in q:
        return (f"Threshold {bundle['threshold']:.3f} hai. Score is se upar ya barabar ho to transaction fraud flag hota hai. "
                "Threshold kam karoge to zyada fraud pakdoge (recall badhega) par false alarms bhi badhenge.")
    if "shap" in q:
        return "SHAP batata hai har feature ne score ko fraud ki taraf (positive) ya genuine ki taraf (negative) kitna dhakela."
    if any(w in q for w in ["smote", "balanc", "imbalance"]):
        return f"Is project mein balancing method: {bundle['balancing']}. Fraud bahut kam hota hai, isliye model ko fraud seekhne ke liye balancing chahiye."
    return ("Offline mode mein main sirf 'why flagged', 'threshold', 'SHAP' aur 'balancing' jaise sawal samajh paata hun. "
            "Poori chat ke liye sidebar mein Gemini (free) ya Anthropic API key daalo.")
