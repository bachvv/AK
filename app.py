import os
import json
import math
import yfinance as yf
import pandas as pd
import streamlit as st
from openai import OpenAI

# ====================================================
# LOAD API KEY (Mode A: config.py or environment or Streamlit secrets)
# ====================================================
try:
    from config import OPENAI_API_KEY
except:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")  # Streamlit Cloud Secrets

MODEL_NAME = "gpt-5.1"

# ====================================================
# STREAMLIT CONFIG
# ====================================================
st.set_page_config(page_title="Adam Khoo 12-Criteria – Hybrid LLM", layout="wide")

# ====================================================
# LLM PROMPT
# ====================================================
LLM_SYSTEM_PROMPT = """
You are an equity analyst using Adam Khoo's 12-criteria Value Momentum Investing framework.

You will ONLY score the QUALITATIVE criteria:
1) Competitive advantage / moat
3) Large & growing market
4) Management quality (competent & honest)
5) Insider ownership / alignment
6) Low dilution (no constant share issuance)

Task:
Assign:
- score_0_10 (float)
- pass (bool)
- evidence (short explanation)
- notes (nuance)

Score ≥ 7 = pass.

Return ONLY JSON:

{
  "ticker": "",
  "criteria": [
    {"id":1,"name":"Competitive advantage / moat","score_0_10":0,"pass":false,"evidence":"","notes":""},
    {"id":3,"name":"Large & growing market","score_0_10":0,"pass":false,"evidence":"","notes":""},
    {"id":4,"name":"Management quality (competent & honest)","score_0_10":0,"pass":false,"evidence":"","notes":""},
    {"id":5,"name":"Insider ownership / alignment","score_0_10":0,"pass":false,"evidence":"","notes":""},
    {"id":6,"name":"Low dilution (no constant share issuance)","score_0_10":0,"pass":false,"evidence":"","notes":""}
  ]
}
"""

# ====================================================
# FETCH YFINANCE
# ====================================================
def get_data(ticker):
    tk = yf.Ticker(ticker)
    return {
        "info": getattr(tk, "info", {}) or {},
        "fin": tk.financials if hasattr(tk, "financials") else pd.DataFrame(),
        "bs": tk.balance_sheet if hasattr(tk, "balance_sheet") else pd.DataFrame(),
        "cf": tk.cashflow if hasattr(tk, "cashflow") else pd.DataFrame(),
        "hist": tk.history(period="1y")
    }

# ====================================================
# METRICS
# ====================================================
def cagr(series):
    if series is None or len(series) < 2:
        return None
    try:
        start = float(series.iloc[-1])
        end = float(series.iloc[0])
        if start <= 0:
            return None
        years = len(series) - 1
        return (end / start) ** (1 / years) - 1
    except:
        return None

def compute_metrics(data):
    info = data["info"]
    fin, bs, cf, hist = data["fin"], data["bs"], data["cf"], data["hist"]
    m = {}

    # Revenue CAGR
    try:
        rev = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else None
        m["cagr"] = cagr(rev)
    except:
        m["cagr"] = None

    m["roe"] = info.get("returnOnEquity")
    m["net_margin"] = info.get("profitMargins")
    m["pe"] = info.get("trailingPE")

    # Debt
    debt, equity = info.get("totalDebt"), info.get("totalStockholderEquity")
    m["dte"] = debt / equity if equity not in (None, 0) else None

    # Current ratio
    try:
        col = bs.columns[0]
        m["cr"] = bs.loc["Total Current Assets"][col] / bs.loc["Total Current Liabilities"][col]
    except:
        m["cr"] = None

    # FCF margin
    try:
        col = cf.columns[0]
        op = cf.loc["Total Cash From Operating Activities"][col]
        cap = cf.loc["Capital Expenditures"][col]
        fcf = op + cap
        latest_rev = rev.iloc[0]
        m["fcf_margin"] = fcf / latest_rev if latest_rev != 0 else None
    except:
        m["fcf_margin"] = None

    # Trend
    try:
        close = hist["Close"]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma150 = close.rolling(150).mean().iloc[-1]
        m["trend"] = close.iloc[-1] > ma50 > ma150
    except:
        m["trend"] = None

    return m

# ====================================================
# RULE SCORES FOR 2,7,8,9,10,11,12
# ====================================================
def rule_score(val, good, ok):
    if val is None:
        return 5
    if val >= good:
        return 9
    if val >= ok:
        return 7
    return 3

def quantitative(metrics):
    c = []

    # 2 Recurring revenue
    s = 5 if metrics["cagr"] is None else rule_score(metrics["cagr"], 0.10, 0.05)
    c.append({"id":2,"name":"Recurring revenue","score_0_10":s,"pass":s>=7,
              "evidence": f"CAGR: {metrics['cagr']:.2%}" if metrics["cagr"] else "Missing"})

    # 7 Growth
    s = rule_score(metrics["cagr"], 0.10, 0.05)
    c.append({"id":7,"name":"Revenue growth","score_0_10":s,"pass":s>=7})

    # 8 FCF
    f = metrics["fcf_margin"]
    s = rule_score(f, 0.10, 0.05)
    c.append({"id":8,"name":"FCF strength","score_0_10":s,"pass":s>=7,
              "evidence": f"FCF margin: {f:.2%}" if f else "Missing"})

    # 9 Profitability
    roe, nm = metrics["roe"], metrics["net_margin"]
    if roe and nm:
        if roe>=0.15 and nm>=0.10: s=9
        elif roe>=0.10 and nm>=0.05: s=7
        else: s=4
    else:
        s=5
    c.append({"id":9,"name":"Profitability","score_0_10":s,"pass":s>=7})

    # 10 Balance sheet
    dte, cr = metrics["dte"], metrics["cr"]
    if dte and cr:
        if dte<=1 and cr>=1.5: s=9
        elif dte<=2 and cr>=1: s=7
        else: s=4
    else:
        s=6
    c.append({"id":10,"name":"Balance sheet","score_0_10":s,"pass":s>=7})

    # 11 Valuation
    pe = metrics["pe"]
    if pe:
        if pe<=15: s=9
        elif pe<=25: s=7
        else: s=4
    else: s=5
    c.append({"id":11,"name":"Valuation","score_0_10":s,"pass":s>=7})

    # 12 Trend
    t = metrics["trend"]
    s = 9 if t else 4 if t is False else 5
    c.append({"id":12,"name":"Technical trend","score_0_10":s,"pass":s>=7})

    return c

# ====================================================
# LLM qualitative scoring (1,3,4,5,6)
# ====================================================
def llm_prompt(ticker, info, metrics):
    subset = {
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "summary": info.get("longBusinessSummary"),
        "marketCap": info.get("marketCap"),
        "profitMargins": info.get("profitMargins"),
        "roe": info.get("returnOnEquity"),
        "debt": info.get("totalDebt"),
        "shares": info.get("sharesOutstanding"),
    }
    return (
        f"Ticker: {ticker}\n"
        f"Info: {json.dumps(subset, default=str)}\n"
        f"Metrics: {json.dumps(metrics, default=str)}"
    )

def llm_score(api_key, ticker, info, metrics):
    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role":"system","content":LLM_SYSTEM_PROMPT},
            {"role":"user","content":llm_prompt(ticker,info,metrics)}
        ],
        response_format={"type":"json_object"},
    )
    return json.loads(resp.output[0].content[0].text)["criteria"]

# ====================================================
# STREAMLIT UI
# ====================================================
st.title("📊 Adam Khoo 12-Criteria — Hybrid LLM + Rules")
st.caption("OpenAI key auto-loaded (Mode A).")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    run = st.button("Evaluate")

if run:
    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ No API key found. Add OPENAI_API_KEY to Streamlit Secrets.")
        st.stop()

    with st.spinner("Fetching data..."):
        data = get_data(ticker)
        metrics = compute_metrics(data)

    with st.spinner("LLM evaluating qualitative criteria..."):
        qual = llm_score(api_key, ticker, data["info"], metrics)

    quant = quantitative(metrics)

    combined = {c["id"]: c for c in qual + quant}
    ordered = [combined[i] for i in sorted(combined.keys())]

    scores = [c["score_0_10"] for c in ordered]
    overall = sum(scores) / len(scores)

    st.subheader(f"📌 Results for {ticker}")
    st.metric("Overall Score", f"{overall:.2f}")

    # Score color
    if overall >= 8:
        color = "#06c258"
    elif overall >= 5:
        color = "#ffcc00"
    else:
        color = "#ff4d4d"

    st.markdown(
        f"""<div style="background:{color};padding:12px;border-radius:8px;color:white;text-align:center;">
            {'Strong' if overall>=8 else 'Mixed' if overall>=5 else 'Weak'} candidate
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### ✅ Full Checklist")
    for c in ordered:
        cid, name, score, passed = c["id"], c["name"], c["score_0_10"], c["pass"]

        if passed:
            icon = "✅"; bg = "#e8ffe8"; border = "#0f9d58"
        elif score >= 5:
            icon = "⚠️"; bg = "#fff7cc"; border = "#ffcc00"
        else:
            icon = "❌"; bg = "#ffe6e6"; border = "#cc0000"

        st.markdown(
            f"""
            <div style="border-left:6px solid {border};background:{bg};padding:12px;margin-bottom:10px;border-radius:6px;">
                <h4>{icon} {cid}. {name}
                <span style="float:right">Score: {score}</span></h4>
                <p><b>Evidence:</b> {c.get("evidence","")}</p>
                {("<p><b>Notes:</b> "+c.get("notes","")+"</p>") if c.get("notes") else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )
