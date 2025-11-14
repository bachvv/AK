import os
import json
import math

import yfinance as yf
import pandas as pd
import streamlit as st
from openai import OpenAI

# ====================================================
# CONFIG / API KEY (Mode A: no UI box)
# ====================================================
try:
    from config import OPENAI_API_KEY as CONFIG_KEY
except ImportError:
    CONFIG_KEY = None

MODEL_NAME = "gpt-4.1"  # change if needed (e.g., "gpt-4o-mini")


# ====================================================
# STREAMLIT PAGE CONFIG
# ====================================================
st.set_page_config(
    page_title="Adam Khoo 12-Criteria – Hybrid LLM",
    layout="wide",
)


# ====================================================
# LLM SYSTEM PROMPT (QUALITATIVE 1,3,4,5,6)
# ====================================================
LLM_SYSTEM_PROMPT = """
You are an equity analyst using Adam Khoo's 12-criteria Value Momentum Investing framework.

You will ONLY score the QUALITATIVE criteria:
1) Competitive advantage / moat
3) Large & growing market
4) Management quality (competent & honest)
5) Insider ownership / alignment
6) Low dilution (no constant share issuance)

You are given:
- Basic company info (sector, industry, business summary, etc.)
- Precomputed metrics (revenue CAGR, ROE, margins, debt/equity, FCF margin, etc.)

For each of the 5 qualitative criteria:
- Assign score_0_10 (float from 0 to 10)
- pass (bool; True if score_0_10 >= 7)
- evidence (short explanation, referencing info/metrics)
- notes (optional nuance or uncertainty)

Scoring guide:
0–3 = Very weak
4–6 = Mixed / unclear
7–8 = Solid
9–10 = Excellent

Return ONLY this JSON structure:

{
  "ticker": "",
  "criteria": [
    {
      "id": 1,
      "name": "Competitive advantage / moat",
      "score_0_10": 0.0,
      "pass": false,
      "evidence": "",
      "notes": ""
    },
    {
      "id": 3,
      "name": "Large & growing market",
      "score_0_10": 0.0,
      "pass": false,
      "evidence": "",
      "notes": ""
    },
    {
      "id": 4,
      "name": "Management quality (competent & honest)",
      "score_0_10": 0.0,
      "pass": false,
      "evidence": "",
      "notes": ""
    },
    {
      "id": 5,
      "name": "Insider ownership / alignment",
      "score_0_10": 0.0,
      "pass": false,
      "evidence": "",
      "notes": ""
    },
    {
      "id": 6,
      "name": "Low dilution (no constant share issuance)",
      "score_0_10": 0.0,
      "pass": false,
      "evidence": "",
      "notes": ""
    }
  ]
}
"""


# ====================================================
# YFINANCE HELPERS
# ====================================================
def get_data(ticker: str):
    tk = yf.Ticker(ticker)
    info = getattr(tk, "info", {}) or {}
    fin = tk.financials if hasattr(tk, "financials") else pd.DataFrame()
    bs = tk.balance_sheet if hasattr(tk, "balance_sheet") else pd.DataFrame()
    cf = tk.cashflow if hasattr(tk, "cashflow") else pd.DataFrame()
    hist = tk.history(period="1y")
    return {"info": info, "fin": fin, "bs": bs, "cf": cf, "hist": hist}


def cagr(series: pd.Series):
    if series is None or len(series) < 2:
        return None
    try:
        start = float(series.iloc[-1])
        end = float(series.iloc[0])
        if start <= 0:
            return None
        years = len(series) - 1
        return (end / start) ** (1 / years) - 1
    except Exception:
        return None


def compute_metrics(data: dict):
    info = data["info"]
    fin = data["fin"]
    bs = data["bs"]
    cf = data["cf"]
    hist = data["hist"]

    metrics = {}

    # Revenue CAGR
    rev = None
    if not fin.empty:
        if "Total Revenue" in fin.index:
            rev = fin.loc["Total Revenue"]
        elif "TotalRevenue" in fin.index:
            rev = fin.loc["TotalRevenue"]
    metrics["cagr"] = cagr(rev)

    # Basic profitability & valuation
    metrics["roe"] = info.get("returnOnEquity")
    metrics["net_margin"] = info.get("profitMargins")
    metrics["pe"] = info.get("trailingPE")

    # Debt to equity
    debt = info.get("totalDebt")
    equity = info.get("totalStockholderEquity")
    metrics["dte"] = debt / equity if equity not in (None, 0) and debt is not None else None

    # Current ratio
    cr = None
    try:
        col = bs.columns[0]
        ca = bs.loc["Total Current Assets"][col]
        cl = bs.loc["Total Current Liabilities"][col]
        cr = ca / cl if cl != 0 else None
    except Exception:
        pass
    metrics["cr"] = cr

    # FCF margin
    fcf_margin = None
    try:
        col = cf.columns[0]
        op = cf.loc["Total Cash From Operating Activities"][col]
        cap = cf.loc["Capital Expenditures"][col]
        fcf = op + cap
        if rev is not None:
            latest_rev = rev.iloc[0]
            if latest_rev != 0:
                fcf_margin = fcf / latest_rev
    except Exception:
        pass
    metrics["fcf_margin"] = fcf_margin

    # Trend (price > 50DMA > 150DMA)
    trend = None
    try:
        if not hist.empty:
            close = hist["Close"]
            ma50 = close.rolling(50).mean().iloc[-1]
            ma150 = close.rolling(150).mean().iloc[-1]
            trend = close.iloc[-1] > ma50 > ma150
    except Exception:
        pass
    metrics["trend"] = trend

    return metrics


# ====================================================
# QUANTITATIVE SCORING (2,7,8,9,10,11,12)
# ====================================================
def score_ratio(val, good, ok):
    if val is None:
        return 5
    if val >= good:
        return 9
    if val >= ok:
        return 7
    return 3


def quantitative(metrics: dict):
    c = []

    # 2) Recurring revenue (proxy: positive decent CAGR)
    cagr_val = metrics["cagr"]
    s = 5 if cagr_val is None else score_ratio(cagr_val, 0.10, 0.05)
    c.append({
        "id": 2,
        "name": "Recurring / predictable revenue (proxy via CAGR)",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"CAGR: {cagr_val:.2%}" if cagr_val is not None else "Missing / not available",
        "notes": "Approximation: true business model still needs qualitative check."
    })

    # 7) Revenue growth
    s = score_ratio(cagr_val, 0.10, 0.05)
    c.append({
        "id": 7,
        "name": "Revenue growth",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"CAGR: {cagr_val:.2%}" if cagr_val is not None else "Missing / not available",
        "notes": ""
    })

    # 8) FCF strength
    f = metrics["fcf_margin"]
    s = score_ratio(f, 0.10, 0.05)
    c.append({
        "id": 8,
        "name": "Earnings & free cash flow",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"FCF margin: {f:.2%}" if f is not None else "Missing / not available",
        "notes": ""
    })

    # 9) Profitability
    roe = metrics["roe"]
    nm = metrics["net_margin"]
    if roe is not None and nm is not None:
        if roe >= 0.15 and nm >= 0.10:
            s = 9
        elif roe >= 0.10 and nm >= 0.05:
            s = 7
        else:
            s = 4
    else:
        s = 5
    c.append({
        "id": 9,
        "name": "Profitability (ROE & margins)",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"ROE: {roe}, Net margin: {nm}",
        "notes": ""
    })

    # 10) Balance sheet
    dte = metrics["dte"]
    cr = metrics["cr"]
    if dte is not None and cr is not None:
        if dte <= 1.0 and cr >= 1.5:
            s = 9
        elif dte <= 2.0 and cr >= 1.0:
            s = 7
        else:
            s = 4
    else:
        s = 6
    c.append({
        "id": 10,
        "name": "Balance sheet strength",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"D/E: {dte}, Current ratio: {cr}",
        "notes": ""
    })

    # 11) Valuation
    pe = metrics["pe"]
    if pe is not None:
        if pe <= 15:
            s = 9
        elif pe <= 25:
            s = 7
        else:
            s = 4
    else:
        s = 5
    c.append({
        "id": 11,
        "name": "Valuation (P/E)",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Trailing P/E: {pe}",
        "notes": "Very rough; does not adjust for growth/sector."
    })

    # 12) Technical trend
    t = metrics["trend"]
    if t is True:
        s = 9
    elif t is False:
        s = 4
    else:
        s = 5
    c.append({
        "id": 12,
        "name": "Technical uptrend",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": "Price > 50DMA > 150DMA" if t is not None else "Insufficient price history",
        "notes": ""
    })

    return c


# ====================================================
# LLM SCORING (QUALITATIVE 1,3,4,5,6)
# ====================================================
def llm_prompt(ticker: str, info: dict, metrics: dict) -> str:
    subset = {
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "summary": info.get("longBusinessSummary"),
        "marketCap": info.get("marketCap"),
        "profitMargins": info.get("profitMargins"),
        "roe": info.get("returnOnEquity"),
        "totalDebt": info.get("totalDebt"),
        "sharesOutstanding": info.get("sharesOutstanding"),
    }
    return (
        f"Ticker: {ticker}\n"
        f"Info: {json.dumps(subset, default=str)}\n"
        f"Metrics: {json.dumps(metrics, default=str)}"
    )


def llm_score(api_key: str, ticker: str, info: dict, metrics: dict):
    client = OpenAI(api_key=api_key)

    user_text = llm_prompt(ticker, info, metrics)

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = resp.choices[0].message.content
    # v1 can return string or list-of-parts; handle both
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = content

    data = json.loads(text)
    return data["criteria"]


# ====================================================
# STREAMLIT UI
# ====================================================
st.title("📊 Adam Khoo 12-Criteria — Hybrid LLM + Rule-Based")
st.caption("LLM scores qualitative (1,3,4,5,6); rules score quantitative (2,7,8,9,10,11,12).")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    run = st.button("Evaluate")

if run:
    api_key = CONFIG_KEY or st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OpenAI API key not found. Set it in config.py, Streamlit Secrets, or environment.")
        st.stop()

    with st.spinner(f"Fetching data for {ticker}..."):
        data = get_data(ticker)
        metrics = compute_metrics(data)

    info = data["info"]
    hist = data["hist"]
    long_name = info.get("longName") or info.get("shortName") or ticker
    price = info.get("currentPrice") or info.get("regularMarketPrice")

    st.subheader(f"📌 {long_name} ({ticker})")

    c1, c2 = st.columns([1.5, 2])

    with c1:
        st.metric("Price", f"{price:.2f}" if price is not None else "N/A")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        if metrics["cagr"] is not None:
            st.write(f"**Revenue CAGR:** {metrics['cagr']:.2%}")
        if metrics["roe"] is not None:
            st.write(f"**ROE:** {metrics['roe']:.2%}")
        if metrics["net_margin"] is not None:
            st.write(f"**Net margin:** {metrics['net_margin']:.2%}")

    with c2:
        st.write("### 📈 1-Year Price")
        if not hist.empty:
            st.line_chart(hist["Close"])
        else:
            st.info("No price history available.")

    with st.spinner("LLM evaluating qualitative criteria..."):
        qual_criteria = llm_score(api_key, ticker, info, metrics)

    quant_criteria = quantitative(metrics)

    # Merge 1–12 by ID
    all_crit = {c["id"]: c for c in qual_criteria + quant_criteria}
    ordered = [all_crit[i] for i in sorted(all_crit.keys())]

    scores = [c["score_0_10"] for c in ordered]
    overall = sum(scores) / len(scores)

    st.markdown("---")
    st.subheader("📊 Overall Score")

    if overall >= 8:
        color = "#8cff8c"
        label = "Strong candidate"
    elif overall >= 5:
        color = "#fff59d"
        label = "Mixed / borderline"
    else:
        color = "#ff9e80"
        label = "Weak / risky"

    st.metric("Score (0–10)", f"{overall:.2f}")

    st.markdown(
        f"""
        <div style="
            background:{color};
            padding:12px;
            border-radius:8px;
            color:black;
            text-align:center;
            font-weight:600;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("✅ Adam Khoo 12-Criteria Checklist")

    for c in ordered:
        cid = c["id"]
        name = c["name"]
        score = c["score_0_10"]
        passed = c["pass"]
        evidence = c.get("evidence", "")
        notes = c.get("notes", "")

        if passed:
            icon = "✅"
            bg = "#e8ffe8"
            border = "#0f9d58"
        elif score >= 5:
            icon = "⚠️"
            bg = "#fff7cc"
            border = "#ffcc00"
        else:
            icon = "❌"
            bg = "#ffe6e6"
            border = "#cc0000"

        st.markdown(
            f"""
            <div style="
                border-left:6px solid {border};
                background:{bg};
                padding:12px;
                margin-bottom:10px;
                border-radius:6px;
                color:black;
            ">
                <h4 style="margin:0;">
                    {icon} {cid}. {name}
                    <span style="float:right;font-weight:600;">Score: {score:.1f}</span>
                </h4>
                <p><b>Evidence:</b> {evidence}</p>
                {("<p><b>Notes:</b> " + notes + "</p>") if notes else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.info("Enter a ticker and click **Evaluate**. API key is loaded from config / secrets / env.")
