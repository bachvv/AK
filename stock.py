import os
import json
import math
from datetime import datetime
import requests

import yfinance as yf
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Finviz integration
try:
    from finvizfinance.quote import finvizfinance
    FINVIZ_AVAILABLE = True
except ImportError:
    FINVIZ_AVAILABLE = False
    # Note: We'll silently handle this - Finviz is optional

# ====================================================
# CONFIG / API KEY (Mode A: no UI box)
# ====================================================
try:
    from config import OPENAI_API_KEY as CONFIG_KEY
except ImportError:
    CONFIG_KEY = None

MODEL_NAME = "gpt-4.1-mini"  # or "gpt-4o-mini", "gpt-4.1", etc.


# ====================================================
# STREAMLIT PAGE CONFIG
# ====================================================
st.set_page_config(
    page_title="Synergy Trading Stock Analyzer",
    layout="wide",
)


# ====================================================
# LLM SYSTEM PROMPT (QUALITATIVE - IDs 1-5)
# ====================================================
LLM_SYSTEM_PROMPT = """You are an equity analyst scoring 5 QUALITATIVE criteria using Adam Khoo's methodology:

1) Competitive advantage/moat - Does the company have a durable advantage protecting it from competitors? 
   Examples: Brand power (Apple), network effects (Meta), high switching costs (Microsoft), cost advantage (Walmart), patents/IP, scale/distribution.
   Red flags: No moat commodity businesses, new competitors gaining share, falling margins, price wars.
   Score: Strong moat = 8-10, Moderate = 5-7, Weak/no moat = 0-4.

2) Large & growing market - Is the company in a market with room to grow?
   Positive signals: Industry CAGR >5-10%, emerging markets (EVs, AI, cloud, biotech), global expansion.
   Red flags: Disruption killing old industries, shrinking market, regulation killing demand.
   Score: High growth market = 8-10, Moderate = 5-7, Shrinking = 0-4.

3) Management quality - Are leaders competent, honest, shareholder-focused?
   Good signs: Transparent communication, consistent execution, long-term thinking, disciplined capital allocation, avoiding dilution/risky acquisitions.
   Red flags: Accounting irregularities, over-promising/under-delivering, scandals, excessive compensation, massive dilution, pump-style marketing.
   Score: Excellent = 8-10, Competent = 5-7, Poor = 0-4.

4) Insider ownership/alignment - Do key executives own meaningful shares?
   Why it matters: When insiders own shares, incentives align. They act responsibly, avoid reckless risks, focus on long-term value.
   Red flags: CEO holds tiny % of shares, insiders frequently selling, heavy option-based comp without performance requirements.
   Score: High ownership = 8-10, Moderate = 5-7, Low/minimal = 0-4.

5) Low dilution - Does the company avoid continually issuing new shares?
   Why it matters: Issuing shares = giving away pieces of business. Your ownership shrinks over time.
   Green flags: Stable/decreasing share count, occasional buybacks, limited equity compensation.
   Red flags: Frequent stock offerings, massive stock-based compensation, growth funded solely by dilution.
   Score: Low dilution = 8-10, Moderate = 5-7, High dilution = 0-4.

For each: id (1-5), name (criteria name), score_0_10 (0-10), pass (true if ≥7), evidence (brief), notes (optional).

Scoring: 0-3=weak, 4-6=mixed, 7-8=solid, 9-10=excellent.

Return JSON with "ticker" and "criteria" array. Each criterion must have: id, name, score_0_10, pass, evidence, notes (optional). Evidence: max 2 sentences. Notes: max 1 sentence."""


# ====================================================
# YFINANCE & FINVIZ HELPERS
# ====================================================
def get_finviz_data(ticker: str):
    """Fetch data from Finviz, including insider ownership and shares outstanding."""
    finviz_data = {}
    
    if not FINVIZ_AVAILABLE:
        return finviz_data
    
    try:
        stock = finvizfinance(ticker)
        # Get full financial data from Finviz
        full_data = stock.ticker_full_info()
        
        # The data is in the 'fundament' key (not directly in full_data)
        # full_data structure: {'fundament': {...}, 'ratings_outer': DataFrame, 'news': DataFrame, 'inside trader': DataFrame}
        fundament_data = full_data.get("fundament", {})
        
        # Extract key metrics we need from the fundament dictionary
        finviz_data = {
            "insider_ownership_pct": None,
            "institutional_ownership_pct": None,
            "shares_outstanding": None,
            "float_shares": None,
            "current_ratio_finviz": None,
            "quick_ratio": None,
            "debt_to_equity_finviz": None,
            "return_on_assets_finviz": None,
            "return_on_equity_finviz": None,
            "profit_margin_finviz": None,
            "operating_margin": None,
            "free_cash_flow_finviz": None,
            "quarterly_earnings_growth_finviz": None,
            "annual_earnings_growth_finviz": None,
            "eps_growth_this_year_finviz": None,
            "eps_growth_past_five_years_finviz": None,
            "eps_growth_qoq_finviz": None,
            "price_to_fcf_finviz": None,
            "dividend_yield_finviz": None,
            "sales_growth_qoq_finviz": None,
            "peg_finviz": None,
            "sales_finviz": None,  # Total Sales/Revenue from Finviz
            "market_cap_finviz": None,  # Market Cap from Finviz
        }
        
        # Parse insider ownership
        insider_own_str = fundament_data.get("Insider Own", "")
        if insider_own_str:
            try:
                finviz_data["insider_ownership_pct"] = float(str(insider_own_str).replace("%", "")) / 100
            except (ValueError, AttributeError):
                pass
        
        # Parse institutional ownership
        inst_own_str = fundament_data.get("Inst Own", "")
        if inst_own_str:
            try:
                finviz_data["institutional_ownership_pct"] = float(str(inst_own_str).replace("%", "")) / 100
            except (ValueError, AttributeError):
                pass
        
        # Parse shares outstanding
        shares_str = fundament_data.get("Shs Outstand", "")
        if shares_str:
            try:
                # Handle formats like "15.88B", "500M", "1.2K"
                shares_str = str(shares_str).upper().strip()
                multiplier = 1
                if shares_str.endswith("B"):
                    multiplier = 1e9
                    shares_str = shares_str[:-1]
                elif shares_str.endswith("M"):
                    multiplier = 1e6
                    shares_str = shares_str[:-1]
                elif shares_str.endswith("K"):
                    multiplier = 1e3
                    shares_str = shares_str[:-1]
                finviz_data["shares_outstanding"] = float(shares_str) * multiplier
            except (ValueError, AttributeError):
                pass
        
        # Parse float shares
        float_str = fundament_data.get("Shs Float", "")
        if float_str:
            try:
                float_str = str(float_str).upper().strip()
                multiplier = 1
                if float_str.endswith("B"):
                    multiplier = 1e9
                    float_str = float_str[:-1]
                elif float_str.endswith("M"):
                    multiplier = 1e6
                    float_str = float_str[:-1]
                elif float_str.endswith("K"):
                    multiplier = 1e3
                    float_str = float_str[:-1]
                finviz_data["float_shares"] = float(float_str) * multiplier
            except (ValueError, AttributeError):
                pass
        
        # Parse current ratio - try multiple field name variations
        current_ratio_str = fundament_data.get("Current Ratio", "")
        if current_ratio_str:
            try:
                current_ratio_str = str(current_ratio_str).replace(",", "").strip()
                if current_ratio_str and current_ratio_str != "-":
                    finviz_data["current_ratio_finviz"] = float(current_ratio_str)
            except (ValueError, AttributeError):
                pass
        
        # Parse quick ratio
        quick_ratio_str = fundament_data.get("Quick Ratio", "")
        if quick_ratio_str:
            try:
                quick_ratio_str = str(quick_ratio_str).replace(",", "").strip()
                if quick_ratio_str and quick_ratio_str != "-":
                    finviz_data["quick_ratio"] = float(quick_ratio_str)
            except (ValueError, AttributeError):
                pass
        
        # Parse debt to equity - try Debt/Eq first, then LT Debt/Eq
        debt_eq_str = fundament_data.get("Debt/Eq", "") or fundament_data.get("LT Debt/Eq", "")
        if debt_eq_str:
            try:
                debt_eq_str = str(debt_eq_str).replace(",", "").strip()
                if debt_eq_str and debt_eq_str != "-":
                    finviz_data["debt_to_equity_finviz"] = float(debt_eq_str)
            except (ValueError, AttributeError):
                pass
        
        # Parse ROA
        roa_str = fundament_data.get("ROA", "")
        if roa_str:
            try:
                roa_str = str(roa_str).replace("%", "").strip()
                if roa_str and roa_str != "-":
                    finviz_data["return_on_assets_finviz"] = float(roa_str) / 100
            except (ValueError, AttributeError):
                pass
        
        # Parse ROE
        roe_str = fundament_data.get("ROE", "")
        if roe_str:
            try:
                roe_str = str(roe_str).replace("%", "").strip()
                if roe_str and roe_str != "-":
                    finviz_data["return_on_equity_finviz"] = float(roe_str) / 100
            except (ValueError, AttributeError):
                pass
        
        # Parse profit margin
        pm_str = fundament_data.get("Profit Margin", "")
        if pm_str:
            try:
                pm_str = str(pm_str).replace("%", "").strip()
                if pm_str and pm_str != "-":
                    finviz_data["profit_margin_finviz"] = float(pm_str) / 100
            except (ValueError, AttributeError):
                pass
        
        # Parse operating margin
        om_str = fundament_data.get("Oper. Margin", "")
        if om_str:
            try:
                om_str = str(om_str).replace("%", "").strip()
                if om_str and om_str != "-":
                    finviz_data["operating_margin"] = float(om_str) / 100
            except (ValueError, AttributeError):
                pass
        
        # Parse quarterly earnings growth (EPS Q/Q)
        eps_qq_str = fundament_data.get("EPS Q/Q", "")
        
        if eps_qq_str:
            try:
                eps_qq_original = eps_qq_str
                eps_qq_str = eps_qq_str.replace("%", "").strip()
                # Handle negative values and special characters
                eps_qq_str = eps_qq_str.replace(",", "").replace(" ", "")
                if eps_qq_str and eps_qq_str != "-" and eps_qq_str.lower() != "none":
                    parsed_value = float(eps_qq_str) / 100
                    finviz_data["quarterly_earnings_growth_finviz"] = parsed_value
                    # Also set as EPS growth QoQ (same metric)
                    finviz_data["eps_growth_qoq_finviz"] = parsed_value
                    # Debug: Uncomment to see parsing success
                    # import sys
                    # print(f"DEBUG: Successfully parsed EPS Q/Q for {ticker}: {eps_qq_original} -> {parsed_value:.4f}", file=sys.stderr)
            except (ValueError, AttributeError) as e:
                # Debug: Uncomment to see parsing errors
                # import sys
                # print(f"DEBUG: Error parsing EPS Q/Q for {ticker}: '{eps_qq_str}', error: {e}", file=sys.stderr)
                pass
        # else:
        #     # Debug: Uncomment to see available fields
        #     # import sys
        #     # print(f"DEBUG: EPS Q/Q field not found for {ticker}. Available EPS/Q keys: {[k for k in full_data.keys() if 'EPS' in str(k).upper() or 'Q' in str(k).upper()]}", file=sys.stderr)
        
        # Parse EPS growth this year (EPS this Y) - this is the current year growth
        eps_this_y_str = fundament_data.get("EPS this Y", "")
        if eps_this_y_str:
            try:
                eps_this_y_str = str(eps_this_y_str).replace("%", "").strip()
                if eps_this_y_str and eps_this_y_str != "-":
                    finviz_data["eps_growth_this_year_finviz"] = float(eps_this_y_str) / 100
            except (ValueError, AttributeError):
                pass
        
        # Parse EPS next year (EPS next Y) - this is forward-looking
        eps_next_y_str = fundament_data.get("EPS next Y", "")
        if eps_next_y_str:
            try:
                eps_next_y_str = str(eps_next_y_str).replace("%", "").strip()
                if eps_next_y_str and eps_next_y_str != "-":
                    # Store as next year growth (different from this year)
                    finviz_data["eps_growth_next_year_finviz"] = float(eps_next_y_str) / 100
                    # Also use as annual earnings growth if we don't have it yet
                    if finviz_data["annual_earnings_growth_finviz"] is None:
                        finviz_data["annual_earnings_growth_finviz"] = float(eps_next_y_str) / 100
            except (ValueError, AttributeError):
                pass
        
        # Parse EPS growth past 5 years (EPS past 5Y or EPS past 3/5Y)
        # Try "EPS past 5Y" first
        eps_5y_str = fundament_data.get("EPS past 5Y", "")
        if eps_5y_str:
            try:
                eps_5y_str = str(eps_5y_str).replace("%", "").strip()
                if eps_5y_str and eps_5y_str != "-":
                    finviz_data["eps_growth_past_five_years_finviz"] = float(eps_5y_str) / 100
            except (ValueError, AttributeError):
                pass
        # Also try "EPS past 3/5Y" format (sometimes shown as "past 3/5Y")
        elif "EPS past 3/5Y" in fundament_data:
            eps_5y_str = fundament_data.get("EPS past 3/5Y", "")
            try:
                # Format might be like "-17.13% -" (past 3Y - past 5Y)
                eps_5y_str = str(eps_5y_str).split("-")[0].replace("%", "").strip()
                if eps_5y_str and eps_5y_str != "-":
                    finviz_data["eps_growth_past_five_years_finviz"] = float(eps_5y_str) / 100
            except (ValueError, AttributeError):
                pass
        
        # Parse annual earnings growth (earningsGrowth from yahoo or "EPS next 5Y")
        # Try "EPS next 5Y" first, then "EPS next Y" as annual growth
        eps_next_5y_str = fundament_data.get("EPS next 5Y", "")
        if eps_next_5y_str:
            try:
                eps_next_5y_str = str(eps_next_5y_str).replace("%", "").strip()
                if eps_next_5y_str:
                    # Convert 5-year growth to annual
                    annual_growth = ((1 + float(eps_next_5y_str) / 100) ** (1/5)) - 1
                    finviz_data["annual_earnings_growth_finviz"] = annual_growth
            except (ValueError, AttributeError):
                pass
        
        # Parse Price/FCF
        price_fcf_str = fundament_data.get("P/FCF", "")
        if price_fcf_str:
            try:
                price_fcf_str = str(price_fcf_str).strip().replace(",", "")
                if price_fcf_str and price_fcf_str != "-" and price_fcf_str.lower() != "none":
                    finviz_data["price_to_fcf_finviz"] = float(price_fcf_str)
            except (ValueError, AttributeError):
                pass
        
        # Parse PEG (Price/Earnings to Growth ratio)
        peg_str = fundament_data.get("PEG", "")
        if peg_str:
            try:
                peg_str = str(peg_str).strip()
                if peg_str and peg_str != "-":
                    finviz_data["peg_finviz"] = float(peg_str)
            except (ValueError, AttributeError):
                pass
        
        # Parse Dividend Yield
        div_str = fundament_data.get("Dividend %", "") or fundament_data.get("Dividend TTM", "") or fundament_data.get("Dividend", "")
        if div_str:
            try:
                div_str = str(div_str).replace("%", "").strip()
                if div_str and div_str != "-" and div_str.lower() != "none":
                    finviz_data["dividend_yield_finviz"] = float(div_str) / 100
            except (ValueError, AttributeError):
                pass
        
        # Parse Sales/Revenue (total sales)
        sales_str = fundament_data.get("Sales", "") or fundament_data.get("Revenue", "")
        if sales_str:
            try:
                # Handle formats like "605.55M", "1.2B", "500K"
                sales_str = str(sales_str).upper().strip().replace(",", "")
                if sales_str and sales_str != "-":
                    multiplier = 1
                    if sales_str.endswith("B"):
                        multiplier = 1e9
                        sales_str = sales_str[:-1]
                    elif sales_str.endswith("M"):
                        multiplier = 1e6
                        sales_str = sales_str[:-1]
                    elif sales_str.endswith("K"):
                        multiplier = 1e3
                        sales_str = sales_str[:-1]
                    finviz_data["sales_finviz"] = float(sales_str) * multiplier
            except (ValueError, AttributeError):
                pass
        
        # Parse Market Cap
        market_cap_str = fundament_data.get("Market Cap", "")
        if market_cap_str:
            try:
                # Handle formats like "24.38B", "500M", "1.2K"
                market_cap_str = str(market_cap_str).upper().strip().replace(",", "")
                if market_cap_str and market_cap_str != "-":
                    multiplier = 1
                    if market_cap_str.endswith("B"):
                        multiplier = 1e9
                        market_cap_str = market_cap_str[:-1]
                    elif market_cap_str.endswith("M"):
                        multiplier = 1e6
                        market_cap_str = market_cap_str[:-1]
                    elif market_cap_str.endswith("K"):
                        multiplier = 1e3
                        market_cap_str = market_cap_str[:-1]
                    finviz_data["market_cap_finviz"] = float(market_cap_str) * multiplier
            except (ValueError, AttributeError):
                pass
        
        # Parse Sales Q/Q (quarterly sales growth)
        sales_qq_str = fundament_data.get("Sales Q/Q", "")
        if sales_qq_str:
            try:
                sales_qq_str = str(sales_qq_str).replace("%", "").strip()
                if sales_qq_str and sales_qq_str != "-":
                    finviz_data["sales_growth_qoq_finviz"] = float(sales_qq_str) / 100
            except (ValueError, AttributeError):
                pass
        
    except Exception as e:
        # Silently fail - Finviz data is supplementary
        pass
    
    return finviz_data


def get_data(ticker: str):
    tk = yf.Ticker(ticker)
    info = getattr(tk, "info", {}) or {}
    fin = tk.financials if hasattr(tk, "financials") else pd.DataFrame()
    bs = tk.balance_sheet if hasattr(tk, "balance_sheet") else pd.DataFrame()
    cf = tk.cashflow if hasattr(tk, "cashflow") else pd.DataFrame()
    hist = tk.history(period="2y")  # 2 years for 52-week calculations
    
    # Get Finviz data
    finviz_data = get_finviz_data(ticker)
    
    return {"info": info, "fin": fin, "bs": bs, "cf": cf, "hist": hist, "finviz": finviz_data}


def fetch_news_with_openai(ticker: str, company_name: str, api_key: str, limit: int = 5):
    """
    Fetch recent news about a stock using web search and OpenAI to format results.
    Returns a list of news items in the same format as yfinance news.
    """
    if not api_key:
        return []
    
    try:
        # Try using Alpha Vantage News API (free tier available)
        # Or use a simple web search approach
        # For now, let's use a combination of web scraping and OpenAI formatting
        
        # Use OpenAI to help generate news search queries and format results
        client = OpenAI(api_key=api_key)
        
        # First, try to get news from a news API or web search
        # Using a simple approach: search Google News RSS or similar
        search_query = f"{company_name} {ticker} stock news"
        
        # Try fetching from Google News RSS (free, no API key needed)
        try:
            from urllib.parse import quote_plus
            encoded_query = quote_plus(search_query)
            google_news_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(google_news_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                from xml.etree import ElementTree as ET
                root = ET.fromstring(response.content)
                # Get more items to filter later
                items = root.findall('.//item')[:limit * 2]
                
                news_items = []
                for item in items:
                    try:
                        # Google News RSS format
                        title_elem = item.find('title')
                        link_elem = item.find('link')
                        pub_date_elem = item.find('pubDate')
                        source_elem = item.find('source')
                        description_elem = item.find('description')
                        
                        # Get title - Google News often has HTML entities, strip them
                        if title_elem is not None and title_elem.text:
                            title = title_elem.text.strip()
                            # Remove common prefixes like " - " that Google News adds
                            if title.startswith('- '):
                                title = title[2:].strip()
                        else:
                            continue  # Skip items without titles
                        
                        # Get link
                        link = link_elem.text.strip() if (link_elem is not None and link_elem.text) else ""
                        
                        # Get publisher/source
                        if source_elem is not None and source_elem.text:
                            source = source_elem.text.strip()
                        else:
                            # Try to extract from title (format: "Title - Publisher")
                            if ' - ' in title:
                                parts = title.split(' - ', 1)
                                if len(parts) == 2:
                                    title = parts[0].strip()
                                    source = parts[1].strip()
                                else:
                                    source = "Google News"
                            else:
                                source = "Google News"
                        
                        # Get description/summary
                        if description_elem is not None and description_elem.text:
                            description = description_elem.text.strip()
                            # Remove HTML tags if present
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(description, 'html.parser')
                            description = soup.get_text()[:500].strip()
                        else:
                            description = ""
                        
                        # Parse date
                        pub_date = pub_date_elem.text.strip() if (pub_date_elem is not None and pub_date_elem.text) else ""
                        try:
                            if pub_date:
                                from dateutil import parser
                                dt = parser.parse(pub_date)
                                timestamp = int(dt.timestamp())
                            else:
                                timestamp = int(datetime.now().timestamp())
                        except:
                            timestamp = int(datetime.now().timestamp())
                        
                        # Only add if we have at least a title
                        if title and title != "No title":
                            news_items.append({
                                "title": title,
                                "publisher": source,
                                "link": link,
                                "summary": description,
                                "providerPublishTime": timestamp
                            })
                    except Exception as e:
                        # Skip items that fail to parse
                        continue
                
                if news_items:
                    return news_items[:limit]
        except Exception:
            pass
        
        # Fallback: Use OpenAI to generate news summaries based on knowledge
        # (Note: This won't be real-time news, but can provide context)
        prompt = f"""Based on your knowledge, provide 3-5 recent or relevant news items about {company_name} (ticker: {ticker}).
        Return a JSON object with a "news" array. Each item should have:
        - title: headline
        - publisher: news source
        - link: placeholder URL or empty string
        - summary: 1-2 sentence summary
        - providerPublishTime: current Unix timestamp
        
        Focus on financial, business, or market-relevant information."""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a financial news aggregator. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=800,
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        
        # Extract news items from response
        news_items = []
        if isinstance(data, dict) and "news" in data:
            news_items = data["news"]
        elif isinstance(data, list):
            news_items = data
        
        # Format to match yfinance news structure
        formatted_news = []
        for item in news_items[:limit]:
            formatted_news.append({
                "title": item.get("title", "No title"),
                "publisher": item.get("publisher", "AI Generated"),
                "link": item.get("link", ""),
                "summary": item.get("summary", ""),
                "providerPublishTime": item.get("providerPublishTime", int(datetime.now().timestamp()))
            })
        
        return formatted_news
        
    except Exception as e:
        # Silently fail and return empty list
        return []


def create_stock_chart_with_news(df, news_items, ticker):
    """
    Create an interactive Plotly chart with stock price and news markers.
    
    Args:
        df: DataFrame with stock price data (OHLCV)
        news_items: List of news items with timestamps
        ticker: Stock ticker symbol
    """
    if df.empty:
        return None
    
    # Remove empty candles (no trading data)
    # Filter out rows where volume is 0 or OHLC values are invalid
    df_filtered = df.copy()
    if 'Volume' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Volume'] > 0]
    
    # Remove rows with invalid OHLC data
    df_filtered = df_filtered.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df_filtered = df_filtered[
        (df_filtered['Open'] > 0) & 
        (df_filtered['High'] > 0) & 
        (df_filtered['Low'] > 0) & 
        (df_filtered['Close'] > 0)
    ]
    
    if df_filtered.empty:
        return None
    
    # Use filtered dataframe
    df = df_filtered
    
    # Check if we have volume data to determine subplot structure
    has_volume = 'Volume' in df.columns and len(df) > 0
    
    if has_volume:
        # Create subplots: price chart on top, volume on bottom
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{ticker} Price (1-Year)', 'Volume')
        )
    else:
        # Single plot if no volume
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=(f'{ticker} Price (1-Year)',)
        )
    
    # Add candlestick chart
    try:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
    except Exception:
        # Fallback to line chart if candlestick fails
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    
    # Add news markers
    news_x = []
    news_y = []
    news_titles = []
    news_times = []
    news_links = []
    
    if news_items and isinstance(news_items, list) and len(news_items) > 0:
        for news in news_items:
            if not isinstance(news, dict):
                continue
                
            # Get publication time from news item - try multiple possible keys
            pub_time = (news.get('providerPublishTime') or 
                       news.get('publishTime') or 
                       news.get('timestamp') or
                       0)
            
            # If pub_time is a datetime object, convert to timestamp
            if isinstance(pub_time, datetime):
                pub_time = pub_time.timestamp()
            elif isinstance(pub_time, pd.Timestamp):
                pub_time = pub_time.timestamp()
            
            if not pub_time or pub_time == 0:
                continue
            
            try:
                news_time = pd.Timestamp(datetime.fromtimestamp(pub_time))
                
                # Handle timezone issues
                if news_time.tz is None:
                    news_time = news_time.tz_localize('UTC')
                
                # Convert to same timezone as dataframe index (if any)
                if df.index.tz is not None:
                    news_time = news_time.tz_convert(df.index.tz)
                elif news_time.tz is not None:
                    news_time = news_time.tz_localize(None)
                
                # Find the closest price point to the news time
                try:
                    # Get indexer to find closest timestamp
                    idx_pos = df.index.get_indexer([news_time], method='nearest')[0]
                    
                    if idx_pos >= 0 and idx_pos < len(df):
                        closest_idx = df.index[idx_pos]
                        closest_price = df.loc[closest_idx, 'High'] * 1.002  # Place marker slightly above the high
                        
                        news_x.append(closest_idx)
                        news_y.append(closest_price)
                        news_titles.append(news.get('title', 'No title'))
                        news_times.append(news_time.strftime('%Y-%m-%d %H:%M'))
                        news_links.append(news.get('link', ''))
                except Exception:
                    continue
            except Exception:
                continue
    
    # Add news markers as scatter points
    if news_x and len(news_x) > 0:
        # Truncate titles for display
        display_titles = [f"{time}<br>{title[:50] + '...' if len(title) > 50 else title}" 
                         for time, title in zip(news_times, news_titles)]
        
        fig.add_trace(
            go.Scatter(
                x=news_x,
                y=news_y,
                mode='markers+text',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                text=[title[:30] + '...' if len(title) > 30 else title for title in news_titles],
                textposition='top center',
                textfont=dict(size=9, color='red'),
                name='News',
                hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Price: $%{y:.2f}<extra></extra>',
                customdata=list(zip(news_times, news_titles, news_links))
            ),
            row=1, col=1
        )
        
        # Add vertical lines for news events
        for x_val in news_x:
            fig.add_vline(
                x=x_val,
                line_dash="dash",
                line_color="red",
                opacity=0.3,
                row=1, col=1
            )
    
    # Add volume bars - always show volume if available
    if has_volume:
        try:
            colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                     for i in range(len(df))]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=2, col=1
            )
        except Exception:
            # If volume coloring fails, just show volume without colors
            try:
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        name='Volume',
                        opacity=0.6
                    ),
                    row=2, col=1
                )
            except Exception:
                pass
    
    # Update layout
    layout_updates = {
        'title': f'{ticker} Price Chart with News Markers',
        'xaxis_title': 'Date',
        'yaxis_title': 'Price (USD)',
        'height': 600 if has_volume else 400,
        'hovermode': 'x unified',
        'template': 'plotly_white',
        'xaxis_rangeslider_visible': False,
        'showlegend': True
    }
    
    if has_volume:
        layout_updates['yaxis2_title'] = 'Volume'
    
    fig.update_layout(**layout_updates)
    
    return fig

def safe_cagr(values):
    """
    Calculates CAGR safely:
    - Accepts a list or 1D array-like of numbers
    - Removes NaN, zero, and negative values
    - Assumes values are ordered oldest->newest
    - Requires at least 2 valid points
    """
    if values is None:
        return None

    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    s = s[s > 0]  # remove zero & negative

    if len(s) < 2:
        return None

    start = s.iloc[0]
    end = s.iloc[-1]
    years = len(s) - 1
    try:
        return (end / start) ** (1 / years) - 1
    except Exception:
        return None


def compute_metrics(data: dict, ticker: str = None):
    info = data["info"]
    fin = data["fin"]
    bs = data["bs"]
    cf = data["cf"]
    hist = data["hist"]
    finviz = data.get("finviz", {})
    
    # Create ticker object if needed for quarterly data
    tk = yf.Ticker(ticker) if ticker else None

    metrics = {}
    
    # ---- Insider/Institutional ownership (Finviz preferred, more accurate) ----
    metrics["insider_ownership_pct"] = finviz.get("insider_ownership_pct") or info.get("heldPercentInsiders")
    metrics["institutional_ownership_pct"] = finviz.get("institutional_ownership_pct") or info.get("heldPercentInstitutions")

    # ---- Revenue CAGR (safe) ----
    rev_row = None
    if not fin.empty:
        for key in ["Total Revenue", "TotalRevenue", "totalRevenue"]:
            if key in fin.index:
                rev_row = fin.loc[key]
                break

    if isinstance(rev_row, pd.Series):
        # yfinance financials columns are usually newest -> oldest
        # reverse so it's oldest -> newest for CAGR
        rev_values = list(rev_row.fillna(0).astype(float).values[::-1])
    else:
        rev_values = None

    metrics["cagr"] = safe_cagr(rev_values)
    metrics["revenue_growth"] = metrics["cagr"]  # Alias

    # ---- Basic profitability & valuation ----
    metrics["roe"] = info.get("returnOnEquity")
    # Override with Finviz if available
    if finviz.get("return_on_equity_finviz") is not None:
        metrics["roe"] = finviz.get("return_on_equity_finviz")
    metrics["return_on_equity"] = metrics["roe"]  # Alias
    
    metrics["net_margin"] = info.get("profitMargins")
    # Override with Finviz if available
    if finviz.get("profit_margin_finviz") is not None:
        metrics["net_margin"] = finviz.get("profit_margin_finviz")
    metrics["profit_margin"] = metrics["net_margin"]  # Alias
    
    metrics["pe"] = info.get("trailingPE")
    metrics["trailing_pe"] = info.get("trailingPE")  # Alias
    
    # PEG (Price/Earnings to Growth) - prefer Finviz, fallback to calculation
    metrics["peg"] = finviz.get("peg_finviz")
    if metrics["peg"] is None and metrics["pe"] is not None:
        # Calculate PEG = P/E / Growth Rate (annual earnings growth)
        growth = metrics.get("annual_earnings_growth") or metrics.get("earnings_growth_10y")
        if growth is not None and growth > 0:
            metrics["peg"] = metrics["pe"] / (growth * 100)  # Convert growth to percentage
    # Also check yfinance directly
    if metrics["peg"] is None:
        metrics["peg"] = info.get("pegRatio")
    
    # Return on Assets
    metrics["return_on_assets"] = info.get("returnOnAssets")
    # Override with Finviz if available
    if finviz.get("return_on_assets_finviz") is not None:
        metrics["return_on_assets"] = finviz.get("return_on_assets_finviz")
    
    # Operating margin
    metrics["operating_margin"] = finviz.get("operating_margin") or info.get("operatingMargins")

    # ---- Debt to equity ----
    # Prefer Finviz data (more reliable)
    metrics["dte"] = finviz.get("debt_to_equity_finviz")
    if metrics["dte"] is None:
        # Fallback to yfinance calculation
        debt = info.get("totalDebt")
        equity = info.get("totalStockholderEquity")
        metrics["dte"] = debt / equity if equity not in (None, 0) and debt is not None else None
    metrics["debt_to_equity"] = metrics["dte"]  # Alias

    # ---- Current ratio ----
    # Prefer Finviz data (more reliable)
    cr = finviz.get("current_ratio_finviz")
    if cr is None:
        # Fallback to yfinance calculation
        try:
            col = bs.columns[0]
            ca = bs.loc["Total Current Assets"][col]
            cl = bs.loc["Total Current Liabilities"][col]
            cr = ca / cl if cl != 0 else None
        except Exception:
            cr = None
    metrics["cr"] = cr
    metrics["current_ratio"] = cr  # Alias
    
    # ---- Quick ratio ----
    # Use Finviz data (not available in yfinance)
    metrics["quick_ratio"] = finviz.get("quick_ratio")

    # ---- FCF margin and Price/FCF ----
    # FCF Margin = Free Cash Flow / Revenue (Sales)
    # Free Cash Flow (Absolute) = (FCF per share) × Shares Outstanding
    # Where FCF per share = Price / P/FCF
    fcf_margin = None
    fcf_absolute = None
    price_to_fcf_finviz = finviz.get("price_to_fcf_finviz")
    sales_finviz = finviz.get("sales_finviz")
    market_cap_finviz = finviz.get("market_cap_finviz")
    
    # Method 1 (Preferred): Calculate FCF per share from Price / P/FCF, then multiply by shares
    # Total FCF = (FCF per share) × Shares Outstanding
    if price_to_fcf_finviz is not None and price_to_fcf_finviz > 0:
        current_price = None
        if not hist.empty:
            current_price = hist["Close"].iloc[-1]
        if current_price is None:
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        
        if current_price is not None and current_price > 0:
            # FCF per share = Price / P/FCF
            fcf_per_share = current_price / price_to_fcf_finviz
            # Get shares outstanding (prefer Finviz)
            shares_outstanding = finviz.get("shares_outstanding") or info.get("sharesOutstanding")
            if shares_outstanding is not None and shares_outstanding > 0:
                # Total FCF = FCF per share × Shares Outstanding
                fcf_absolute = fcf_per_share * shares_outstanding
                
                # Calculate FCF Margin = FCF / Revenue (Sales from Finviz preferred)
                if sales_finviz is not None and sales_finviz > 0:
                    fcf_margin = fcf_absolute / sales_finviz
                elif isinstance(rev_row, pd.Series):
                    latest_rev = float(rev_row.iloc[0])
                    if latest_rev != 0:
                        fcf_margin = fcf_absolute / latest_rev
    
    # Method 2 (Fallback): Use Finviz P/FCF and Market Cap to calculate Total FCF
    if fcf_absolute is None and price_to_fcf_finviz is not None and price_to_fcf_finviz > 0:
        if market_cap_finviz is not None and market_cap_finviz > 0:
            # Total FCF = Market Cap / P/FCF
            fcf_absolute = market_cap_finviz / price_to_fcf_finviz
            
            # Calculate FCF Margin = FCF / Revenue (Sales from Finviz)
            if fcf_margin is None:
                if sales_finviz is not None and sales_finviz > 0:
                    fcf_margin = fcf_absolute / sales_finviz
    
    # Method 3: Fallback to calculating from cashflow statement if Finviz not available
    if fcf_absolute is None or fcf_margin is None:
        try:
            col_cf = cf.columns[0]
            op = cf.loc["Total Cash From Operating Activities"][col_cf]
            cap = cf.loc["Capital Expenditures"][col_cf]
            fcf = op + cap
            if fcf_absolute is None:
                fcf_absolute = fcf
            
            # Calculate FCF Margin
            if fcf_margin is None:
                if sales_finviz is not None and sales_finviz > 0:
                    fcf_margin = fcf_absolute / sales_finviz
                elif isinstance(rev_row, pd.Series):
                    latest_rev = float(rev_row.iloc[0])
                    if latest_rev != 0:
                        fcf_margin = fcf_absolute / latest_rev
        except Exception:
            pass
    
    metrics["fcf_margin"] = fcf_margin
    metrics["free_cash_flow"] = fcf_absolute  # Absolute value
    metrics["price_to_fcf"] = price_to_fcf_finviz  # Store Price/FCF from Finviz if available
    metrics["sales_finviz"] = sales_finviz  # Store Sales from Finviz if available
    metrics["market_cap_finviz"] = market_cap_finviz  # Store Market Cap from Finviz if available

    # ---- Trend (price > 50DMA > 150DMA) ----
    trend = None
    try:
        if not hist.empty:
            close = hist["Close"]
            ma50 = close.rolling(50).mean().iloc[-1]
            ma150 = close.rolling(150).mean().iloc[-1]
            if not math.isnan(ma50) and not math.isnan(ma150):
                trend = close.iloc[-1] > ma50 > ma150
    except Exception:
        pass
    metrics["trend"] = trend

    # ---- Additional metrics ----
    
    # Price to Book
    metrics["price_to_book"] = info.get("priceToBook")
    
    # Dividend Yield - prefer Finviz, fallback to yfinance
    metrics["dividend_yield"] = finviz.get("dividend_yield_finviz") or info.get("dividendYield")
    
    # Return on Assets
    metrics["return_on_assets"] = info.get("returnOnAssets")
    
    # Total Liabilities
    try:
        col = bs.columns[0]
        metrics["total_liabilities"] = bs.loc["Total Liab"][col] if "Total Liab" in bs.index else None
    except Exception:
        metrics["total_liabilities"] = None
    
    # Shares Outstanding - prefer Finviz, fallback to yfinance
    metrics["shares_outstanding"] = finviz.get("shares_outstanding") or info.get("sharesOutstanding")
    metrics["float_shares"] = finviz.get("float_shares") or info.get("floatShares")
    
    # Dilution calculation - check if shares outstanding increased over time
    dilution_rate = None
    shares_history = []
    try:
        if not bs.empty and len(bs.columns) >= 2:
            # Look for shares outstanding in balance sheet (may vary by company)
            share_keys = ["Share Issued", "Common Stock Shares Outstanding", "Shares Outstanding"]
            for key in share_keys:
                if key in bs.index:
                    shares_history = list(bs.loc[key].fillna(0).astype(float).values)
                    break
        
        # If we have historical shares data, calculate dilution
        if len(shares_history) >= 2:
            # Reverse to get oldest -> newest
            shares_history = shares_history[::-1]
            oldest = shares_history[0]
            newest = shares_history[-1]
            if oldest > 0:
                # Calculate annualized dilution rate
                years = len(shares_history) - 1
                if years > 0:
                    dilution_rate = ((newest / oldest) ** (1 / years)) - 1
    except Exception:
        pass
    
    metrics["dilution_rate"] = dilution_rate  # Positive = dilution, negative = buyback
    
    # EPS TTM
    metrics["eps_ttm"] = info.get("trailingEps")
    
    # Book Value Per Share
    metrics["book_value_per_share"] = info.get("bookValue")
    
    # Price history metrics
    if not hist.empty:
        close = hist["Close"]
        volume = hist["Volume"] if "Volume" in hist.columns else None
        current_price = close.iloc[-1]
        
        # 52-week high/low
        hist_1y = hist.tail(252) if len(hist) >= 252 else hist  # ~252 trading days = 1 year
        metrics["fifty_two_week_high"] = hist_1y["High"].max() if "High" in hist_1y.columns else None
        metrics["fifty_two_week_low"] = hist_1y["Low"].min() if "Low" in hist_1y.columns else None
        
        # Distance from 52w high
        if metrics["fifty_two_week_high"] is not None and metrics["fifty_two_week_high"] != 0:
            metrics["distance_from_52w_high"] = (current_price - metrics["fifty_two_week_high"]) / metrics["fifty_two_week_high"]
        else:
            metrics["distance_from_52w_high"] = None
        
        # SMA 50
        if len(close) >= 50:
            metrics["sma_50"] = close.rolling(50).mean().iloc[-1]
        else:
            metrics["sma_50"] = None
        
        # SMA 200
        if len(close) >= 200:
            metrics["sma_200"] = close.rolling(200).mean().iloc[-1]
        else:
            metrics["sma_200"] = None
        
        # 6-month return
        if len(hist) >= 126:  # ~126 trading days = 6 months
            price_6m_ago = close.iloc[-126]
            metrics["six_month_return"] = (current_price - price_6m_ago) / price_6m_ago if price_6m_ago != 0 else None
        else:
            metrics["six_month_return"] = None
        
        # Relative Strength (price vs market/S&P proxy - simplified: price vs SMA)
        if metrics["sma_200"] is not None and metrics["sma_200"] != 0:
            metrics["relative_strength"] = (current_price - metrics["sma_200"]) / metrics["sma_200"]
        else:
            metrics["relative_strength"] = None
        
        # Average volume (10 and 50 days)
        if volume is not None:
            if len(volume) >= 10:
                metrics["avg_volume_10"] = volume.tail(10).mean()
            else:
                metrics["avg_volume_10"] = None
            if len(volume) >= 50:
                metrics["avg_volume_50"] = volume.tail(50).mean()
            else:
                metrics["avg_volume_50"] = None
        else:
            metrics["avg_volume_10"] = None
            metrics["avg_volume_50"] = None
        
        # === ADDITIONAL TECHNICAL INDICATORS ===
        
        # RSI (Relative Strength Index) - 14 period
        if len(close) >= 14:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            metrics["rsi"] = rsi.iloc[-1] if not math.isnan(rsi.iloc[-1]) else None
        else:
            metrics["rsi"] = None
        
        # MACD (Moving Average Convergence Divergence)
        if len(close) >= 26:
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            metrics["macd"] = macd_line.iloc[-1] if not math.isnan(macd_line.iloc[-1]) else None
            metrics["macd_signal"] = signal_line.iloc[-1] if not math.isnan(signal_line.iloc[-1]) else None
            metrics["macd_histogram"] = histogram.iloc[-1] if not math.isnan(histogram.iloc[-1]) else None
        else:
            metrics["macd"] = None
            metrics["macd_signal"] = None
            metrics["macd_histogram"] = None
        
        # Fibonacci Retracement Levels (based on recent swing high/low)
        if len(hist) >= 60:
            recent_high = hist["High"].tail(60).max()
            recent_low = hist["Low"].tail(60).min()
            fib_range = recent_high - recent_low
            if fib_range > 0:
                metrics["fib_236"] = recent_low + (fib_range * 0.236)
                metrics["fib_382"] = recent_low + (fib_range * 0.382)
                metrics["fib_500"] = recent_low + (fib_range * 0.500)
                metrics["fib_618"] = recent_low + (fib_range * 0.618)
                metrics["fib_786"] = recent_low + (fib_range * 0.786)
                # Calculate distance from current price to nearest Fibonacci level
                fib_levels = [metrics["fib_236"], metrics["fib_382"], metrics["fib_500"], 
                             metrics["fib_618"], metrics["fib_786"]]
                distances = [abs(current_price - fib) / current_price for fib in fib_levels]
                metrics["distance_to_nearest_fib"] = min(distances) if distances else None
            else:
                metrics["fib_236"] = None
                metrics["fib_382"] = None
                metrics["fib_500"] = None
                metrics["fib_618"] = None
                metrics["fib_786"] = None
                metrics["distance_to_nearest_fib"] = None
        else:
            metrics["fib_236"] = None
            metrics["fib_382"] = None
            metrics["fib_500"] = None
            metrics["fib_618"] = None
            metrics["fib_786"] = None
            metrics["distance_to_nearest_fib"] = None
        
        # Support and Resistance Levels
        if len(hist) >= 20:
            # Support: recent lows (minimum of rolling windows)
            support_levels = []
            for window in [10, 20, 50]:
                if len(hist) >= window:
                    support_levels.append(hist["Low"].tail(window).min())
            if support_levels:
                metrics["support_level"] = max(support_levels)  # Strongest support (highest of lows)
            
            # Resistance: recent highs (maximum of rolling windows)
            resistance_levels = []
            for window in [10, 20, 50]:
                if len(hist) >= window:
                    resistance_levels.append(hist["High"].tail(window).max())
            if resistance_levels:
                metrics["resistance_level"] = min(resistance_levels)  # Strongest resistance (lowest of highs)
            
            # Distance to support/resistance
            if metrics.get("support_level") is not None:
                metrics["distance_to_support"] = (current_price - metrics["support_level"]) / current_price
            else:
                metrics["distance_to_support"] = None
                
            if metrics.get("resistance_level") is not None:
                metrics["distance_to_resistance"] = (metrics["resistance_level"] - current_price) / current_price
            else:
                metrics["distance_to_resistance"] = None
        else:
            metrics["support_level"] = None
            metrics["resistance_level"] = None
            metrics["distance_to_support"] = None
            metrics["distance_to_resistance"] = None
        
        # Volume Analysis
        if volume is not None and len(volume) >= 20:
            # Volume trend (increasing vs decreasing)
            recent_vol = volume.tail(10).mean()
            older_vol = volume.tail(20).head(10).mean()
            if older_vol > 0:
                metrics["volume_trend"] = (recent_vol - older_vol) / older_vol
            else:
                metrics["volume_trend"] = None
            
            # Volume vs Price (confirmation)
            price_change = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] if len(close) >= 10 else None
            if price_change is not None and metrics["volume_trend"] is not None:
                # Positive volume trend with price increase = bullish
                # Negative volume trend with price decrease = bearish
                if price_change > 0 and metrics["volume_trend"] > 0:
                    metrics["volume_price_confirmation"] = "bullish"
                elif price_change < 0 and metrics["volume_trend"] < 0:
                    metrics["volume_price_confirmation"] = "bearish"
                else:
                    metrics["volume_price_confirmation"] = "divergence"
            else:
                metrics["volume_price_confirmation"] = None
            
            # Volume relative to average
            if metrics["avg_volume_50"] is not None and metrics["avg_volume_50"] > 0:
                current_vol = volume.iloc[-1]
                metrics["volume_relative"] = current_vol / metrics["avg_volume_50"]
            else:
                metrics["volume_relative"] = None
        else:
            metrics["volume_trend"] = None
            metrics["volume_price_confirmation"] = None
            metrics["volume_relative"] = None
    else:
        # Initialize all price history and technical indicators as None if no history
        metrics["fifty_two_week_high"] = None
        metrics["fifty_two_week_low"] = None
        metrics["distance_from_52w_high"] = None
        metrics["sma_200"] = None
        metrics["six_month_return"] = None
        metrics["relative_strength"] = None
        metrics["sma_50"] = None
        metrics["sma_200"] = None
        metrics["avg_volume_10"] = None
        metrics["avg_volume_50"] = None
        metrics["rsi"] = None
        metrics["macd"] = None
        metrics["macd_signal"] = None
        metrics["macd_histogram"] = None
        metrics["fib_236"] = None
        metrics["fib_382"] = None
        metrics["fib_500"] = None
        metrics["fib_618"] = None
        metrics["fib_786"] = None
        metrics["distance_to_nearest_fib"] = None
        metrics["support_level"] = None
        metrics["resistance_level"] = None
        metrics["distance_to_support"] = None
        metrics["distance_to_resistance"] = None
        metrics["volume_trend"] = None
        metrics["volume_price_confirmation"] = None
        metrics["volume_relative"] = None
    
    # Earnings growth metrics - prefer Finviz, fallback to yfinance
    metrics["earnings_growth_10y"] = info.get("earningsGrowth")
    # EPS growth this year - prefer Finviz
    metrics["eps_growth_this_year"] = finviz.get("eps_growth_this_year_finviz") or info.get("earningsQuarterlyGrowth")
    metrics["eps_growth_next_year"] = info.get("earningsGrowth")
    metrics["eps_growth_next_five_years"] = info.get("earningsGrowth")
    
    # EPS growth past five years - prefer Finviz
    metrics["eps_growth_past_five_years"] = finviz.get("eps_growth_past_five_years_finviz") or info.get("earningsGrowth")
    
    # Quarterly EPS growth (QoQ) - prefer Finviz, fallback to yfinance calculation
    metrics["eps_growth_qoq"] = finviz.get("eps_growth_qoq_finviz")
    if metrics["eps_growth_qoq"] is None:
        try:
            if tk is not None:
                earnings = tk.quarterly_earnings if hasattr(tk, "quarterly_earnings") else pd.DataFrame()
                if not earnings.empty and len(earnings) >= 2:
                    latest = earnings.iloc[-1]["earnings"]
                    previous = earnings.iloc[-2]["earnings"]
                    if previous != 0 and not math.isnan(latest) and not math.isnan(previous):
                        metrics["eps_growth_qoq"] = (latest - previous) / abs(previous)
                    else:
                        metrics["eps_growth_qoq"] = None
                else:
                    metrics["eps_growth_qoq"] = None
            else:
                metrics["eps_growth_qoq"] = None
        except Exception:
            metrics["eps_growth_qoq"] = None
    
    # Sales growth (quarterly) - prefer Finviz, fallback to yfinance calculation
    metrics["sales_growth_qoq"] = finviz.get("sales_growth_qoq_finviz")
    if metrics["sales_growth_qoq"] is None:
        try:
            if isinstance(rev_row, pd.Series) and len(rev_row) >= 2:
                latest_rev = rev_row.iloc[0]
                prev_rev = rev_row.iloc[1]
                if prev_rev != 0 and not math.isnan(latest_rev) and not math.isnan(prev_rev):
                    metrics["sales_growth_qoq"] = (latest_rev - prev_rev) / abs(prev_rev)
                else:
                    metrics["sales_growth_qoq"] = None
            else:
                metrics["sales_growth_qoq"] = None
        except Exception:
            metrics["sales_growth_qoq"] = None
    
    # Sales growth past five years (approximation via CAGR)
    metrics["sales_growth_past_five_years"] = metrics["cagr"]
    
    # Quarterly earnings growth - prefer Finviz, fallback to yfinance calculation
    metrics["quarterly_earnings_growth"] = finviz.get("quarterly_earnings_growth_finviz")
    # Debug: Uncomment to check if Finviz value was found
    # if metrics["quarterly_earnings_growth"] is not None:
    #     import sys
    #     print(f"DEBUG: quarterly_earnings_growth from Finviz for {ticker}: {metrics['quarterly_earnings_growth']:.4f}", file=sys.stderr)
    if metrics["quarterly_earnings_growth"] is None:
        try:
            if tk is not None:
                earnings = tk.quarterly_earnings if hasattr(tk, "quarterly_earnings") else pd.DataFrame()
                if not earnings.empty and len(earnings) >= 2:
                    latest = earnings.iloc[-1]["earnings"]
                    previous = earnings.iloc[-2]["earnings"]
                    if previous != 0 and not math.isnan(latest) and not math.isnan(previous):
                        metrics["quarterly_earnings_growth"] = (latest - previous) / abs(previous)
                    else:
                        metrics["quarterly_earnings_growth"] = None
                else:
                    metrics["quarterly_earnings_growth"] = None
            else:
                metrics["quarterly_earnings_growth"] = None
        except Exception:
            metrics["quarterly_earnings_growth"] = None
    
    # Annual earnings growth - prefer Finviz, fallback to yfinance
    metrics["annual_earnings_growth"] = finviz.get("annual_earnings_growth_finviz") or info.get("earningsGrowth")
    
    # Institutional transactions (not directly available via yfinance, set to None)
    metrics["institutional_transactions"] = None
    
    # Graham intrinsic value (simplified: sqrt(22.5 * EPS * BVPS))
    eps = metrics["eps_ttm"]
    bvps = metrics["book_value_per_share"]
    if eps is not None and bvps is not None and eps > 0 and bvps > 0:
        metrics["graham_intrinsic_value"] = math.sqrt(22.5 * eps * bvps)
    else:
        metrics["graham_intrinsic_value"] = None
    
    # Recent news (not directly calculable, will be None for now)
    metrics["recent_news"] = None
    
    # Current price (for Graham intrinsic value comparison)
    if not hist.empty:
        metrics["current_price"] = hist["Close"].iloc[-1]
    else:
        metrics["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")

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


def quantitative(metrics: dict, data: dict = None):
    c = []

    # 6) Recurring revenue (proxy: positive decent CAGR) - renumbered from 1
    cagr_val = metrics["cagr"]
    s = 5 if cagr_val is None else score_ratio(cagr_val, 0.10, 0.05)
    c.append({
        "id": 6,
        "name": "Recurring / predictable revenue (proxy via CAGR)",
        "category": "Revenue & Growth",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"CAGR: {cagr_val:.2%}" if cagr_val is not None else "Missing / not available",
        "notes": "Measures revenue stability & predictability. Strong indicators: subscription models, contracts, essential products, repeat purchases. Red flags: one-time sales, project-based revenue, cyclical dependence, few large customers. ≥10% CAGR suggests decent recurring base."
    })

    # 7) Revenue growth - renumbered from 2
    s = score_ratio(cagr_val, 0.10, 0.05)
    c.append({
        "id": 7,
        "name": "Revenue growth",
        "category": "Revenue & Growth",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"CAGR: {cagr_val:.2%}" if cagr_val is not None else "Missing / not available",
        "notes": "Revenue is the 'top-line engine'. No growth = difficult long-term value creation. Benchmarks: ≥10% CAGR = excellent, 5-10% = decent, <5% = weak, declining = major warning sign."
    })

    # 8) FCF strength - renumbered from 3
    # FCF Margin = Free Cash Flow / Revenue
    # Scoring: ≥10% = 9/10 (Excellent), ≥5% = 7/10 (Solid), <5% = 3/10 (Weak), None = 5 (Neutral)
    f = metrics["fcf_margin"]
    if f is not None:
        if f >= 0.10:  # ≥ 10% - Excellent FCF generator
            s = 9
        elif f >= 0.05:  # ≥ 5% - Solid, acceptable
            s = 7
        else:  # < 5% - Weak FCF, may be risky
            s = 3
    else:
        s = 5  # Missing data - neutral
    
    # Get additional info for evidence
    fcf_abs = metrics.get("free_cash_flow")
    sales = metrics.get("sales_finviz")
    evidence_text = f"FCF margin: {f:.2%}" if f is not None else "Missing / not available"
    if fcf_abs is not None and sales is not None:
        evidence_text += f" (FCF: ${fcf_abs:,.0f}, Sales: ${sales:,.0f})"
    elif fcf_abs is not None:
        evidence_text += f" (FCF: ${fcf_abs:,.0f})"
    
    c.append({
        "id": 8,
        "name": "Earnings & free cash flow",
        "category": "Profitability & Efficiency",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": evidence_text,
        "notes": "Measures real earnings & free cash flow (money after maintaining business). Strong indicator: FCF margin ≥10%, consistent positive FCF. Red flags: negative FCF, heavy CapEx draining cash, earnings not supported by cash flow. FCF can fund dividends, buybacks, debt reduction, growth."
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
    
    # Format evidence with percentages
    roe_str = f"{roe:.1%}" if roe is not None else "N/A"
    nm_str = f"{nm:.1%}" if nm is not None else "N/A"
    evidence_text = f"ROE: {roe_str}, Net margin: {nm_str}"
    
    c.append({
        "id": 9,
        "name": "Profitability (ROE & margins)",  # renumbered from 4
        "category": "Profitability & Efficiency",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": evidence_text,
        "notes": "Shows efficiency & how well company converts sales to profit. ROE ≥15% = strong, <10% = weak. Net margin ≥10% = strong, <5% = weak. Red flags: falling margins (competitive pressure), low ROE (poor capital allocation)."
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
        "name": "Balance sheet strength",  # renumbered from 5
        "category": "Balance Sheet & Financial Strength",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"D/E: {dte}, Current ratio: {cr}",
        "notes": "Measures financial stability. D/E <1.0 = good, >2.0 = risky. Current ratio ≥1.5 = good, <1.0 = liquidity risk. Weak balance sheets cannot survive recessions, must raise capital (dilution), have less flexibility."
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
        "name": "Valuation (P/E)",  # renumbered from 6
        "category": "Valuation",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Trailing P/E: {pe}",
        "notes": "A great company can be a terrible investment if price is too high. Benchmarks: P/E <15 = undervalued, 15-25 = fair, >25 = overvalued (unless high growth). Red flags: sky-high P/E without growth, 'story stocks' with hype but no profits, multiples far above peers."
    })

    # 12) Technical trend
    t = metrics["trend"]
    # Get actual price and moving average values for evidence
    current_price_val = metrics.get("current_price")
    hist_data = None
    if data is not None:
        hist_data = data.get("hist")
    
    ma50_val = None
    ma150_val = None
    evidence_text = ""
    
    if hist_data is not None and not hist_data.empty:
        try:
            close = hist_data["Close"]
            if len(close) >= 150:
                ma50_val = close.rolling(50).mean().iloc[-1]
                ma150_val = close.rolling(150).mean().iloc[-1]
            elif len(close) >= 50:
                ma50_val = close.rolling(50).mean().iloc[-1]
        except Exception:
            pass
    
    if t is True:
        s = 9
        if current_price_val is not None and ma50_val is not None and ma150_val is not None:
            evidence_text = f"✓ UPTREND: Price (${current_price_val:.2f}) > 50DMA (${ma50_val:.2f}) > 150DMA (${ma150_val:.2f})"
        else:
            evidence_text = "✓ UPTREND: Price > 50DMA > 150DMA (all conditions met)"
    elif t is False:
        s = 4
        if current_price_val is not None and ma50_val is not None and ma150_val is not None:
            if current_price_val <= ma50_val:
                evidence_text = f"✗ DOWNTREND: Price (${current_price_val:.2f}) ≤ 50DMA (${ma50_val:.2f})"
            elif ma50_val <= ma150_val:
                evidence_text = f"✗ DOWNTREND: 50DMA (${ma50_val:.2f}) ≤ 150DMA (${ma150_val:.2f})"
            else:
                evidence_text = f"✗ DOWNTREND: Price (${current_price_val:.2f}) vs 50DMA (${ma50_val:.2f}) vs 150DMA (${ma150_val:.2f}) - Trend broken"
        else:
            evidence_text = "✗ DOWNTREND: Price relationship not favorable (Price ≤ 50DMA or 50DMA ≤ 150DMA)"
    else:
        s = 5
        if current_price_val is not None and ma50_val is not None:
            if ma150_val is None:
                evidence_text = f"⚠ INSUFFICIENT DATA: Have Price (${current_price_val:.2f}) and 50DMA (${ma50_val:.2f}), but need ≥150 days for 150DMA"
            else:
                evidence_text = f"⚠ INSUFFICIENT DATA: Price (${current_price_val:.2f}), 50DMA (${ma50_val:.2f}), 150DMA (${ma150_val:.2f}) - Calculation failed"
        else:
            evidence_text = "⚠ INSUFFICIENT DATA: Need at least 150 trading days of price history to calculate trend"
    
    c.append({
        "id": 12,
        "name": "Technical uptrend",  # renumbered from 7
        "category": "Technical Analysis",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": evidence_text,
        "notes": "Measures if stock is trending up. Uses Price > 50DMA > 150DMA. Momentum improves win rate because institutional buying pushes price up and trend followers join. Red flags: price below 50DMA (bearish), death cross (50DMA < 150DMA), lower lows and lower highs."
    })

    # 13) Price to Book
    pb = metrics.get("price_to_book")
    if pb is not None:
        if pb <= 1.0:
            s = 9
        elif pb <= 2.0:
            s = 7
        elif pb <= 3.0:
            s = 5
        else:
            s = 3
    else:
        s = 5
    c.append({
        "id": 13,
        "name": "Price to Book ratio",  # renumbered from 8
        "category": "Valuation",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"P/B: {pb:.2f}" if pb is not None else "Missing / not available",
        "notes": ""
    })

    # 14) Dividend Yield
    div_yield = metrics.get("dividend_yield")
    if div_yield is not None and div_yield > 0:
        if div_yield >= 0.03:
            s = 8
        elif div_yield >= 0.02:
            s = 7
        elif div_yield >= 0.01:
            s = 6
        else:
            s = 5
    else:
        s = 5
    c.append({
        "id": 14,
        "name": "Dividend Yield",  # renumbered from 9
        "category": "Valuation",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Dividend Yield: {div_yield:.2%}" if div_yield is not None else "No dividend",
        "notes": ""
    })

    # 15) Return on Assets
    roa = metrics.get("return_on_assets")
    s = score_ratio(roa, 0.10, 0.05) if roa is not None else 5
    c.append({
        "id": 15,
        "name": "Return on Assets (ROA)",  # renumbered from 10
        "category": "Profitability & Efficiency",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"ROA: {roa:.2%}" if roa is not None else "Missing / not available",
        "notes": ""
    })

    # 16) Current Ratio
    cr = metrics.get("current_ratio")
    if cr is not None:
        if cr >= 2.0:
            s = 9
        elif cr >= 1.5:
            s = 7
        elif cr >= 1.0:
            s = 5
        else:
            s = 3
    else:
        s = 5
    c.append({
        "id": 16,
        "name": "Current Ratio (Liquidity)",  # renumbered from 11
        "category": "Balance Sheet & Financial Strength",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Current Ratio: {cr:.2f}" if cr is not None else "Missing / not available",
        "notes": ""
    })

    # 17) Quick Ratio
    qr = metrics.get("quick_ratio")
    if qr is not None:
        if qr >= 1.5:
            s = 9
        elif qr >= 1.0:
            s = 7
        elif qr >= 0.5:
            s = 5
        else:
            s = 3
    else:
        s = 5
    c.append({
        "id": 17,
        "name": "Quick Ratio (Acid Test)",  # renumbered from 12
        "category": "Balance Sheet & Financial Strength",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Quick Ratio: {qr:.2f}" if qr is not None else "Missing / not available",
        "notes": ""
    })

    # 18) Six Month Return
    six_mo_ret = metrics.get("six_month_return")
    if six_mo_ret is not None:
        if six_mo_ret >= 0.20:
            s = 9
        elif six_mo_ret >= 0.10:
            s = 7
        elif six_mo_ret >= 0:
            s = 5
        else:
            s = 3
    else:
        s = 5
    c.append({
        "id": 18,
        "name": "6-Month Price Return",  # renumbered from 13
        "category": "Technical Analysis",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"6M Return: {six_mo_ret:.2%}" if six_mo_ret is not None else "Missing / not available",
        "notes": ""
    })

    # 19) Distance from 52-Week High
    dist_52w = metrics.get("distance_from_52w_high")
    if dist_52w is not None:
        if dist_52w >= -0.10:  # Within 10% of high
            s = 8
        elif dist_52w >= -0.20:  # Within 20% of high
            s = 7
        elif dist_52w >= -0.30:  # Within 30% of high
            s = 5
        else:
            s = 3
    else:
        s = 5
    c.append({
        "id": 19,
        "name": "Distance from 52-Week High",  # renumbered from 14
        "category": "Technical Analysis",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"{dist_52w:.2%} from 52w high" if dist_52w is not None else "Missing / not available",
        "notes": "Closer to high = better"
    })

    # 20) Relative Strength
    rs = metrics.get("relative_strength")
    if rs is not None:
        if rs >= 0.20:
            s = 9
        elif rs >= 0.10:
            s = 7
        elif rs >= 0:
            s = 5
        else:
            s = 3
    else:
        s = 5
    c.append({
        "id": 20,
        "name": "Relative Strength (vs SMA 200)",  # renumbered from 15
        "category": "Technical Analysis",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Relative Strength: {rs:.2%}" if rs is not None else "Missing / not available",
        "notes": ""
    })

    # 30) RSI (Relative Strength Index)
    rsi = metrics.get("rsi")
    if rsi is not None:
        # RSI > 70 = overbought (bearish), RSI < 30 = oversold (bullish)
        # RSI between 40-60 = neutral, 50-70 = bullish, 30-40 = cautious
        if 50 <= rsi <= 70:
            s = 9  # Strong bullish momentum
        elif 40 <= rsi < 50:
            s = 7  # Moderate bullish
        elif 30 <= rsi < 40:
            s = 6  # Oversold but recovering
        elif 70 < rsi <= 80:
            s = 5  # Overbought, caution
        elif rsi > 80:
            s = 3  # Extremely overbought
        elif rsi < 30:
            s = 4  # Extremely oversold
        else:
            s = 5
    else:
        s = 5
    c.append({
        "id": 21,
        "name": "RSI (Relative Strength Index)",  # renumbered from 16
        "category": "Technical Analysis",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"RSI: {rsi:.1f}" if rsi is not None else "Missing / not available (need ≥14 days)",
        "notes": "50-70 = optimal bullish zone. >70 = overbought, <30 = oversold"
    })

    # 31) MACD Signal
    macd = metrics.get("macd")
    macd_signal = metrics.get("macd_signal")
    macd_hist = metrics.get("macd_histogram")
    if macd is not None and macd_signal is not None:
        # MACD above signal = bullish, below = bearish
        if macd > macd_signal and macd_hist is not None and macd_hist > 0:
            s = 9  # Strong bullish crossover
        elif macd > macd_signal:
            s = 8  # Bullish (MACD above signal)
        elif macd < macd_signal and macd_hist is not None and macd_hist < 0:
            s = 3  # Bearish crossover
        elif macd < macd_signal:
            s = 4  # Bearish (MACD below signal)
        else:
            s = 6  # Neutral
    else:
        s = 5
    c.append({
        "id": 22,
        "name": "MACD (Moving Average Convergence Divergence)",  # renumbered from 17
        "category": "Technical Analysis",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"MACD: {macd:.2f}, Signal: {macd_signal:.2f}" if macd is not None and macd_signal is not None else "Missing / not available (need ≥26 days)",
        "notes": "MACD > Signal = bullish, MACD < Signal = bearish"
    })

    # 32) Fibonacci Retracement Position
    fib_dist = metrics.get("distance_to_nearest_fib")
    current_price_val = metrics.get("current_price")
    fib_618 = metrics.get("fib_618")
    fib_382 = metrics.get("fib_382")
    if fib_dist is not None and current_price_val is not None:
        # Near Fibonacci levels (especially 0.618 and 0.382) are important support/resistance
        if fib_dist <= 0.02:  # Within 2% of a Fib level
            # Check if price is at key Fib levels
            if fib_618 is not None and abs(current_price_val - fib_618) / current_price_val <= 0.02:
                s = 8  # At golden ratio (0.618) - strong support/resistance
            elif fib_382 is not None and abs(current_price_val - fib_382) / current_price_val <= 0.02:
                s = 8  # At 0.382 - important level
            else:
                s = 7  # Near other Fib levels
        elif fib_dist <= 0.05:  # Within 5%
            s = 6  # Close to Fib level
        else:
            s = 5  # Not near Fib levels
    else:
        s = 5
    c.append({
        "id": 23,
        "name": "Fibonacci Retracement Position",  # renumbered from 18
        "category": "Technical Analysis",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Distance to nearest Fib level: {fib_dist:.2%}" if fib_dist is not None else "Missing / not available (need ≥60 days)",
        "notes": "Price near Fibonacci levels (especially 61.8% and 38.2%) indicates potential support/resistance"
    })

    # 33) Support and Resistance Levels
    support_dist = metrics.get("distance_to_support")
    resistance_dist = metrics.get("distance_to_resistance")
    support = metrics.get("support_level")
    resistance = metrics.get("resistance_level")
    if support_dist is not None and resistance_dist is not None:
        # Price closer to support than resistance = bearish, closer to resistance = bullish
        # But also want to check if price is near key levels
        if support_dist <= 0.05 and support_dist > 0:  # Within 5% above support
            s = 9  # Near support (potential bounce)
        elif 0.05 < support_dist <= 0.10:
            s = 7  # Moderately above support
        elif resistance_dist <= 0.05:  # Within 5% below resistance
            s = 7  # Near resistance (potential breakout or rejection)
        elif resistance_dist <= 0.10:
            s = 6  # Moderately below resistance
        else:
            s = 5  # Mid-range
    else:
        s = 5
    c.append({
        "id": 24,
        "name": "Support and Resistance Levels",  # renumbered from 19
        "category": "Technical Analysis",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Support: ${support:.2f} ({support_dist:.2%} below), Resistance: ${resistance:.2f} ({resistance_dist:.2%} above)" if support is not None and resistance is not None and support_dist is not None and resistance_dist is not None else "Missing / not available",
        "notes": "Price near support = potential bounce, near resistance = potential breakout/rejection"
    })

    # 34) Volume Analysis
    vol_confirmation = metrics.get("volume_price_confirmation")
    vol_relative = metrics.get("volume_relative")
    vol_trend = metrics.get("volume_trend")
    if vol_confirmation is not None:
        if vol_confirmation == "bullish":
            # Bullish confirmation with high volume
            if vol_relative is not None and vol_relative >= 1.2:
                s = 9  # Strong bullish with high volume
            elif vol_relative is not None and vol_relative >= 1.0:
                s = 8  # Bullish with average+ volume
            else:
                s = 7  # Bullish but low volume
        elif vol_confirmation == "divergence":
            s = 4  # Volume/price divergence (warning)
        else:  # bearish
            s = 3  # Bearish volume confirmation
    elif vol_trend is not None and vol_relative is not None:
        # Fallback to volume trend if no confirmation
        if vol_trend > 0.1 and vol_relative >= 1.2:
            s = 8  # Increasing volume
        elif vol_trend > 0:
            s = 6  # Slight volume increase
        elif vol_relative >= 1.5:
            s = 7  # High volume regardless of trend
        else:
            s = 5
    else:
        s = 5
    c.append({
        "id": 25,
        "name": "Volume Analysis",  # renumbered from 20
        "category": "Technical Analysis",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Volume: {vol_confirmation}, Relative: {vol_relative:.2f}x avg" if vol_confirmation is not None and vol_relative is not None else f"Volume trend: {vol_trend:.2%}, Relative: {vol_relative:.2f}x avg" if vol_trend is not None and vol_relative is not None else "Missing / not available",
        "notes": "Bullish confirmation = price up with volume up. High relative volume (>1.2x) confirms moves"
    })

    # 35) Bollinger Bands Position (additional indicator)
    # Calculate Bollinger Bands if we have enough data
    hist_data = data.get("hist") if data is not None else None
    bb_position = None
    if hist_data is not None and not hist_data.empty:
        try:
            close = hist_data["Close"]
            if len(close) >= 20:
                bb_mean = close.tail(20).mean()
                bb_std = close.tail(20).std()
                bb_upper = bb_mean + (2 * bb_std)
                bb_lower = bb_mean - (2 * bb_std)
                current_price_val = close.iloc[-1]
                if bb_upper != bb_lower:
                    bb_position = (current_price_val - bb_lower) / (bb_upper - bb_lower)
        except Exception:
            pass
    
    if bb_position is not None:
        # Position in Bollinger Bands: 0 = lower band, 1 = upper band
        if 0.5 <= bb_position <= 0.8:
            s = 8  # Upper half but not overbought
        elif 0.2 <= bb_position < 0.5:
            s = 7  # Mid-lower half
        elif bb_position > 0.95:
            s = 4  # Near upper band (overbought)
        elif bb_position < 0.05:
            s = 6  # Near lower band (oversold but potential bounce)
        elif bb_position > 0.8:
            s = 5  # Upper region
        else:
            s = 5
    else:
        s = 5
    c.append({
        "id": 26,
        "name": "Bollinger Bands Position",  # renumbered from 21
        "category": "Technical Analysis",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Position: {bb_position:.1%} (0% = lower band, 100% = upper band)" if bb_position is not None else "Missing / not available (need ≥20 days)",
        "notes": "0.5-0.8 = optimal bullish zone. >0.95 = overbought, <0.05 = oversold"
    })

    # 36) SMA 50 Position
    sma_50 = metrics.get("sma_50")
    current_price_val = metrics.get("current_price")
    if sma_50 is not None and current_price_val is not None:
        # Price above SMA 50 = bullish, below = bearish
        price_vs_sma50 = (current_price_val - sma_50) / sma_50
        if price_vs_sma50 >= 0.10:  # Price >10% above SMA 50
            s = 9  # Strong bullish
        elif price_vs_sma50 >= 0.05:  # Price >5% above SMA 50
            s = 8  # Bullish
        elif price_vs_sma50 >= 0:  # Price above SMA 50
            s = 7  # Moderately bullish
        elif price_vs_sma50 >= -0.05:  # Price within 5% below SMA 50
            s = 5  # Neutral/cautious
        else:
            s = 3  # Bearish (price well below SMA 50)
    else:
        s = 5
        price_vs_sma50 = None
    c.append({
        "id": 27,
        "name": "SMA 50 (50-Day Moving Average)",  # renumbered from 22
        "category": "Technical Analysis",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Price: ${current_price_val:.2f}, SMA 50: ${sma_50:.2f} ({price_vs_sma50:+.2%})" if sma_50 is not None and current_price_val is not None and price_vs_sma50 is not None else f"Price: ${current_price_val:.2f}, SMA 50: ${sma_50:.2f}" if sma_50 is not None and current_price_val is not None else "Missing / not available (need ≥50 days)",
        "notes": "Price above SMA 50 = bullish trend. Price well above (>5%) = strong bullish"
    })

    # 37) SMA 200 Position
    sma_200 = metrics.get("sma_200")
    if sma_200 is not None and current_price_val is not None:
        # Price above SMA 200 = bullish, below = bearish
        price_vs_sma200 = (current_price_val - sma_200) / sma_200
        if price_vs_sma200 >= 0.10:  # Price >10% above SMA 200
            s = 9  # Strong bullish (major uptrend)
        elif price_vs_sma200 >= 0.05:  # Price >5% above SMA 200
            s = 8  # Bullish
        elif price_vs_sma200 >= 0:  # Price above SMA 200
            s = 7  # Moderately bullish (in uptrend)
        elif price_vs_sma200 >= -0.05:  # Price within 5% below SMA 200
            s = 5  # Neutral/cautious
        else:
            s = 3  # Bearish (price well below SMA 200)
    else:
        s = 5
        price_vs_sma200 = None
    c.append({
        "id": 28,
        "name": "SMA 200 (200-Day Moving Average)",  # renumbered from 23
        "category": "Technical Analysis",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Price: ${current_price_val:.2f}, SMA 200: ${sma_200:.2f} ({price_vs_sma200:+.2%})" if sma_200 is not None and current_price_val is not None and price_vs_sma200 is not None else f"Price: ${current_price_val:.2f}, SMA 200: ${sma_200:.2f}" if sma_200 is not None and current_price_val is not None else "Missing / not available (need ≥200 days)",
        "notes": "Price above SMA 200 = major bullish trend. Critical long-term trend indicator"
    })

    # 29) Quarterly Earnings Growth - renumbered from 24
    qeg = metrics.get("quarterly_earnings_growth")
    if qeg is not None:
        if qeg >= 0.20:
            s = 9
        elif qeg >= 0.10:
            s = 7
        elif qeg >= 0:
            s = 5
        else:
            s = 3
    else:
        s = 5
    c.append({
        "id": 29,
        "name": "Quarterly Earnings Growth (QoQ)",
        "category": "Revenue & Growth",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"QoQ Earnings Growth: {qeg:.2%}" if qeg is not None else "Missing / not available",
        "notes": ""
    })

    # 30) Annual Earnings Growth - renumbered from 25
    aeg = metrics.get("annual_earnings_growth")
    s = score_ratio(aeg, 0.15, 0.10) if aeg is not None else 5
    c.append({
        "id": 30,
        "name": "Annual Earnings Growth",
        "category": "Revenue & Growth",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Annual Earnings Growth: {aeg:.2%}" if aeg is not None else "Missing / not available",
        "notes": ""
    })

    # 31) EPS Growth This Year - renumbered from 26
    eps_ty = metrics.get("eps_growth_this_year")
    s = score_ratio(eps_ty, 0.20, 0.10) if eps_ty is not None else 5
    c.append({
        "id": 31,
        "name": "EPS Growth This Year",
        "category": "Revenue & Growth",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"EPS Growth YTD: {eps_ty:.2%}" if eps_ty is not None else "Missing / not available",
        "notes": ""
    })

    # 32) EPS Growth Past 5 Years - renumbered from 27
    eps_5y = metrics.get("eps_growth_past_five_years")
    s = score_ratio(eps_5y, 0.15, 0.10) if eps_5y is not None else 5
    c.append({
        "id": 32,
        "name": "EPS Growth Past 5 Years",
        "category": "Revenue & Growth",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"5Y EPS Growth: {eps_5y:.2%}" if eps_5y is not None else "Missing / not available",
        "notes": ""
    })

    # 33) EPS Growth Next 5 Years
    eps_next5 = metrics.get("eps_growth_next_five_years")
    s = score_ratio(eps_next5, 0.15, 0.10) if eps_next5 is not None else 5
    c.append({
        "id": 33,
        "name": "EPS Growth Next 5 Years",
        "category": "Revenue & Growth",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Projected 5Y EPS Growth: {eps_next5:.2%}" if eps_next5 is not None else "Missing / not available",
        "notes": "Uses analyst expectations for long-term EPS expansion. Consistent double-digit projections signal durable growth drivers."
    })

    # 34) EPS Growth QoQ - renumbered from 28
    eps_qoq = metrics.get("eps_growth_qoq")
    if eps_qoq is not None:
        if eps_qoq >= 0.15:
            s = 9
        elif eps_qoq >= 0.05:
            s = 7
        elif eps_qoq >= 0:
            s = 5
        else:
            s = 3
    else:
        s = 5
    c.append({
        "id": 34,
        "name": "EPS Growth Quarter-over-Quarter",
        "category": "Revenue & Growth",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"EPS QoQ Growth: {eps_qoq:.2%}" if eps_qoq is not None else "Missing / not available",
        "notes": ""
    })

    # 35) Sales Growth QoQ - renumbered from 29
    sales_qoq = metrics.get("sales_growth_qoq")
    if sales_qoq is not None:
        if sales_qoq >= 0.10:
            s = 9
        elif sales_qoq >= 0.05:
            s = 7
        elif sales_qoq >= 0:
            s = 5
        else:
            s = 3
    else:
        s = 5
    c.append({
        "id": 35,
        "name": "Sales Growth Quarter-over-Quarter",
        "category": "Revenue & Growth",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Sales QoQ Growth: {sales_qoq:.2%}" if sales_qoq is not None else "Missing / not available",
        "notes": ""
    })

    # 36) Operating Margin - renumbered from 30
    om = metrics.get("operating_margin")
    s = score_ratio(om, 0.15, 0.10) if om is not None else 5
    c.append({
        "id": 36,
        "name": "Operating Margin",
        "category": "Profitability & Efficiency",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"Operating Margin: {om:.2%}" if om is not None else "Missing / not available",
        "notes": ""
    })

    # 37) Free Cash Flow (Absolute) - renumbered from 31
    fcf_abs = metrics.get("free_cash_flow")
    if fcf_abs is not None and fcf_abs > 0:
        # Positive FCF is good
        s = 8
    elif fcf_abs is not None and fcf_abs <= 0:
        s = 3
    else:
        s = 5
    c.append({
        "id": 37,
        "name": "Free Cash Flow (Absolute Value)",
        "category": "Profitability & Efficiency",
        "score_0_10": s,
        "pass": s >= 7,
        "evidence": f"FCF: ${fcf_abs:,.0f}" if fcf_abs is not None else "Missing / not available",
        "notes": "Positive FCF is essential"
    })

    # 38) Graham Intrinsic Value vs Current Price - renumbered from 32
    graham_val = metrics.get("graham_intrinsic_value")
    current_price = metrics.get("current_price")  # We'll need to add this
    if graham_val is not None and current_price is not None and graham_val > 0:
        margin_of_safety = (graham_val - current_price) / graham_val
        if margin_of_safety >= 0.30:
            s = 9
        elif margin_of_safety >= 0.20:
            s = 8
        elif margin_of_safety >= 0:
            s = 6
        else:
            s = 3
        c.append({
            "id": 38,
            "name": "Graham Intrinsic Value vs Price",
            "category": "Valuation",
            "score_0_10": s,
            "pass": s >= 7,
            "evidence": f"Intrinsic: ${graham_val:.2f}, Margin of Safety: {margin_of_safety:.2%}",
            "notes": "Higher margin of safety = better value"
        })
    else:
        c.append({
            "id": 38,
            "name": "Graham Intrinsic Value vs Price",  # renumbered from 32
            "category": "Valuation",
            "score_0_10": 5,
            "pass": False,
            "evidence": "Cannot calculate - missing data",
        "notes": ""
    })

    return c


# ====================================================
# LLM SCORING (QUALITATIVE 1,3,4,5,6)
# ====================================================
def llm_prompt(ticker: str, info: dict, metrics: dict) -> str:
    # Truncate business summary to max 500 characters to save tokens
    summary = info.get("longBusinessSummary") or ""
    if len(summary) > 500:
        summary = summary[:500] + "..."
    
    # Only include essential info - remove redundant fields
    subset = {
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "summary": summary,  # Truncated
        "marketCap": info.get("marketCap"),
    }
    
    # Only include metrics directly relevant to qualitative scoring
    # Remove duplicates and non-essential metrics
    key_metrics_for_llm = {
        "insider_own": metrics.get("insider_ownership_pct"),
        "inst_own": metrics.get("institutional_ownership_pct"),
        "dilution": metrics.get("dilution_rate"),
        "shares": metrics.get("shares_outstanding"),
        "debt_eq": metrics.get("debt_to_equity"),
        "roe": metrics.get("roe"),
        "profit_m": metrics.get("profit_margin"),
        "fcf_m": metrics.get("fcf_margin"),
        "cagr": metrics.get("cagr"),
    }
    
    # Compact format - remove None values to reduce JSON size
    subset_clean = {k: v for k, v in subset.items() if v is not None}
    metrics_clean = {k: v for k, v in key_metrics_for_llm.items() if v is not None}
    
    return (
        f"Ticker: {ticker}\n"
        f"Info: {json.dumps(subset_clean, default=str)}\n"
        f"Metrics: {json.dumps(metrics_clean, default=str)}"
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
        max_tokens=800,  # Limit response size to save tokens (5 criteria * ~150 tokens each = ~750 tokens)
    )

    content = resp.choices[0].message.content
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = content

    data = json.loads(text)
    criteria = data["criteria"]
    
    # Ensure LLM criteria have proper names and notes (IDs 1-5 stay as-is - they come first)
    name_mapping = {
        1: "Competitive advantage/moat",
        2: "Large & growing market",
        3: "Management quality",
        4: "Insider ownership/alignment",
        5: "Low dilution"
    }
    notes_mapping = {
        1: "Does the company have a durable advantage protecting it from competitors? Examples: Brand power, network effects, high switching costs, cost advantage, patents/IP, scale/distribution. Red flags: No moat commodity businesses, new competitors gaining share, falling margins.",
        2: "Is the company in a market with room to grow? Positive signals: Industry CAGR >5-10%, emerging markets (EVs, AI, cloud, biotech), global expansion. Red flags: Disruption killing old industries, shrinking market, regulation killing demand.",
        3: "Are leaders competent, honest, shareholder-focused? Good signs: Transparent communication, consistent execution, long-term thinking, disciplined capital allocation. Red flags: Accounting irregularities, over-promising/under-delivering, scandals, excessive compensation.",
        4: "Do key executives own meaningful shares? When insiders own shares, incentives align. They act responsibly, avoid reckless risks, focus on long-term value. Red flags: CEO holds tiny % of shares, insiders frequently selling.",
        5: "Does the company avoid continually issuing new shares? Issuing shares = giving away pieces of business. Green flags: Stable/decreasing share count, occasional buybacks. Red flags: Frequent stock offerings, massive stock-based compensation."
    }
    for crit in criteria:
        old_id = crit.get("id")
        # Ensure name field exists, use default if missing
        if "name" not in crit and old_id in name_mapping:
            crit["name"] = name_mapping[old_id]
        elif "name" not in crit:
            crit["name"] = f"Qualitative criterion {old_id or 'unknown'}"
        
        # Ensure notes field exists with Adam Khoo-style explanation if missing
        if "notes" not in crit or not crit.get("notes"):
            if old_id in notes_mapping:
                crit["notes"] = notes_mapping[old_id]
    
    return criteria


# ====================================================
# STREAMLIT UI
# ====================================================
st.title("📊 Complete Criteria Analysis")
st.caption("LLM scores qualitative criteria; rules score quantitative criteria. All criteria displayed with pass/fail status.")

# Hide form hint using CSS and JavaScript
# Inject CSS and JavaScript to hide form hints
hide_form_hint_html = """
<style>
    /* Aggressively hide all form hints */
    .stForm small,
    form small,
    [data-testid="stForm"] small,
    .stForm > div > div > div > div > small,
    form > div > div > div > div > small,
    div[data-testid="stForm"] small,
    .stForm div small,
    form div small,
    .stForm p,
    form p,
    .stForm div[class*="small"],
    form div[class*="small"],
    .stForm *:has-text("Press Enter"),
    form *:has-text("Press Enter") {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        font-size: 0 !important;
        line-height: 0 !important;
        opacity: 0 !important;
    }
</style>
<script>
    (function() {
        function hideFormHints() {
            // Method 1: Hide all small and p tags in forms
            document.querySelectorAll('form small, .stForm small, form p, .stForm p').forEach(el => {
                el.style.display = 'none';
                el.style.visibility = 'hidden';
                el.style.height = '0';
                el.style.margin = '0';
                el.style.padding = '0';
                el.style.fontSize = '0';
            });
            
            // Method 2: Find text nodes containing "Press Enter" and hide their parents
            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );
            
            let node;
            while (node = walker.nextNode()) {
                const text = node.textContent || '';
                if (text.includes('Press Enter') || 
                    text.includes('Enter to submit') ||
                    (text.includes('Enter') && text.length < 50)) {
                    let parent = node.parentElement;
                    while (parent && parent !== document.body) {
                        if (parent.tagName === 'SMALL' || 
                            parent.tagName === 'P' || 
                            parent.classList.contains('stMarkdown') ||
                            parent.querySelector('small') ||
                            parent.textContent === text.trim()) {
                            parent.style.display = 'none';
                            parent.style.visibility = 'hidden';
                            parent.style.height = '0';
                            break;
                        }
                        parent = parent.parentElement;
                    }
                }
            }
        }
        
        // Run immediately
        hideFormHints();
        
        // Run on DOM ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', hideFormHints);
        }
        
        // Run after delays to catch dynamically added content
        setTimeout(hideFormHints, 100);
        setTimeout(hideFormHints, 500);
        setTimeout(hideFormHints, 1000);
        setTimeout(hideFormHints, 2000);
        
        // Use MutationObserver to watch for new elements
        const observer = new MutationObserver(hideFormHints);
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    })();
</script>
"""

components.html(hide_form_hint_html, height=0)

with st.sidebar:
    with st.form("ticker_form"):
        ticker = st.text_input("Ticker", "AAPL", key="ticker_input").upper().strip()
        run = st.form_submit_button("Evaluate", use_container_width=True)

if run:
    api_key = CONFIG_KEY or st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OpenAI API key not found. Set it in config.py, Streamlit Secrets, or environment.")
        st.stop()

    with st.spinner(f"Fetching data for {ticker}..."):
        data = get_data(ticker)
        metrics = compute_metrics(data, ticker)
        
        # Fetch news - try yfinance first, then Google News as fallback
        news = []
        yfinance_news_found = False
        try:
            ticker_obj = yf.Ticker(ticker)
            yf_news = ticker_obj.news
            if yf_news and isinstance(yf_news, list) and len(yf_news) > 0:
                news = yf_news
                yfinance_news_found = True
        except Exception as e:
            pass
        
        # If no news from yfinance, try Google News
        if not yfinance_news_found:
            info = data["info"]
            long_name = info.get("longName") or info.get("shortName") or ticker
            try:
                news = fetch_news_with_openai(ticker, long_name, api_key, limit=30)
            except Exception as e:
                pass

    info = data["info"]
    hist = data["hist"]
    long_name = info.get("longName") or info.get("shortName") or ticker
    price = info.get("currentPrice") or info.get("regularMarketPrice")

    st.subheader(f"📌 {long_name} ({ticker})")

    # Display recent news from last 30 days
    st.markdown("### 📰 Major News (Last 30 Days)")
    
    if news and isinstance(news, list) and len(news) > 0:
        # Filter news from last 30 days
        thirty_days_ago = datetime.now().timestamp() - (30 * 24 * 60 * 60)
        
        filtered_news = []
        
        for item in news:
            if not isinstance(item, dict):
                continue
            
            # Check top-level keys first
            pub_time = (item.get('providerPublishTime') or 
                       item.get('publishTime') or 
                       item.get('timestamp') or
                       0)
            
            # If no timestamp at top level, check inside 'content' dict
            if (not pub_time or pub_time == 0) and 'content' in item:
                content = item.get('content', {})
                if isinstance(content, dict):
                    # Try pubDate or displayTime from content
                    pub_date = content.get('pubDate') or content.get('displayTime')
                    if pub_date:
                        # pubDate might be a string or timestamp
                        if isinstance(pub_date, (int, float)):
                            pub_time = pub_date
                        elif isinstance(pub_date, str):
                            try:
                                # Try parsing as Unix timestamp string
                                pub_time = float(pub_date)
                            except:
                                try:
                                    # Try parsing as ISO date string
                                    from dateutil import parser
                                    pub_time = parser.parse(pub_date).timestamp()
                                except:
                                    pub_time = 0
                        elif isinstance(pub_date, datetime):
                            pub_time = pub_date.timestamp()
                        elif isinstance(pub_date, pd.Timestamp):
                            pub_time = pub_date.timestamp()
                        else:
                            pub_time = 0
                    else:
                        pub_time = (content.get('providerPublishTime') or 
                                   content.get('publishTime') or 
                                   content.get('timestamp') or
                                   0)
            
            # Handle datetime objects
            if isinstance(pub_time, datetime):
                pub_time = pub_time.timestamp()
            elif isinstance(pub_time, pd.Timestamp):
                pub_time = pub_time.timestamp()
            
            if not pub_time or pub_time == 0:
                continue
            
            # Only include news from last 30 days
            if pub_time >= thirty_days_ago:
                filtered_news.append(item)
        
        # Sort by publication time (most recent first)
        def get_timestamp(item):
            pub_time = (item.get('providerPublishTime') or 
                       item.get('publishTime') or 
                       item.get('timestamp') or
                       0)
            # Check content dict if not found at top level
            if (not pub_time or pub_time == 0) and 'content' in item:
                content = item.get('content', {})
                if isinstance(content, dict):
                    # Try pubDate or displayTime from content
                    pub_date = content.get('pubDate') or content.get('displayTime')
                    if pub_date:
                        # pubDate might be a string or timestamp
                        if isinstance(pub_date, (int, float)):
                            pub_time = pub_date
                        elif isinstance(pub_date, str):
                            try:
                                # Try parsing as Unix timestamp string
                                pub_time = float(pub_date)
                            except:
                                try:
                                    # Try parsing as ISO date string
                                    from dateutil import parser
                                    pub_time = parser.parse(pub_date).timestamp()
                                except:
                                    pub_time = 0
                        elif isinstance(pub_date, datetime):
                            pub_time = pub_date.timestamp()
                        elif isinstance(pub_date, pd.Timestamp):
                            pub_time = pub_date.timestamp()
                        else:
                            pub_time = 0
                    else:
                        pub_time = (content.get('providerPublishTime') or 
                                   content.get('publishTime') or 
                                   content.get('timestamp') or
                                   0)
            # Convert datetime/pd.Timestamp to timestamp
            if isinstance(pub_time, datetime):
                return pub_time.timestamp()
            elif isinstance(pub_time, pd.Timestamp):
                return pub_time.timestamp()
            return pub_time or 0
        
        filtered_news.sort(key=get_timestamp, reverse=True)
        
        # Major news sources (prioritize these)
        major_sources = [
            'Reuters', 'Bloomberg', 'Wall Street Journal', 'Financial Times', 
            'CNBC', 'MarketWatch', 'Yahoo Finance', 'Seeking Alpha', 
            'Barron\'s', 'Forbes', 'The Motley Fool', 'Investor\'s Business Daily',
            'Business Insider', 'CNN Business', 'BBC Business', 'Associated Press'
        ]
        
        # Separate major and other news
        major_news = []
        other_news = []
        
        for item in filtered_news:
            publisher = item.get('publisher', '')
            if isinstance(publisher, dict):
                publisher = publisher.get('displayName', '')
            publisher_lower = str(publisher).lower()
            
            is_major = any(source.lower() in publisher_lower for source in major_sources)
            if is_major:
                major_news.append(item)
            else:
                other_news.append(item)
        
        # Combine: major news first, then other news
        sorted_news = major_news + other_news
        
        displayed_count = 0
        # Display news items (up to 30)
        for idx, item in enumerate(sorted_news[:30], 1):
            try:
                if not isinstance(item, dict):
                    continue
                
                # Initialize publisher_url
                publisher_url = ''
                
                # Try to get publication time from various possible locations
                pub_time = None
                
                # Check top-level keys first
                pub_time = (item.get('providerPublishTime') or 
                           item.get('publishTime') or 
                           item.get('publishedAt') or
                           item.get('pubDate') or
                           item.get('timestamp'))
                
                # If not found, check in content dict
                if not pub_time and 'content' in item:
                    content = item.get('content', {})
                    if isinstance(content, dict):
                        pub_time = (content.get('providerPublishTime') or 
                                   content.get('publishTime') or 
                                   content.get('publishedAt') or
                                   content.get('pubDate') or
                                   content.get('timestamp'))
                
                # Parse the timestamp
                if pub_time:
                    try:
                        # If it's already a datetime object
                        if isinstance(pub_time, datetime):
                            pub_date = pub_time
                        # If it's a Unix timestamp (integer or float)
                        elif isinstance(pub_time, (int, float)):
                            pub_date = datetime.fromtimestamp(pub_time)
                        # If it's a string, try to parse it
                        elif isinstance(pub_time, str):
                            # Try Unix timestamp string first
                            try:
                                pub_date = datetime.fromtimestamp(float(pub_time))
                            except:
                                # Try ISO format or other date formats
                                from dateutil import parser
                                pub_date = parser.parse(pub_time)
                        else:
                            pub_date = None
                        
                        if pub_date:
                            time_str = pub_date.strftime('%Y-%m-%d %H:%M')
                        else:
                            time_str = "Unknown date"
                    except Exception as e:
                        time_str = "Unknown date"
                else:
                    time_str = "Unknown date"
                
                # Handle different news structures
                # Standard yfinance format: top-level keys (title, publisher, link, etc.)
                # Alternative format: nested in 'content' key
                # Google News format: direct keys
                
                # First try top-level keys (standard yfinance)
                title = item.get('title') or item.get('headline') or 'No title'
                
                # Get publisher - could be a string or a dict with displayName
                publisher_raw = (item.get('publisher') or 
                                item.get('source') or 
                                item.get('provider') or
                                item.get('author') or
                                item.get('site'))
                
                # Extract publisher name from dict or use string directly
                if isinstance(publisher_raw, dict):
                    publisher = publisher_raw.get('displayName') or publisher_raw.get('name') or publisher_raw.get('title') or 'Unknown'
                    publisher_url = publisher_raw.get('url') or ''
                elif isinstance(publisher_raw, str):
                    publisher = publisher_raw
                    publisher_url = ''
                else:
                    publisher = 'Unknown'
                    publisher_url = ''
                
                # Try multiple possible link key names
                # Handle case where link might be a dict
                link_raw = (item.get('link') or 
                           item.get('url') or 
                           item.get('canonicalUrl') or
                           item.get('href') or
                           item.get('webUrl') or
                           item.get('articleUrl') or
                           item.get('uri'))
                
                # Extract string URL from link if it's a dict
                if isinstance(link_raw, dict):
                    link = (link_raw.get('url') or 
                           link_raw.get('href') or 
                           link_raw.get('link') or 
                           link_raw.get('canonicalUrl') or
                           '')
                elif isinstance(link_raw, str):
                    link = link_raw
                else:
                    link = str(link_raw) if link_raw else ''
                summary = item.get('summary') or item.get('description') or item.get('text') or ''
                thumbnail = (item.get('thumbnail') or 
                           item.get('image') or 
                           item.get('thumbnailUrl') or
                           item.get('thumbnailResolutions') or
                           None)
                
                # Handle thumbnail - could be a dict with resolutions or a direct URL
                if thumbnail:
                    if isinstance(thumbnail, dict):
                        # Try to get the largest resolution
                        if 'resolutions' in thumbnail:
                            resolutions = thumbnail.get('resolutions', [])
                            if resolutions:
                                thumbnail = resolutions[-1].get('url') if isinstance(resolutions[-1], dict) else None
                        elif 'url' in thumbnail:
                            thumbnail = thumbnail.get('url')
                        else:
                            thumbnail = None
                    elif not isinstance(thumbnail, str):
                        thumbnail = None
                
                # If no title at top level, check if there's a 'content' key
                if 'content' in item:
                    content = item.get('content', {})
                    if isinstance(content, dict):
                        # content is a dict, extract from it
                        if not title or title == 'No title':
                            title = content.get('title') or title or 'No title'
                        if not publisher or publisher == 'Unknown':
                            publisher_raw = (content.get('publisher') or 
                                            content.get('source') or 
                                            content.get('provider'))
                            if isinstance(publisher_raw, dict):
                                publisher = publisher_raw.get('displayName') or publisher_raw.get('name') or publisher
                                if not publisher_url:
                                    publisher_url = publisher_raw.get('url') or ''
                            elif isinstance(publisher_raw, str):
                                publisher = publisher_raw
                        if not link:
                            link = (content.get('link') or 
                                   content.get('url') or 
                                   content.get('canonicalUrl') or
                                   content.get('href') or
                                   content.get('webUrl') or
                                   content.get('articleUrl') or
                                   link)
                        if not summary:
                            summary = content.get('summary') or content.get('description') or summary
                        if not thumbnail:
                            thumbnail = (content.get('thumbnail') or 
                                      content.get('image') or 
                                      thumbnail)
                    elif isinstance(content, str):
                        # content might be a JSON string
                        try:
                            content_dict = json.loads(content)
                            if isinstance(content_dict, dict):
                                if not title or title == 'No title':
                                    title = content_dict.get('title') or title or 'No title'
                                if not publisher or publisher == 'Unknown':
                                    publisher_raw = (content_dict.get('publisher') or 
                                                   content_dict.get('source') or 
                                                   content_dict.get('provider'))
                                    if isinstance(publisher_raw, dict):
                                        publisher = publisher_raw.get('displayName') or publisher_raw.get('name') or publisher
                                        if not publisher_url:
                                            publisher_url = publisher_raw.get('url') or ''
                                    elif isinstance(publisher_raw, str):
                                        publisher = publisher_raw
                                if not link:
                                    link = (content_dict.get('link') or 
                                           content_dict.get('url') or 
                                           content_dict.get('canonicalUrl') or
                                           content_dict.get('href') or
                                           content_dict.get('webUrl') or
                                           content_dict.get('articleUrl') or
                                           link)
                                if not summary:
                                    summary = content_dict.get('summary') or content_dict.get('description') or summary
                                if not thumbnail:
                                    thumbnail = (content_dict.get('thumbnail') or 
                                              content_dict.get('image') or 
                                              thumbnail)
                        except:
                            pass
                
                # Show if we have any title (even if it's "No title", we'll still show it)
                if title:
                    displayed_count += 1
                    # Create expander label with clickable title if link is available
                    if link:
                        # Use HTML to make title clickable in expander (Streamlit supports some HTML)
                        expander_label = f"{displayed_count}. {title} - {time_str}"
                    else:
                        expander_label = f"{displayed_count}. {title} - {time_str}"
                    
                    with st.expander(expander_label):
                        # Show clickable title at the top - this is the main link to the article
                        # Ensure link is a string before processing
                        link_str = None
                        if link:
                            if isinstance(link, str):
                                link_str = link.strip()
                            elif isinstance(link, dict):
                                # If link is a dict, try to extract URL from it
                                link_str = (link.get('url') or link.get('href') or link.get('link') or '').strip()
                            else:
                                link_str = str(link).strip() if link else ''
                        
                        # Show thumbnail if available
                        if thumbnail:
                            try:
                                st.image(thumbnail, width=300)
                            except:
                                pass
                        
                        if summary:
                            st.write(f"**Summary:** {summary}")
                        
                        # Add "More" button linking to article
                        if link_str and link_str.startswith('http'):
                            st.markdown(f"[**More →**]({link_str})")
                        
                        # Publisher and Published date at the bottom
                        col1, col2 = st.columns(2)
                        with col1:
                            # Make publisher name link to the article
                            # Use link_str from above if available
                            if link_str and link_str.startswith('http'):
                                st.markdown(f"**Publisher:** [{publisher}]({link_str})")
                            elif publisher_url:
                                st.markdown(f"**Publisher:** [{publisher}]({publisher_url})")
                            else:
                                st.write(f"**Publisher:** {publisher}")
                        with col2:
                            st.write(f"**Published:** {time_str}")
                else:
                    pass  # Skip items without title
            except Exception as e:
                continue  # Skip items that fail to process
    else:
        st.info("No recent news available for this ticker.")

    c1, c2 = st.columns([1.5, 2])

    with c1:
        st.metric("Price", f"{price:.2f}" if price is not None else "N/A")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        if metrics["cagr"] is not None:
            st.write(f"**Revenue CAGR (approx):** {metrics['cagr']:.2%}")
        if metrics["roe"] is not None:
            st.write(f"**ROE:** {metrics['roe']:.2%}")
        if metrics["net_margin"] is not None:
            st.write(f"**Net margin:** {metrics['net_margin']:.2%}")

    with c2:
        st.write("### 📈 1-Year Price")
        if not hist.empty:
            # Create chart with news markers
            fig = create_stock_chart_with_news(hist, news, ticker)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(hist["Close"])
        else:
            st.info("No price history available.")

    with st.spinner("LLM evaluating qualitative criteria..."):
        qual_criteria = llm_score(api_key, ticker, info, metrics)

    quant_criteria = quantitative(metrics, data)

    # Ensure all criteria have category field (for LLM results that might not have it)
    for crit in qual_criteria:
        if "category" not in crit:
            crit["category"] = "Qualitative Fundamentals"

    # Merge ALL criteria by ID (not just 1-12)
    all_crit = {c["id"]: c for c in qual_criteria + quant_criteria}
    
    # Group by category
    from collections import defaultdict
    by_category = defaultdict(list)
    for crit_id in sorted(all_crit.keys()):
        crit = all_crit[crit_id]
        category = crit.get("category", "Other")
        by_category[category].append(crit)
    
    # Define category display order
    category_order = [
        "Qualitative Fundamentals",
        "Revenue & Growth",
        "Profitability & Efficiency",
        "Balance Sheet & Financial Strength",
        "Valuation",
        "Technical Analysis",
        "Other"
    ]
    
    # Sort categories and flatten - sort by ID within each category
    ordered = []
    for cat in category_order:
        if cat in by_category:
            ordered.extend(sorted(by_category[cat], key=lambda x: x["id"]))
    # Add any remaining categories not in the order list
    for cat, crits in by_category.items():
        if cat not in category_order:
            ordered.extend(sorted(crits, key=lambda x: x["id"]))
    
    # Renumber all criteria sequentially from 1, 2, 3... going down
    for idx, crit in enumerate(ordered, start=1):
        crit["display_id"] = idx  # Store original ID for reference if needed
        crit["id"] = idx  # Update ID to sequential number

    scores = [c["score_0_10"] for c in ordered]
    overall = sum(scores) / len(scores) if scores else 0
    
    # Create summary using Adam Khoo methodology
    def generate_summary(all_criteria_dict, overall_score):
        """Generate Adam Khoo-style summary based on criteria scores"""
        summary_parts = []
        strengths = []
        weaknesses = []
        warnings = []
        
        # Core 12 criteria (IDs 1-12)
        core_criteria_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        core_criteria = {id: all_criteria_dict.get(id) for id in core_criteria_ids if id in all_criteria_dict}
        
        # Analyze qualitative fundamentals (1-5)
        qual_scores = [core_criteria[id]["score_0_10"] for id in [1, 2, 3, 4, 5] if id in core_criteria]
        qual_avg = sum(qual_scores) / len(qual_scores) if qual_scores else 0
        
        # Analyze quantitative fundamentals (6-12)
        quant_scores = [core_criteria[id]["score_0_10"] for id in [6, 7, 8, 9, 10, 11, 12] if id in core_criteria]
        quant_avg = sum(quant_scores) / len(quant_scores) if quant_scores else 0
        
        # Categorize by score
        for id, crit in core_criteria.items():
            score = crit.get("score_0_10", 0)
            name = crit.get("name", f"Criterion {id}")
            if score >= 8:
                strengths.append(f"{name} (Score: {score:.1f}/10)")
            elif score < 5:
                weaknesses.append(f"{name} (Score: {score:.1f}/10)")
            elif score < 7:
                warnings.append(f"{name} (Score: {score:.1f}/10)")
        
        # Build summary
        summary_parts.append("### 📋 Investment Analysis Summary (Adam Khoo Methodology)")
        
        # Overall assessment
        if overall_score >= 8:
            summary_parts.append("**Overall Assessment: STRONG CANDIDATE** ✅")
            summary_parts.append("This stock meets most of Adam Khoo's 12 criteria and appears to be a solid investment opportunity.")
        elif overall_score >= 5:
            summary_parts.append("**Overall Assessment: MIXED / BORDERLINE** ⚠️")
            summary_parts.append("This stock has mixed signals - some strong points but also areas of concern. Proceed with caution.")
        else:
            summary_parts.append("**Overall Assessment: WEAK / RISKY** ❌")
            summary_parts.append("This stock fails multiple criteria and presents significant investment risk. Consider avoiding or wait for improvement.")
        
        summary_parts.append("")
        
        # Category breakdown
        summary_parts.append(f"**Qualitative Fundamentals (Criteria 1-5):** {qual_avg:.1f}/10")
        summary_parts.append(f"**Quantitative Fundamentals (Criteria 6-12):** {quant_avg:.1f}/10")
        summary_parts.append("")
        
        # Strengths
        if strengths:
            summary_parts.append("**✅ Key Strengths:**")
            for strength in strengths[:5]:  # Top 5 strengths
                summary_parts.append(f"- {strength}")
            summary_parts.append("")
        
        # Weaknesses
        if weaknesses:
            summary_parts.append("**❌ Key Weaknesses:**")
            for weakness in weaknesses[:5]:  # Top 5 weaknesses
                summary_parts.append(f"- {weakness}")
            summary_parts.append("")
        
        # Warnings
        if warnings and len(warnings) > 0:
            summary_parts.append("**⚠️ Areas of Concern:**")
            for warning in warnings[:3]:  # Top 3 warnings
                summary_parts.append(f"- {warning}")
            summary_parts.append("")
        
        # Recommendations
        summary_parts.append("**💡 Investment Recommendation:**")
        if overall_score >= 8:
            summary_parts.append("Consider this stock for your portfolio. It demonstrates strong fundamentals across multiple criteria.")
        elif overall_score >= 5:
            summary_parts.append("Monitor closely or wait for improvement in weak areas before investing. Consider dollar-cost averaging if proceeding.")
        else:
            summary_parts.append("Avoid this stock or wait for significant improvement in fundamentals before considering investment.")
        
        return "\n".join(summary_parts)
    
    summary_text = generate_summary(all_crit, overall)

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

    # Display Summary
    st.markdown("---")
    st.markdown(summary_text)

    st.markdown("---")
    st.subheader(f"✅ Complete Criteria Checklist ({len(ordered)} criteria)")

    # Display criteria grouped by category
    current_category = None
    for c in ordered:
        cid = c.get("id", "?")
        name = c.get("name", f"Criterion {cid}")  # Fallback if name missing
        category = c.get("category", "Other")
        score = c.get("score_0_10", 0)
        passed = c.get("pass", False)
        evidence = c.get("evidence", "")
        notes = c.get("notes", "")

        # Show category header when category changes
        if category != current_category:
            if current_category is not None:
                st.markdown("<br>", unsafe_allow_html=True)  # Space between categories
            
            # Category header styling
            category_colors = {
                "Qualitative Fundamentals": "#4A90E2",
                "Revenue & Growth": "#50C878",
                "Profitability & Efficiency": "#FF6B6B",
                "Balance Sheet & Financial Strength": "#9B59B6",
                "Valuation": "#F39C12",
                "Technical Analysis": "#1ABC9C",
                "Other": "#95A5A6"
            }
            cat_color = category_colors.get(category, "#95A5A6")
            
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(90deg, {cat_color} 0%, {cat_color}22 100%);
                    padding: 12px 16px;
                    margin: 20px 0 10px 0;
                    border-radius: 8px;
                    border-left: 6px solid {cat_color};
                ">
                    <h3 style="margin:0; color: white; font-weight: 700; font-size: 1.2em;">
                        📁 {category}
                    </h3>
                </div>
                """,
                unsafe_allow_html=True,
            )
            current_category = category

        # Determine pass/fail status and colors
        if passed:
            icon = "✅ PASS"
            bg = "#e8ffe8"
            border = "#0f9d58"
            status_color = "#0f9d58"
            reason = "✓ Meets minimum threshold (score ≥ 7)"
        elif score >= 5:
            icon = "⚠️ WARNING"
            bg = "#fff7cc"
            border = "#ffcc00"
            status_color = "#f57c00"
            reason = "⚠ Borderline - does not meet threshold (score < 7)"
        else:
            icon = "❌ FAIL"
            bg = "#ffe6e6"
            border = "#cc0000"
            status_color = "#cc0000"
            reason = "✗ Does not meet threshold (score < 7)"

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
                <h4 style="margin:0; display:flex; justify-content:space-between; align-items:center;">
                    <span>{icon} <strong>{cid}.</strong> {name}</span>
                    <span style="font-weight:600; color:{status_color};">Score: {score:.1f}/10</span>
                </h4>
                <p style="margin:8px 0 4px 0; color:{status_color}; font-weight:600;">
                    <b>Status:</b> {reason}
                </p>
                <p style="margin:4px 0;"><b>Evidence:</b> {evidence}</p>
                {("<p style='margin:4px 0;'><b>Notes:</b> " + notes + "</p>") if notes else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.info("Enter a ticker and click **Evaluate**. API key is loaded from config / secrets / env.")
