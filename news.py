import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

# Set page config
st.set_page_config(
    page_title="Gold Chart with Macro News",
    layout="wide",
)

# Gold ticker symbols
GOLD_TICKERS = {
    "Gold Futures (GC=F)": "GC=F",
    "Gold ETF (GLD)": "GLD",
    "Gold Spot (XAUUSD=X)": "XAUUSD=X"
}

# Additional curated RSS feeds for category-based news
RSS_FEEDS = {
    "Politics": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
    "Sports": "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml",
    "Nature": "https://rss.nytimes.com/services/xml/rss/nyt/Climate.xml",
    "Business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "Technology": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
}

REGIONAL_FEEDS = {
    "Canada": [
        "https://rss.cbc.ca/lineup/topstories.xml",
        "https://www.thestar.com/feeds.articles.news.canada.rss",
    ],
    "Vietnam": [
        "https://vnexpress.net/rss/tin-moi-nhat.rss",
        "https://vietnamnews.vn/rss/general.rss",
    ],
}

SEEKING_ALPHA_ECONOMY_FEED = "https://seekingalpha.com/market-news/economy/rss"
MINING_COM_GOLD_FEED = "https://www.mining.com/tag/gold/feed/"
MARKETWATCH_BULLETINS_FEED = "https://feeds.content.dowjones.io/public/rss/mw_bulletins"
MARKETWATCH_TOPSTORIES_FEED = "https://feeds.content.dowjones.io/public/rss/mw_topstories"

def fetch_gold_data(ticker="GC=F", interval="15m", period="1d"):
    """
    Fetch gold price data with specified interval.
    
    Args:
        ticker: Gold ticker symbol
        interval: Data interval (15m, 1h, 1d, etc.)
        period: Period to fetch (1d, 5d, 1mo, etc.)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        gold = yf.Ticker(ticker)
        hist = gold.history(period=period, interval=interval)
        
        if hist.empty:
            st.warning(f"No data available for {ticker}. Trying alternative ticker...")
            # Try alternative ticker
            if ticker == "GC=F":
                hist = yf.Ticker("GLD").history(period=period, interval=interval)
            elif ticker == "GLD":
                hist = yf.Ticker("XAUUSD=X").history(period=period, interval=interval)
        
        return hist
    except Exception as e:
        st.error(f"Error fetching gold data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def fetch_seeking_alpha_macro_news(limit: int = 10):
    """Fetch macroeconomic news headlines from Seeking Alpha's economy feed."""
    items = []
    try:
        resp = requests.get(SEEKING_ALPHA_ECONOMY_FEED, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
        for entry in soup.find_all("item")[:limit]:
            title = (entry.title.text if entry.title else "No title").strip()
            link = entry.link.text.strip() if entry.link else ""
            pub_text = entry.pubDate.text.strip() if entry.pubDate else ""
            try:
                pub_dt = parsedate_to_datetime(pub_text) if pub_text else datetime.utcnow()
            except Exception:
                pub_dt = datetime.utcnow()
            items.append(
                {
                    "timestamp": pub_dt,
                    "title": title,
                    "publisher": "Seeking Alpha",
                    "link": link,
                    "type": "macro",
                    "source": "Seeking Alpha",
                }
            )
    except Exception:
        pass
    return items

@st.cache_data(ttl=600, show_spinner=False)
def fetch_mining_com_gold_news(limit: int = 10):
    """Fetch gold-specific news from Mining.com gold RSS feed."""
    items = []
    try:
        resp = requests.get(MINING_COM_GOLD_FEED, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
        for entry in soup.find_all("item")[:limit]:
            title = (entry.title.text if entry.title else "No title").strip()
            link = entry.link.text.strip() if entry.link else ""
            publisher = "Mining.com"
            pub_text = entry.pubDate.text.strip() if entry.pubDate else ""
            try:
                pub_dt = parsedate_to_datetime(pub_text) if pub_text else datetime.utcnow()
            except Exception:
                pub_dt = datetime.utcnow()
            items.append(
                {
                    "timestamp": pub_dt,
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "type": "gold",
                    "source": "Mining.com",
                }
            )
    except Exception:
        pass
    return items

def _classify_marketwatch_item(title: str) -> str:
    """Determine whether a MarketWatch headline is macro, gold, or general."""
    macro_keywords = [
        'fed', 'federal reserve', 'inflation', 'gdp', 'employment', 
        'interest rate', 'cpi', 'ppi', 'unemployment', 'central bank', 
        'monetary policy', 'fiscal', 'economic', 'economy', 'recession',
        'growth', 'deficit', 'surplus', 'trade', 'tariff', 'currency',
        'dollar', 'euro', 'yen', 'pound', 'bond', 'treasury', 'yield',
        'jobless', 'payroll', 'retail sales', 'manufacturing', 'pmi',
        'consumer confidence', 'housing', 'mortgage', 'foreclosure',
        'stock market', 'dow', 'nasdaq', 's&p', 'bull market', 'bear market'
    ]
    gold_keywords = [
        'gold', 'xau', 'bullion', 'gold price', 'gold futures',
        'gold etf', 'precious metal', 'gold demand', 'gold supply',
        'gold mining', 'gold market'
    ]
    title_lower = title.lower()
    if any(keyword in title_lower for keyword in gold_keywords):
        return 'gold'
    if any(keyword in title_lower for keyword in macro_keywords):
        return 'macro'
    return 'general'


def _fetch_marketwatch_feed(feed_url: str, limit: int, source_label: str):
    """Generic helper to fetch and classify MarketWatch RSS feeds."""
    items = []
    try:
        resp = requests.get(feed_url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
        for entry in soup.find_all("item")[:limit]:
            title = (entry.title.text if entry.title else "No title").strip()
            link = entry.link.text.strip() if entry.link else ""
            pub_text = entry.pubDate.text.strip() if entry.pubDate else ""
            try:
                pub_dt = parsedate_to_datetime(pub_text) if pub_text else datetime.utcnow()
            except Exception:
                pub_dt = datetime.utcnow()
            news_type = _classify_marketwatch_item(title)
            # Only keep macro and gold items from MarketWatch feeds
            if news_type in ("macro", "gold"):
                items.append(
                    {
                        "timestamp": pub_dt,
                        "title": title,
                        "publisher": "MarketWatch",
                        "link": link,
                        "type": news_type,
                        "source": source_label,
                    }
                )
    except Exception:
        pass
    return items


@st.cache_data(ttl=600, show_spinner=False)
def fetch_marketwatch_news(limit: int = 15):
    """Fetch breaking financial news from MarketWatch bulletins RSS feed."""
    return _fetch_marketwatch_feed(MARKETWATCH_BULLETINS_FEED, limit, "MarketWatch Bulletins")


@st.cache_data(ttl=600, show_spinner=False)
def fetch_marketwatch_topstories(limit: int = 15):
    """Fetch MarketWatch top stories feed."""
    return _fetch_marketwatch_feed(MARKETWATCH_TOPSTORIES_FEED, limit, "MarketWatch Top Stories")

@st.cache_data(ttl=600, show_spinner=False)
def fetch_google_news_macro(limit: int = 15):
    """Fetch macroeconomic news from Google News RSS."""
    items = []
    try:
        search_query = "macroeconomic news economy inflation GDP interest rates Federal Reserve"
        encoded_query = quote_plus(search_query)
        google_news_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(google_news_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            feed_items = root.findall('.//item')[:limit]
            
            for item in feed_items:
                try:
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    pub_date_elem = item.find('pubDate')
                    source_elem = item.find('source')
                    description_elem = item.find('description')
                    
                    if title_elem is not None and title_elem.text:
                        title = title_elem.text.strip()
                        # Remove common prefixes
                        if title.startswith('- '):
                            title = title[2:].strip()
                    else:
                        continue
                    
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
                    
                    # Parse date
                    pub_date = pub_date_elem.text.strip() if (pub_date_elem is not None and pub_date_elem.text) else ""
                    try:
                        if pub_date:
                            from dateutil import parser
                            pub_dt = parser.parse(pub_date)
                            if pub_dt.tzinfo is None:
                                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                        else:
                            pub_dt = datetime.now(timezone.utc)
                    except:
                        pub_dt = datetime.now(timezone.utc)
                    
                    items.append({
                        "timestamp": pub_dt,
                        "title": title,
                        "publisher": source,
                        "link": link,
                        "type": "macro",
                        "source": "Google News"
                    })
                except Exception:
                    continue
    except Exception:
        pass
    return items

@st.cache_data(ttl=600, show_spinner=False)
def fetch_google_news_gold(limit: int = 15):
    """Fetch gold-related news from Google News RSS."""
    items = []
    try:
        search_query = "gold price gold market gold futures bullion precious metals"
        encoded_query = quote_plus(search_query)
        google_news_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(google_news_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            feed_items = root.findall('.//item')[:limit]
            
            for item in feed_items:
                try:
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    pub_date_elem = item.find('pubDate')
                    source_elem = item.find('source')
                    description_elem = item.find('description')
                    
                    if title_elem is not None and title_elem.text:
                        title = title_elem.text.strip()
                        # Remove common prefixes
                        if title.startswith('- '):
                            title = title[2:].strip()
                    else:
                        continue
                    
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
                    
                    # Parse date
                    pub_date = pub_date_elem.text.strip() if (pub_date_elem is not None and pub_date_elem.text) else ""
                    try:
                        if pub_date:
                            from dateutil import parser
                            pub_dt = parser.parse(pub_date)
                            if pub_dt.tzinfo is None:
                                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                        else:
                            pub_dt = datetime.now(timezone.utc)
                    except:
                        pub_dt = datetime.now(timezone.utc)
                    
                    items.append({
                        "timestamp": pub_dt,
                        "title": title,
                        "publisher": source,
                        "link": link,
                        "type": "gold",
                        "source": "Google News"
                    })
                except Exception:
                    continue
    except Exception:
        pass
    return items

def fetch_macro_news(days_window: int = 1):
    """
    Fetch macroeconomic/gold news from various sources within a rolling window.
    days_window controls how far back we look (e.g., 1 = last 24h, 5 = last 5 days).
    """
    news_items = []
    max_age_seconds = max(days_window, 1) * 86400
    now_utc = datetime.now(timezone.utc)
    
    # Keyword buckets
    macro_keywords = [
        'fed', 'federal reserve', 'inflation', 'gdp', 'employment', 
        'interest rate', 'cpi', 'ppi', 'unemployment', 'central bank', 
        'monetary policy', 'fiscal', 'economic', 'economy', 'recession',
        'growth', 'deficit', 'surplus', 'trade', 'tariff', 'currency',
        'dollar', 'euro', 'yen', 'pound', 'bond', 'treasury', 'yield',
        'jobless', 'payroll', 'retail sales', 'manufacturing', 'pmi',
        'consumer confidence', 'housing', 'mortgage', 'oil', 'crude',
        'energy', 'commodity', 'silver', 'precious metal'
    ]

    gold_keywords = [
        'gold', 'xau', 'bullion', 'gold price', 'gold futures',
        'gold etf', 'precious metal', 'gold demand', 'gold supply',
        'gold mining', 'gold market', 'goldman sachs gold'
    ]
    
    # Try fetching from Yahoo Finance news for multiple tickers
    tickers_to_check = ["GC=F", "GLD", "^GSPC", "^DJI", "DX-Y.NYB"]  # Gold, S&P, Dow, Dollar Index
    
    for ticker_symbol in tickers_to_check:
        try:
            ticker = yf.Ticker(ticker_symbol)
            news = ticker.news
            
            if not news:
                continue
                
            for item in news[:15]:  # Limit per ticker
                try:
                    # Parse timestamp
                    pub_time_ts = item.get('providerPublishTime', 0)
                    if pub_time_ts == 0:
                        continue
                    
                    # Make timestamp timezone-aware in UTC
                    pub_time = datetime.fromtimestamp(pub_time_ts, tz=timezone.utc)
                    
                    # Include only items within our rolling window
                    time_diff = now_utc - pub_time
                    if time_diff.total_seconds() > max_age_seconds:
                        continue
                    
                    title = item.get('title', 'No title')
                    summary = item.get('summary', '') or item.get('provider', '')
                    text_blob = f"{title} {summary}".lower()
                    
                    is_gold = any(keyword in text_blob for keyword in gold_keywords)
                    is_macro = any(keyword in text_blob for keyword in macro_keywords)
                    
                    # Avoid duplicates
                    if any(existing['title'].lower() == item.get('title', '').lower() 
                           for existing in news_items):
                        continue
                    
                    news_type = 'gold' if is_gold else ('macro' if is_macro else 'general')

                    news_items.append({
                        'timestamp': pub_time,
                        'title': title,
                        'publisher': item.get('publisher', 'Unknown'),
                        'link': item.get('link', ''),
                        'type': news_type
                    })
                except Exception:
                    continue
        except Exception:
            continue
    # Supplement with Seeking Alpha economy headlines
    sa_items = fetch_seeking_alpha_macro_news(limit=12)
    merge_feed_items(sa_items, news_items, now_utc, max_age_seconds)

    # Supplement with Mining.com gold headlines
    mining_items = fetch_mining_com_gold_news(limit=12)
    merge_feed_items(mining_items, news_items, now_utc, max_age_seconds)

    # Supplement with MarketWatch feeds
    mw_items = fetch_marketwatch_news(limit=15)
    merge_feed_items(mw_items, news_items, now_utc, max_age_seconds)

    mw_top_items = fetch_marketwatch_topstories(limit=15)
    merge_feed_items(mw_top_items, news_items, now_utc, max_age_seconds)

    # Supplement with Google News macroeconomic news
    google_macro_items = fetch_google_news_macro(limit=15)
    merge_feed_items(google_macro_items, news_items, now_utc, max_age_seconds)

    # Supplement with Google News gold news
    google_gold_items = fetch_google_news_gold(limit=15)
    merge_feed_items(google_gold_items, news_items, now_utc, max_age_seconds)

    # Sort by timestamp (most recent first) and remove duplicates
    news_items.sort(key=lambda x: x['timestamp'], reverse=True)
    news_items = deduplicate_news(news_items)
    
    # Return full list (chart and UI will decide which types to show where)
    return news_items

@st.cache_data(ttl=600, show_spinner=False)
def fetch_category_news(feed_url: str, limit: int = 5):
    """Fetch additional news for a specific category via RSS."""
    headlines = []
    try:
        resp = requests.get(feed_url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
        items = soup.find_all("item")[:limit]
        for item in items:
            title = item.title.text if item.title else "Untitled"
            link = item.link.text if item.link else ""
            published = item.pubDate.text if item.pubDate else ""
            headlines.append(
                {
                    "title": title.strip(),
                    "link": link.strip(),
                    "published": published.strip(),
                }
            )
    except Exception:
        pass
    return headlines


def fetch_categorized_news(limit_per_category: int = 4):
    """Fetch multiple categories of news from curated RSS feeds."""
    categorized = {}
    for category, feed in RSS_FEEDS.items():
        headlines = fetch_category_news(feed, limit_per_category)
        if headlines:
            categorized[category] = headlines
    return categorized


def merge_feed_items(source_items, news_items, now_utc, max_age_seconds: int = 86400):
    """Merge external feed items into primary news list with deduplication."""
    for item in source_items:
        ts = item.get("timestamp")
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        if (now_utc - ts).total_seconds() > max_age_seconds:
            continue
        title = item.get("title", "")
        if not title:
            continue
        if any(existing.get("title", "").lower() == title.lower() for existing in news_items):
            continue
        item["timestamp"] = ts
        news_items.append(item)


def deduplicate_news(items):
    """Remove duplicate news entries based on title/link."""
    seen = set()
    unique_items = []
    for item in items:
        title = (item.get("title") or "").strip().lower()
        link = (item.get("link") or "").strip().lower()
        key = (title, link)
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)
    return unique_items


def chunk_list(seq, size):
    """Yield successive chunks from seq."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


@st.cache_data(ttl=600, show_spinner=False)
def fetch_regional_news(limit_per_region: int = 5):
    """Fetch regional news for predefined countries."""
    regional = {}
    for region, feeds in REGIONAL_FEEDS.items():
        aggregated = []
        seen_titles = set()
        for feed in feeds:
            feed_headlines = fetch_category_news(feed, limit_per_region)
            for item in feed_headlines:
                title_key = item["title"].lower()
                if title_key in seen_titles:
                    continue
                aggregated.append(item)
                seen_titles.add(title_key)
        if aggregated:
            regional[region] = aggregated[:limit_per_region]
    return regional


def create_gold_chart_with_news(df, news_items):
    """
    Create an interactive Plotly chart with gold price and news markers.
    
    Args:
        df: DataFrame with gold price data (OHLCV)
        news_items: List of news items with timestamps
    """
    if df.empty:
        st.error("No gold price data available to plot.")
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
        st.error("No valid gold price data available to plot.")
        return None
    
    # Use filtered dataframe
    df = df_filtered
    
    # Create subplots: price chart on top, volume on bottom
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Gold Price (15-minute)', 'Volume')
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Gold Price'
        ),
        row=1, col=1
    )
    
    # Add news markers
    news_x = []
    news_y = []
    news_titles = []
    news_links = []
    
    for news in news_items:
        raw_time = news.get('timestamp')
        if raw_time is None:
            continue
        news_time = pd.Timestamp(raw_time)
        
        # Handle timezone issues
        if news_time.tz is None:
            # Assume UTC if no timezone info
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
                news_titles.append(news['title'])
                news_links.append(news.get('link', ''))
        except Exception:
            # Skip if we can't find a matching timestamp
            continue
    
    # Add news markers as scatter points
    if news_x and len(news_x) > 0:
        # Truncate titles for display
        display_titles = [title[:40] + '...' if len(title) > 40 else title for title in news_titles]
        
        fig.add_trace(
            go.Scatter(
                x=news_x,
                y=news_y,
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                name='Macro News',
                hovertemplate='<b>%{text}</b><br>Time: %{x|%Y-%m-%d %H:%M}<br>Price: $%{y:.2f}<extra></extra>',
                text=display_titles,
                customdata=news_links
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
    
    # Add volume bars
    if 'Volume' in df.columns:
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
    
    # Update layout
    fig.update_layout(
        title='Gold Price Chart with Macroeconomic News Markers (15-minute timeframe)',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        yaxis2_title='Volume',
        height=800,
        hovermode='x unified',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    # Update x-axis to show time properly
    fig.update_xaxes(
        tickformat='%H:%M',
        row=1, col=1
    )
    fig.update_xaxes(
        tickformat='%H:%M',
        row=2, col=1
    )
    
    return fig

def main():
    st.title("📈 Gold Chart with Macroeconomic News")
    st.caption("15-minute timeframe with today's macroeconomic news markers")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        # Ticker selection
        selected_ticker_name = st.selectbox(
            "Gold Instrument",
            options=list(GOLD_TICKERS.keys()),
            index=0
        )
        selected_ticker = GOLD_TICKERS[selected_ticker_name]
        
        # Period selection
        period = st.selectbox(
            "Time Period",
            options=["1d", "5d", "1mo"],
            index=2,  # Default to 1mo
            help="Period of data to fetch"
        )
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Fetch data
    with st.spinner("Fetching gold price data..."):
        df = fetch_gold_data(ticker=selected_ticker, interval="15m", period=period)
    
    if df.empty:
        st.error("❌ Unable to fetch gold price data. Please try again later.")
        st.info("💡 Tip: Make sure you have an internet connection and the market is open.")
        return
    
    # Display current price info
    if not df.empty:
        current_price = df['Close'].iloc[-1]
        price_change = current_price - df['Close'].iloc[0]
        price_change_pct = (price_change / df['Close'].iloc[0]) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
        with col3:
            st.metric("High (Period)", f"${df['High'].max():.2f}")
        with col4:
            st.metric("Low (Period)", f"${df['Low'].min():.2f}")
    
    # Fetch news (window depends on selected period)
    if period == "1d":
        days_window = 1
    elif period == "5d":
        days_window = 5
    else:  # "1mo" or any other longer period
        days_window = 30

    with st.spinner("Fetching macro & gold news..."):
        all_news = fetch_macro_news(days_window=days_window)
    
    # Split into chart-eligible (macro/gold) vs uncategorized/other
    chart_news = [n for n in all_news if n.get("type") in ("macro", "gold")]
    other_news = [n for n in all_news if n.get("type") not in ("macro", "gold")]

    # Display news count
    window_label = f"last {days_window} day{'s' if days_window > 1 else ''}"
    st.info(
        f"📰 Found {len(chart_news)} macro/gold news items in the {window_label} "
        "(sources: MarketWatch bulletins & top stories, Seeking Alpha economy RSS, "
        "Mining.com gold feed, Yahoo Finance tickers)"
    )
    
    # Create and display chart (only macro/gold items)
    if not df.empty:
        fig = create_gold_chart_with_news(df, chart_news)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Display macro/gold news list
    if chart_news:
        st.markdown("---")
        st.subheader("📰 Macroeconomic & Gold News")
        
        for idx, news in enumerate(chart_news, 1):
            raw_type = (news.get('type') or 'general').capitalize()
            category_label = raw_type if raw_type in ("Macro", "Gold") else ""
            prefix = f"[{category_label}] " if category_label else ""
            expander_label = f"{idx}. {prefix}{news['title']} - {news['timestamp'].strftime('%H:%M')}"
            with st.expander(expander_label):
                st.write(f"**Publisher:** {news['publisher']}")
                st.write(f"**Time:** {news['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                if category_label:
                    st.write(f"**Category:** {category_label}")
                if news['link']:
                    st.write(f"**Link:** {news['link']}")
    else:
        st.warning("⚠️ No macro or gold-specific news found in the selected window. News markers will not appear on the chart.")
        st.info("💡 News is fetched from Yahoo Finance, MarketWatch, Seeking Alpha, and Mining.com. If no news appears, there may not be any relevant updates or sources may be temporarily unavailable.")

    # Show uncategorized/other news separately at the bottom
    if other_news:
        st.markdown("---")
        st.subheader("🗞 Other Related Headlines (no macro/gold tag)")
        for idx, news in enumerate(other_news, 1):
            ts = news.get("timestamp")
            time_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, (datetime, pd.Timestamp)) else ""
            with st.expander(f"{idx}. {news.get('title', 'No title')} - {time_str}"):
                st.write(f"**Publisher:** {news.get('publisher', 'Unknown')}")
                if time_str:
                    st.write(f"**Time:** {time_str}")
                if news.get("source"):
                    st.write(f"**Source feed:** {news['source']}")
                if news.get("link"):
                    st.write(f"**Link:** {news['link']}")

    # Additional categorized news section
    with st.spinner("Expanding news coverage by category and region..."):
        category_news = fetch_categorized_news(limit_per_category=4)
        regional_news = fetch_regional_news(limit_per_region=5)

    separator_shown = False

    if category_news:
        st.markdown("---")
        separator_shown = True
        st.subheader("🌐 More News by Category")
        st.caption("Curated from New York Times RSS feeds. Updated every 10 minutes.")

        category_items = list(category_news.items())
        for chunk in chunk_list(category_items, 3):
            cols = st.columns(len(chunk))
            for col, (category, items) in zip(cols, chunk):
                with col:
                    st.markdown(f"**{category}**")
                    for item in items:
                        title = item["title"]
                        link = item["link"]
                        published = item["published"]
                        if link:
                            st.write(f"- [{title}]({link})")
                        else:
                            st.write(f"- {title}")
                        if published:
                            st.caption(published)
    else:
        st.info("Additional category news is temporarily unavailable.")

    if regional_news:
        if not separator_shown:
            st.markdown("---")
            separator_shown = True
        st.subheader("🌏 Regional Focus: Canada & Vietnam")
        st.caption("Latest headlines sourced from Canadian and Vietnamese outlets.")

        region_order = ["Canada", "Vietnam"]
        for region in region_order:
            articles = regional_news.get(region, [])
            flag = "🇨🇦" if region == "Canada" else "🇻🇳"
            st.markdown(f"**{flag} {region}**")
            if not articles:
                st.caption("No headlines available right now.")
                continue
            for article in articles:
                title = article["title"]
                link = article["link"]
                published = article["published"]
                if link:
                    st.write(f"- [{title}]({link})")
                else:
                    st.write(f"- {title}")
                if published:
                    st.caption(published)
    else:
        st.info("Regional headlines for Canada and Vietnam are temporarily unavailable.")

if __name__ == "__main__":
    main()

