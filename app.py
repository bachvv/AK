import streamlit as st

# Configure navigation menu that routes to the two existing apps.
pages = st.navigation(
    [
        st.Page("stock.py", title="Stocks", icon=":material/trending_up:"),
        st.Page("news.py", title="News", icon="📰"),
    ],
    position="sidebar",
    expanded=True,
)

with st.sidebar:
    st.markdown("## Modules")
    st.caption("Pick a page to open the Stocks or News dashboards.")

# Run whichever page is selected in the navigation menu.
pages.run()
