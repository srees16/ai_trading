"""
Options & Derivatives Page Module for Centurion Capital LLC.

Fetches derivatives data from Sensibull, displays live option chain prices,
quotes, and processed derivative analytics for Indian markets.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from ui.components import (
    render_page_header,
    render_footer,
    render_ind_navigation_buttons,
    render_stock_ticker_ribbon,
    render_vix_indicator,
)

logger = logging.getLogger(__name__)

# ── Popular indices / underlyings ────────────────────────────────
_DEFAULT_INDICES = {
    "NIFTY":      260105,
    "BANKNIFTY":  260361,
    "FINNIFTY":   257801,
    "MIDCPNIFTY": 288009,
    "SENSEX":     265,
}

_DEFAULT_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]


# ── Helpers ──────────────────────────────────────────────────────

def _get_quotes_client():
    """Return a cached SensibullQuotes instance (one per session)."""
    if "sensibull_quotes" not in st.session_state:
        from sensibull_quotes import SensibullQuotes
        st.session_state["sensibull_quotes"] = SensibullQuotes()
    return st.session_state["sensibull_quotes"]


def _safe_fetch_derivatives(quotes) -> Optional[Any]:
    """Fetch derivatives data with error handling."""
    try:
        quotes.fetch_derivatives_data()
        return True
    except Exception as exc:
        logger.warning("Failed to fetch derivatives data: %s", exc)
        return None


def _safe_get_quotes(quotes, symbols) -> Dict:
    """Get quotes for one or multiple symbols safely."""
    try:
        return quotes.get_quotes(symbols) or {}
    except Exception as exc:
        logger.warning("Failed to get quotes for %s: %s", symbols, exc)
        return {}


def _safe_get_live_prices(quotes, token: int) -> Optional[Dict]:
    """Get live derivative prices for an instrument token safely."""
    try:
        return quotes.get_live_derivative_prices(token)
    except Exception as exc:
        logger.warning("Failed to get live prices for token %d: %s", token, exc)
        return None


def _build_options_dataframe(expiry_data: Dict) -> pd.DataFrame:
    """Convert a single expiry's options list into a clean DataFrame."""
    options = expiry_data.get("options", [])
    if not options:
        return pd.DataFrame()

    df = pd.DataFrame(options)

    # Identify option type from trading symbol (CE / PE suffix)
    if "tradingsymbol" in df.columns:
        df["type"] = df["tradingsymbol"].apply(
            lambda s: "CE" if str(s).endswith("CE") else ("PE" if str(s).endswith("PE") else "—")
        )

    # Reorder and rename columns for readability
    col_map = {
        "token": "Token",
        "tradingsymbol": "Trading Symbol",
        "type": "Type",
        "strike": "Strike",
        "last_price": "LTP",
        "bid": "Bid",
        "ask": "Ask",
        "oi": "OI",
        "volume": "Volume",
        "iv": "IV (%)",
        "delta": "Delta",
        "gamma": "Gamma",
        "theta": "Theta",
        "vega": "Vega",
    }

    present = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=present)
    ordered = [v for v in col_map.values() if v in df.columns]
    # Include any remaining columns we didn't explicitly map
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    # Numeric formatting
    for col in ("LTP", "Bid", "Ask", "IV (%)", "Delta", "Gamma", "Theta", "Vega"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("OI", "Volume", "Token", "Strike"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


# ── Page renderer ────────────────────────────────────────────────

def render_options_page():
    """Render the Options & Derivatives page."""
    logger.info(
        "[user=%s] Viewing Options & Derivatives page",
        st.session_state.get("username", "unknown"),
    )

    render_page_header(
        "Options & Derivatives",
        subtitle="Live Derivative Prices · Option Chains · Index Quotes",
    )

    render_stock_ticker_ribbon(market="IND")

    render_vix_indicator(market="IND")

    render_ind_navigation_buttons(
        current_page="options",
        back_key_suffix="from_options",
    )

    quotes = _get_quotes_client()

    # Tabs for different sections
    tab_quotes, tab_live, tab_chain, tab_lookup = st.tabs([
        " Index Quotes",
        " Live Derivative Prices",
        " Option Chain",
        " Token Lookup",
    ])

    with tab_quotes:
        _render_index_quotes(quotes)

    with tab_live:
        _render_live_derivative_prices(quotes)

    with tab_chain:
        _render_option_chain(quotes)

    with tab_lookup:
        _render_token_lookup(quotes)

    render_footer()


# ── Tab 1: Index Quotes ─────────────────────────────────────────

def _render_index_quotes(quotes):
    """Fetch and display quotes for major indices."""
    st.markdown("##### Index Quotes")
    st.caption("Real-time quotes for major Indian derivatives indices.")

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected = st.multiselect(
            "Select indices",
            options=_DEFAULT_SYMBOLS,
            default=_DEFAULT_SYMBOLS[:3],
            key="opt_index_sel",
        )
    with col_btn:
        st.markdown('<div style="margin-top:1.6rem;"></div>', unsafe_allow_html=True)
        refresh = st.button("Refresh", key="opt_quotes_refresh")

    if not selected:
        st.info("Select at least one index above.")
        return

    if refresh or "opt_quotes_data" not in st.session_state:
        with st.spinner("Fetching quotes…"):
            data = _safe_get_quotes(quotes, selected if len(selected) > 1 else selected[0])
            # Normalise: single-symbol returns a dict of fields, not {symbol: fields}
            if selected and len(selected) == 1 and not isinstance(data, dict):
                data = {selected[0]: data}
            elif selected and len(selected) == 1 and isinstance(data, dict) and selected[0] not in data:
                data = {selected[0]: data}
            st.session_state["opt_quotes_data"] = data

    data = st.session_state.get("opt_quotes_data", {})
    if not data:
        st.warning("No quote data returned. The market may be closed.")
        return

    # Display as metric cards
    cols = st.columns(min(len(data), 4))
    for idx, (sym, qdata) in enumerate(data.items()):
        with cols[idx % len(cols)]:
            if isinstance(qdata, dict):
                ltp = qdata.get("last_price") or qdata.get("ltp") or qdata.get("close") or "—"
                change = qdata.get("change") or qdata.get("net_change")
                pct = qdata.get("change_pct") or qdata.get("percent_change")
                delta_str = None
                if pct is not None:
                    delta_str = f"{float(pct):+.2f}%"
                elif change is not None:
                    delta_str = f"{float(change):+.2f}"
                st.metric(label=sym, value=ltp, delta=delta_str)
            else:
                st.metric(label=sym, value=str(qdata))

    # Raw data expander
    with st.expander("View raw quote data"):
        st.json(data)


# ── Tab 2: Live Derivative Prices ────────────────────────────────

def _render_live_derivative_prices(quotes):
    """Fetch and display live derivative prices for a selected index."""
    st.markdown("##### Live Derivative Prices")
    st.caption(
        "Select an index to view live derivative prices across all expiries — "
        "including ATM strike, IV, futures price, and max OI."
    )

    col_idx, col_tok, col_btn = st.columns([2, 2, 1])
    with col_idx:
        index_name = st.selectbox(
            "Index",
            list(_DEFAULT_INDICES.keys()),
            key="opt_live_index",
        )
    with col_tok:
        default_token = _DEFAULT_INDICES.get(index_name, 260105)
        token = st.number_input(
            "Instrument token",
            value=default_token,
            step=1,
            key="opt_live_token",
            help="Auto-filled from index selection. Override for custom instruments.",
        )
    with col_btn:
        st.markdown('<div style="margin-top:1.6rem;"></div>', unsafe_allow_html=True)
        fetch = st.button("Fetch Live", key="opt_live_fetch", type="primary")

    if not fetch and "opt_live_data" not in st.session_state:
        st.info("Click **Fetch Live** to load derivative prices.")
        return

    if fetch:
        with st.spinner(f"Fetching live prices for {index_name} (token {token})…"):
            live = _safe_get_live_prices(quotes, int(token))
            st.session_state["opt_live_data"] = live
            st.session_state["opt_live_label"] = index_name

    live = st.session_state.get("opt_live_data")
    label = st.session_state.get("opt_live_label", index_name)

    if not live:
        st.warning("No live data returned. The market may be closed or the token may be invalid.")
        return

    # ── Header metrics
    underlying = live.get("underlying_price", "—")
    updated = live.get("last_updated_at", "—")
    m1, m2 = st.columns(2)
    m1.metric(f"{label} Underlying", underlying)
    m2.caption(f"Last updated: {updated}")

    # ── Per-expiry data
    expiries = live.get("expiries", [])
    if not expiries:
        st.info("No expiry data available.")
        return

    for exp_data in expiries:
        expiry_label = exp_data.get("expiry", "Unknown")
        with st.expander(f"Expiry: {expiry_label}", expanded=(exp_data == expiries[0])):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ATM Strike", exp_data.get("atm_strike", "—"))
            c2.metric("ATM IV", f"{exp_data.get('atm_iv', '—')}%"
                      if exp_data.get("atm_iv") else "—")
            c3.metric("Future Price", exp_data.get("future_price", "—"))
            c4.metric("Max OI", f"{exp_data.get('max_oi', 0):,}")

            # Build options table
            df = _build_options_dataframe(exp_data)
            if df.empty:
                st.info("No options data for this expiry.")
                continue

            # Split CE / PE for side-by-side display
            if "Type" in df.columns:
                ce_df = df[df["Type"] == "CE"].drop(columns=["Type"], errors="ignore")
                pe_df = df[df["Type"] == "PE"].drop(columns=["Type"], errors="ignore")

                left, right = st.columns(2)
                with left:
                    st.markdown(" **Calls (CE)**")
                    st.dataframe(
                        ce_df, width="stretch", hide_index=True,
                        height=min(35 * len(ce_df) + 38, 400),
                    )
                with right:
                    st.markdown(" **Puts (PE)**")
                    st.dataframe(
                        pe_df, width="stretch", hide_index=True,
                        height=min(35 * len(pe_df) + 38, 400),
                    )
            else:
                st.dataframe(df, width="stretch", hide_index=True)


# ── Tab 3: Full Option Chain ─────────────────────────────────────

def _render_option_chain(quotes):
    """Display a consolidated option chain table for a chosen index."""
    st.markdown("##### Option Chain")
    st.caption(
        "Select an index and an expiry to load the full option chain "
        "with CE and PE data side by side."
    )

    col_idx, col_btn = st.columns([3, 1])
    with col_idx:
        index_name = st.selectbox(
            "Index",
            list(_DEFAULT_INDICES.keys()),
            key="opt_chain_index",
        )
    with col_btn:
        st.markdown('<div style="margin-top:1.6rem;"></div>', unsafe_allow_html=True)
        load = st.button("Load Chain", key="opt_chain_load", type="primary")

    token = _DEFAULT_INDICES.get(index_name, 260105)

    if not load and "opt_chain_data" not in st.session_state:
        st.info("Click **Load Chain** to fetch the option chain.")
        return

    if load:
        with st.spinner(f"Loading option chain for {index_name}…"):
            live = _safe_get_live_prices(quotes, int(token))
            st.session_state["opt_chain_data"] = live
            st.session_state["opt_chain_label"] = index_name

    live = st.session_state.get("opt_chain_data")
    label = st.session_state.get("opt_chain_label", index_name)

    if not live:
        st.warning("No data returned.")
        return

    expiries = live.get("expiries", [])
    if not expiries:
        st.info("No expiry data available.")
        return

    # Expiry selector
    expiry_labels = [e.get("expiry", "?") for e in expiries]
    chosen_expiry = st.selectbox(
        "Expiry",
        expiry_labels,
        key="opt_chain_expiry",
    )

    exp_data = next((e for e in expiries if e.get("expiry") == chosen_expiry), expiries[0])

    # Header metrics row
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Spot", live.get("underlying_price", "—"))
    mc2.metric("ATM Strike", exp_data.get("atm_strike", "—"))
    mc3.metric("ATM IV", f"{exp_data.get('atm_iv', '—')}%"
               if exp_data.get("atm_iv") else "—")
    mc4.metric("Fut Price", exp_data.get("future_price", "—"))

    # Build chain as merged CE + PE on strike
    options = exp_data.get("options", [])
    if not options:
        st.info("No options data for this expiry.")
        return

    df = pd.DataFrame(options)
    if "tradingsymbol" in df.columns:
        df["type"] = df["tradingsymbol"].apply(
            lambda s: "CE" if str(s).endswith("CE") else ("PE" if str(s).endswith("PE") else "—")
        )

    if "strike" in df.columns and "type" in df.columns:
        ce = df[df["type"] == "CE"].copy()
        pe = df[df["type"] == "PE"].copy()

        value_cols = ["last_price", "oi", "volume", "iv", "bid", "ask"]
        present_cols = [c for c in value_cols if c in df.columns]

        ce_rename = {c: f"CE {c.replace('_', ' ').title()}" for c in present_cols}
        ce_rename["tradingsymbol"] = "CE Symbol"
        pe_rename = {c: f"PE {c.replace('_', ' ').title()}" for c in present_cols}
        pe_rename["tradingsymbol"] = "PE Symbol"

        ce = ce.rename(columns=ce_rename)
        pe = pe.rename(columns=pe_rename)

        ce_cols = ["strike"] + [v for v in ce_rename.values() if v in ce.columns]
        pe_cols = ["strike"] + [v for v in pe_rename.values() if v in pe.columns]

        merged = pd.merge(
            ce[ce_cols], pe[pe_cols],
            on="strike", how="outer",
        ).sort_values("strike").reset_index(drop=True)
        merged = merged.rename(columns={"strike": "Strike"})

        # Numeric cleanup
        for col in merged.columns:
            if col != "Strike" and "Symbol" not in col:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

        # Highlight ATM row
        atm = exp_data.get("atm_strike")

        st.dataframe(
            merged,
            width="stretch",
            hide_index=True,
            height=min(35 * len(merged) + 38, 600),
        )
    else:
        st.dataframe(pd.DataFrame(options), width="stretch", hide_index=True)

    # Download button
    csv = merged.to_csv(index=False) if "merged" in dir() else pd.DataFrame(options).to_csv(index=False)
    st.download_button(
        " Download CSV",
        data=csv,
        file_name=f"{label}_{chosen_expiry}_chain.csv",
        mime="text/csv",
        key="opt_chain_csv",
    )


# ── Tab 4: Token Lookup ─────────────────────────────────────────

def _render_token_lookup(quotes):
    """Look up a trading symbol from an instrument token."""
    st.markdown("##### Instrument Token Lookup")
    st.caption(
        "Enter an instrument token to find the corresponding trading symbol, "
        "or search derivatives data by symbol prefix."
    )

    mode = st.radio(
        "Lookup mode",
        ["Token Symbol", "Fetch Derivatives Data"],
        horizontal=True,
        key="opt_lookup_mode",
    )

    if mode == "Token Symbol":
        token_input = st.number_input(
            "Instrument token",
            value=9073154,
            step=1,
            key="opt_lookup_token",
        )
        if st.button("Lookup", key="opt_lookup_btn"):
            with st.spinner("Resolving…"):
                try:
                    symbol = quotes.get_tradingsymbol(int(token_input))
                    if symbol:
                        st.success(f"**Token {token_input}** `{symbol}`")
                    else:
                        st.warning(f"No trading symbol found for token {token_input}.")
                except Exception as exc:
                    st.error(f"Lookup failed: {exc}")

    else:
        st.markdown("Refresh the full derivatives instrument list from Sensibull.")
        if st.button(" Fetch Derivatives Data", key="opt_fetch_deriv"):
            with st.spinner("Fetching derivatives data…"):
                result = _safe_fetch_derivatives(quotes)
                if result:
                    st.success("Derivatives data fetched and cached successfully.")
                else:
                    st.error("Failed to fetch derivatives data. Try again later.")
