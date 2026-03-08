"""
US Holdings Page for Centurion Capital LLC.

Integrates with DriveWealth API to:
  1. Authenticate with DriveWealth credentials (client ID / secret / app key)
  2. Link a DriveWealth trading account
  3. Display account summary, cash balances, positions (holdings), and orders
"""

import json
import logging
import os
from pathlib import Path

import streamlit as st

from ui.components import render_header_bar, render_footer, render_navigation_buttons, render_stock_ticker_ribbon, render_vix_indicator

logger = logging.getLogger(__name__)

# Persist linked account credentials to a local JSON file so the user
# doesn't have to re-enter them every session.  The file is gitignored
# and lives next to app.py.
_CREDS_FILE = Path(__file__).resolve().parents[1] / "data" / "dw_credentials.json"


# ── Credential helpers ──────────────────────────────────────────

def _load_saved_creds() -> dict:
    """Return saved DriveWealth credentials (if any)."""
    if _CREDS_FILE.is_file():
        try:
            return json.loads(_CREDS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_creds(client_id: str, client_secret: str, app_key: str, user_id: str, account_id: str):
    _CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CREDS_FILE.write_text(json.dumps({
        "client_id": client_id,
        "client_secret": client_secret,
        "app_key": app_key,
        "user_id": user_id,
        "account_id": account_id,
    }, indent=2))


def _clear_creds():
    if _CREDS_FILE.is_file():
        _CREDS_FILE.unlink()


# ── DriveWealth client singleton (per Streamlit process) ────────

@st.cache_resource(show_spinner=False)
def _get_dw_client(client_id: str, client_secret: str, app_key: str):
    from services.drivewealth import DriveWealthClient
    return DriveWealthClient(client_id, client_secret, app_key)


# ── Main entry point ───────────────────────────────────────────

def render_us_holdings_page():
    """Render the US Holdings page."""
    _user = st.session_state.get("username", "unknown")
    logger.info("[user=%s] Viewing Holdings page", _user)

    render_header_bar(subtitle=" Holdings · DriveWealth")
    render_stock_ticker_ribbon(market="US")
    render_vix_indicator(market="US")
    render_navigation_buttons(current_page="us_holdings", back_key_suffix="from_us_holdings")

    saved = _load_saved_creds()

    # ── 1. Authentication & Account Linking ─────────────────
    if not st.session_state.get("dw_authenticated"):
        _render_auth_form(saved)
        render_footer()
        return

    # If we reach here the user is authenticated
    client_id = st.session_state["dw_client_id"]
    client_secret = st.session_state["dw_client_secret"]
    app_key = st.session_state["dw_app_key"]
    account_id = st.session_state.get("dw_account_id", "")
    user_id = st.session_state.get("dw_user_id", "")

    dw = _get_dw_client(client_id, client_secret, app_key)
    if not dw.is_authenticated:
        try:
            dw.authenticate()
        except Exception as exc:
            st.error(f"Session expired — please re-authenticate: {exc}")
            st.session_state["dw_authenticated"] = False
            st.rerun()

    # ── 2. Account linking (if not linked yet) ──────────────
    if not account_id:
        _render_account_linking(dw, user_id)
        render_footer()
        return

    # ── 3. Disconnect button ────────────────────────────────
    dc1, dc2 = st.columns([6, 1])
    with dc2:
        if st.button(" Disconnect", use_container_width=True, key="dw_disconnect"):
            logger.info("[user=%s] Disconnecting DriveWealth account", _user)
            _clear_creds()
            for k in list(st.session_state.keys()):
                if k.startswith("dw_"):
                    del st.session_state[k]
            st.cache_resource.clear()
            st.rerun()

    # ── 4. Dashboard tabs ───────────────────────────────────
    tab_summary, tab_positions, tab_orders, tab_txns = st.tabs(
        [" Account Summary", " Positions", " Orders", " Transactions"]
    )

    with tab_summary:
        _render_account_summary(dw, account_id)

    with tab_positions:
        _render_positions(dw, account_id)

    with tab_orders:
        _render_orders(dw, account_id)

    with tab_txns:
        _render_transactions(dw, account_id)

    render_footer()


# ── Auth form ───────────────────────────────────────────────────

def _render_auth_form(saved: dict):
    """Show the DriveWealth credential form and authenticate on submit."""

    # Compact styling for the credential form
    st.markdown("""
    <style>
    div[data-testid="stForm"] {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #0f3460 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.5rem 2rem 1rem 2rem;
    }
    div[data-testid="stForm"] label {
        font-size: 0.78rem !important;
        color: #8b949e !important;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        font-weight: 600;
    }
    div[data-testid="stForm"] input {
        font-size: 0.85rem !important;
        padding: 0.35rem 0.6rem !important;
        height: 2.1rem !important;
        border-radius: 6px !important;
    }
    div[data-testid="stForm"] .stTextInput {
        margin-bottom: -0.5rem;
    }
    div[data-testid="stForm"] [data-testid="stCheckbox"] {
        margin-top: 0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p style="color:#8b949e;font-size:0.82rem;margin-bottom:0.6rem;">'
        'Enter your <b>DriveWealth API credentials</b> to link your US stock trading account. '
        'Credentials are stored locally and only sent to DriveWealth\'s API.</p>',
        unsafe_allow_html=True,
    )

    with st.form("dw_auth_form"):
        st.markdown(
            '<p style="font-size:0.92rem;font-weight:700;margin-bottom:0.3rem;">'
            'Connect DriveWealth Account</p>',
            unsafe_allow_html=True,
        )
        r1c1, r1c2, r1c3 = st.columns(3)
        client_id = r1c1.text_input(
            "Client ID",
            value=saved.get("client_id", os.getenv("DW_CLIENT_ID", "")),
            type="password",
        )
        client_secret = r1c2.text_input(
            "Client Secret",
            value=saved.get("client_secret", os.getenv("DW_CLIENT_SECRET", "")),
            type="password",
        )
        app_key = r1c3.text_input(
            "App Key",
            value=saved.get("app_key", os.getenv("DW_APP_KEY", "")),
            type="password",
        )
        r2c1, r2c2 = st.columns(2)
        user_id = r2c1.text_input(
            "User ID (optional)",
            value=saved.get("user_id", os.getenv("DW_USER_ID", "")),
        )
        account_id = r2c2.text_input(
            "Account ID (optional)",
            value=saved.get("account_id", os.getenv("DW_ACCOUNT_ID", "")),
        )
        fc1, fc2 = st.columns([3, 1])
        remember = fc1.checkbox("Save credentials locally", value=bool(saved))
        submitted = fc2.form_submit_button("Connect", use_container_width=True, type="primary")

    if submitted:
        if not client_id or not client_secret or not app_key:
            st.error("Client ID, Client Secret, and App Key are required.")
            return

        with st.spinner("Authenticating with DriveWealth…"):
            try:
                dw = _get_dw_client(client_id, client_secret, app_key)
                auth_data = dw.authenticate()
            except Exception as exc:
                st.error(f"Authentication failed: {exc}")
                return

        st.success("Authenticated successfully!")
        st.session_state["dw_authenticated"] = True
        st.session_state["dw_client_id"] = client_id
        st.session_state["dw_client_secret"] = client_secret
        st.session_state["dw_app_key"] = app_key
        st.session_state["dw_user_id"] = user_id.strip()
        st.session_state["dw_account_id"] = account_id.strip()

        if remember:
            _save_creds(client_id, client_secret, app_key, user_id.strip(), account_id.strip())

        logger.info("[user=%s] DriveWealth auth succeeded — scope=%s",
                    st.session_state.get("username", "unknown"),
                    auth_data.get("scope", ""))
        st.rerun()


# ── Account linking ─────────────────────────────────────────────

def _render_account_linking(dw, user_id: str):
    """Let the user pick from their DriveWealth accounts or enter an ID."""
    st.markdown("### Link a Trading Account")

    accounts = []
    if user_id:
        try:
            resp = dw.list_user_accounts(user_id)
            accounts = resp if isinstance(resp, list) else resp.get("accounts", resp.get("data", []))
        except Exception as exc:
            st.warning(f"Could not fetch accounts for this User ID: {exc}")

    if accounts:
        st.markdown("Select an account to link:")
        for acc in accounts:
            acc_id = acc.get("id", "")
            acc_no = acc.get("accountNo", acc_id)
            nickname = acc.get("nickname", "")
            status = acc.get("status", {})
            status_name = status.get("name", "") if isinstance(status, dict) else status
            label = f"**{acc_no}** — {nickname}" if nickname else f"**{acc_no}**"
            if status_name:
                label += f"  `{status_name}`"

            if st.button(f"Link {acc_no}", key=f"link_{acc_id}", use_container_width=True):
                st.session_state["dw_account_id"] = acc_id
                saved = _load_saved_creds()
                if saved:
                    _save_creds(
                        saved["client_id"], saved["client_secret"],
                        saved["app_key"], user_id, acc_id,
                    )
                logger.info("[user=%s] Linked DW account %s", st.session_state.get("username"), acc_id)
                st.rerun()
            st.caption(label)
    else:
        st.info("Enter your **Account ID** manually to link your trading account.")

    with st.form("dw_manual_link"):
        manual_id = st.text_input("Account ID")
        if st.form_submit_button("Link Account", type="primary"):
            if manual_id.strip():
                st.session_state["dw_account_id"] = manual_id.strip()
                saved = _load_saved_creds()
                if saved:
                    _save_creds(
                        saved["client_id"], saved["client_secret"],
                        saved["app_key"], user_id, manual_id.strip(),
                    )
                st.rerun()
            else:
                st.error("Please enter a valid Account ID.")


# ── Dashboard panels ────────────────────────────────────────────

def _render_account_summary(dw, account_id: str):
    """Account info + cash summary."""
    try:
        acc_resp = dw.get_account(account_id)
        acc = acc_resp.get("account", acc_resp)
    except Exception as exc:
        st.error(f"Failed to load account: {exc}")
        return

    # Header metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Account No", acc.get("accountNo", "—"))
    status = acc.get("status", {})
    c2.metric("Status", status.get("name", "—") if isinstance(status, dict) else str(status))
    trading = acc.get("tradingType", {})
    c3.metric("Trading Type", trading.get("name", "—") if isinstance(trading, dict) else str(trading))
    c4.metric("Leverage", f"{acc.get('leverage', 1)}x")

    # Cash balances
    st.markdown("#### Cash Balances")
    try:
        cash = dw.get_account_cash(account_id)
        cash_data = cash.get("cash", cash)
        if isinstance(cash_data, dict):
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Cash Balance", f"${cash_data.get('cashBalance', 0):,.2f}")
            bc2.metric("Available for Trading", f"${cash_data.get('cashAvailableForTrading', 0):,.2f}")
            bc3.metric("Available for Withdrawal", f"${cash_data.get('cashAvailableForWithdrawal', 0):,.2f}")
            bc4.metric("Equity Value", f"${cash_data.get('equityValue', 0):,.2f}")
        else:
            st.json(cash)
    except Exception as exc:
        st.warning(f"Could not fetch cash summary: {exc}")

    # BOD snapshot from account response
    bod = acc.get("bod")
    if bod and isinstance(bod, dict):
        st.markdown("#### Beginning of Day")
        bd1, bd2, bd3 = st.columns(3)
        bd1.metric("Cash Balance", f"${bod.get('cashBalance', 0):,.2f}")
        bd2.metric("Equity Value", f"${bod.get('equityValue', 0):,.2f}")
        bd3.metric("Money Market", f"${bod.get('moneyMarket', 0):,.2f}")

    # Raw account summary (collapsible)
    with st.expander("Raw Account Data"):
        st.json(acc)


def _render_positions(dw, account_id: str):
    """Positions / holdings table."""
    try:
        resp = dw.list_positions(account_id)
    except Exception as exc:
        st.error(f"Failed to load positions: {exc}")
        return

    positions = resp if isinstance(resp, list) else resp.get("equityPositions", resp.get("positions", []))

    if not positions:
        st.info(" No open positions in this account.")
        return

    import pandas as pd
    rows = []
    for p in positions:
        rows.append({
            "Symbol": p.get("symbol", p.get("instrumentID", "—")),
            "Qty": p.get("openQty", p.get("quantity", 0)),
            "Avg Price": p.get("avgPrice", p.get("averagePrice", 0)),
            "Mkt Value": p.get("marketValue", 0),
            "Unrealized P&L": p.get("unrealizedPL", p.get("unrealizedDayPL", 0)),
            "Side": p.get("side", "—"),
        })

    df = pd.DataFrame(rows)

    # Summary metrics
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Positions", len(df))
    total_value = df["Mkt Value"].sum() if "Mkt Value" in df.columns else 0
    mc2.metric("Total Mkt Value", f"${total_value:,.2f}")
    total_pnl = df["Unrealized P&L"].sum() if "Unrealized P&L" in df.columns else 0
    pnl_color = "normal" if total_pnl >= 0 else "inverse"
    mc3.metric("Total Unrealized P&L", f"${total_pnl:+,.2f}", delta_color=pnl_color)

    # Styled table
    def _style_pnl(val):
        try:
            v = float(val)
            if v > 0:
                return "color:#38a169;font-weight:600"
            if v < 0:
                return "color:#e53e3e;font-weight:600"
        except (ValueError, TypeError):
            pass
        return ""

    styled = df.style.map(_style_pnl, subset=["Unrealized P&L"])
    styled = styled.format({
        "Avg Price": "${:,.2f}",
        "Mkt Value": "${:,.2f}",
        "Unrealized P&L": "${:+,.2f}",
    }, na_rep="—")
    st.dataframe(styled, hide_index=True, use_container_width=True)


def _render_orders(dw, account_id: str):
    """Open / resting orders."""
    try:
        resp = dw.list_resting_orders(account_id)
    except Exception as exc:
        st.error(f"Failed to load orders: {exc}")
        return

    orders = resp if isinstance(resp, list) else resp.get("orders", [])
    if not orders:
        st.info(" No resting orders.")
        return

    import pandas as pd
    rows = []
    for o in orders:
        rows.append({
            "Order ID": o.get("orderID", o.get("id", "—")),
            "Symbol": o.get("symbol", "—"),
            "Side": o.get("side", "—"),
            "Qty": o.get("orderQty", o.get("quantity", 0)),
            "Price": o.get("price", 0),
            "Type": o.get("type", o.get("orderType", "—")),
            "Status": o.get("status", "—"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)


def _render_transactions(dw, account_id: str):
    """Recent transactions."""
    try:
        resp = dw.list_transactions(account_id)
    except Exception as exc:
        st.error(f"Failed to load transactions: {exc}")
        return

    txns = resp if isinstance(resp, list) else resp.get("transactions", [])
    if not txns:
        st.info(" No transactions found.")
        return

    import pandas as pd
    rows = []
    for t in txns:
        rows.append({
            "Date": t.get("tranWhen", t.get("createdWhen", "—")),
            "Type": t.get("tranType", t.get("type", "—")),
            "Symbol": t.get("instrument", {}).get("symbol", t.get("symbol", "—")) if isinstance(t.get("instrument"), dict) else t.get("symbol", "—"),
            "Qty": t.get("qty", t.get("quantity", "—")),
            "Price": t.get("fillPx", t.get("price", "—")),
            "Amount": t.get("tranAmount", t.get("amount", "—")),
            "Commission": t.get("commission", "—"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)
