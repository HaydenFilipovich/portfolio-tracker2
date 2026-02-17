from dotenv import load_dotenv
import os

load_dotenv()
import io
import os
import sqlite3
from openai import OpenAI
from pathlib import Path

import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(page_title="Portfolio Tracker", layout="wide")

# --- Database ---

DB_PATH = Path(__file__).parent / "portfolio.db"


def _get_conn() -> sqlite3.Connection:
    """Return a connection to the SQLite database (one per call)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create the holdings table if it doesn't exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS holdings (
                ticker    TEXT PRIMARY KEY,
                shares    REAL NOT NULL,
                avg_cost  REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)


def load_holdings() -> list[dict]:
    """Read all holdings from the DB into a list of dicts."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT ticker, shares, avg_cost FROM holdings ORDER BY ticker"
        ).fetchall()
    return [
        {"ticker": r[0], "shares": r[1], "cost_basis": r[2]}
        for r in rows
    ]


def upsert_holding(ticker: str, shares: float, cost: float) -> None:
    """Insert or merge a holding using weighted-average cost."""
    now = pd.Timestamp.now().isoformat()
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT shares, avg_cost FROM holdings WHERE ticker = ?", (ticker,)
        ).fetchone()
        if existing:
            old_shares, old_cost = existing
            new_shares = old_shares + shares
            new_avg = (old_shares * old_cost + shares * cost) / new_shares
            conn.execute(
                "UPDATE holdings SET shares = ?, avg_cost = ?, updated_at = ? WHERE ticker = ?",
                (new_shares, new_avg, now, ticker),
            )
        else:
            conn.execute(
                "INSERT INTO holdings (ticker, shares, avg_cost, updated_at) VALUES (?, ?, ?, ?)",
                (ticker, shares, cost, now),
            )


def remove_holding(ticker: str) -> None:
    """Delete a ticker from the DB."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))


init_db()

# --- Cached helpers ---

@st.cache_data(ttl=300, show_spinner=False)
def get_current_price(ticker: str) -> float | None:
    """Return the last price for a ticker, or None on failure."""
    try:
        return yf.Ticker(ticker).fast_info.last_price
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close prices for a ticker in a date range."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float)
    return df["Close"].squeeze()


def peak_to_trough(prices: pd.Series) -> float:
    """Return the max drawdown (negative float) from a price series."""
    if prices.empty or len(prices) < 2:
        return 0.0
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return float(drawdown.min())


STRESS_SCENARIOS = {
    "2008 Financial Crisis": ("2007-10-01", "2009-03-31"),
    "COVID Crash (Mar 2020)": ("2020-02-01", "2020-04-30"),
    "Dot-com Bust (2000-02)": ("2000-03-01", "2002-10-31"),
    "Black Monday (1987)":    ("1987-09-01", "1987-12-31"),
    "2022 Bear Market":       ("2022-01-01", "2022-10-31"),
}


st.title("Portfolio Tracker")

# Initialize holdings in session state from DB
if "holdings" not in st.session_state:
    st.session_state.holdings = load_holdings()

# --- Sidebar: Add Holdings ---
with st.sidebar:
    st.header("Add Holding")
    with st.form("add_holding", clear_on_submit=True):
        ticker = st.text_input("Ticker (e.g. AAPL)", "").upper().strip()
        shares = st.number_input("Shares", min_value=0.0, step=1.0, format="%.4f")
        cost_basis = st.number_input("Cost per share ($)", min_value=0.0, step=0.01, format="%.2f")
        submitted = st.form_submit_button("Add")

        if submitted and ticker and shares > 0:
            upsert_holding(ticker, shares, cost_basis)
            st.session_state.holdings = load_holdings()
            st.success(f"Added {shares} shares of {ticker}")

    # Remove holdings
    if st.session_state.holdings:
        st.divider()
        st.subheader("Remove Holding")
        tickers_in_portfolio = [h["ticker"] for h in st.session_state.holdings]
        remove_ticker = st.selectbox("Select ticker to remove", tickers_in_portfolio)
        if st.button("Remove"):
            remove_holding(remove_ticker)
            st.session_state.holdings = load_holdings()
            st.rerun()

    # --- CSV Import ---
    st.divider()
    st.subheader("Import from CSV")

    TEMPLATE_CSV = "ticker,shares,cost_basis\nAAPL,10,150.00\nMSFT,5,300.00\nGOOGL,3,140.00\n"
    st.download_button(
        "Download template CSV",
        data=TEMPLATE_CSV,
        file_name="portfolio_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"], key="csv_upload")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not parse CSV: {exc}")
            raw = None

        if raw is not None:
            # Normalise column names to lowercase/stripped
            raw.columns = [c.strip().lower() for c in raw.columns]

            REQUIRED = {"ticker", "shares", "cost_basis"}
            missing = REQUIRED - set(raw.columns)
            if missing:
                st.error(
                    f"Missing required column(s): **{', '.join(sorted(missing))}**. "
                    "Expected: ticker, shares, cost_basis."
                )
            else:
                errors: list[str] = []
                valid_rows: list[dict] = []

                for idx, row in raw.iterrows():
                    row_num = idx + 2  # 1-indexed + header

                    tkr = str(row["ticker"]).strip().upper()
                    if not tkr or tkr == "NAN":
                        errors.append(f"Row {row_num}: missing ticker")
                        continue

                    try:
                        shares_val = float(row["shares"])
                    except (ValueError, TypeError):
                        errors.append(f"Row {row_num} ({tkr}): shares is not a number")
                        continue
                    if shares_val <= 0:
                        errors.append(f"Row {row_num} ({tkr}): shares must be > 0")
                        continue

                    try:
                        cb_val = float(row["cost_basis"])
                    except (ValueError, TypeError):
                        errors.append(f"Row {row_num} ({tkr}): cost_basis is not a number")
                        continue
                    if cb_val < 0:
                        errors.append(f"Row {row_num} ({tkr}): cost_basis must be >= 0")
                        continue

                    valid_rows.append({
                        "ticker": tkr,
                        "shares": shares_val,
                        "cost_basis": cb_val,
                    })

                # Detect exact duplicate rows in the CSV
                seen: set[tuple] = set()
                deduped: list[dict] = []
                dup_count = 0
                for r in valid_rows:
                    key = (r["ticker"], r["shares"], r["cost_basis"])
                    if key in seen:
                        dup_count += 1
                    else:
                        seen.add(key)
                        deduped.append(r)

                if dup_count:
                    st.warning(f"Removed {dup_count} exact duplicate row(s) from CSV.")

                if errors:
                    with st.expander(f"{len(errors)} row error(s) — click to view"):
                        for e in errors:
                            st.write(f"- {e}")

                if deduped:
                    # Check for tickers already in portfolio
                    existing = {h["ticker"] for h in st.session_state.holdings}
                    overlap = {r["ticker"] for r in deduped} & existing
                    if overlap:
                        st.info(
                            f"Tickers already in portfolio (will be added as new lots): "
                            f"**{', '.join(sorted(overlap))}**"
                        )

                    st.write(f"**{len(deduped)}** valid holding(s) ready to import.")
                    if st.button("Import", key="csv_import_btn"):
                        for r in deduped:
                            upsert_holding(r["ticker"], r["shares"], r["cost_basis"])
                        st.session_state.holdings = load_holdings()
                        st.success(f"Imported {len(deduped)} holding(s).")
                        st.rerun()
                elif not errors:
                    st.warning("CSV has no valid rows to import.")

# --- Main Area: Portfolio Table ---
if not st.session_state.holdings:
    st.info("Add holdings using the sidebar to get started.")
else:
    # Fetch current prices
    tickers = list({h["ticker"] for h in st.session_state.holdings})

    with st.spinner("Fetching prices..."):
        price_data = {t: get_current_price(t) for t in tickers}

    # Build display table
    rows = []
    for h in st.session_state.holdings:
        price = price_data.get(h["ticker"])
        if price is not None:
            market_value = price * h["shares"]
            cost_total = h["cost_basis"] * h["shares"]
            gain_loss = market_value - cost_total
            gain_loss_pct = (gain_loss / cost_total * 100) if cost_total > 0 else 0.0
        else:
            market_value = None
            gain_loss = None
            gain_loss_pct = None

        rows.append({
            "Ticker": h["ticker"],
            "Shares": h["shares"],
            "Cost Basis": h["cost_basis"],
            "Current Price": price,
            "Market Value": market_value,
            "Gain/Loss ($)": gain_loss,
            "Gain/Loss (%)": gain_loss_pct,
        })

    df = pd.DataFrame(rows)

    # Portfolio summary
    total_value = df["Market Value"].sum()
    total_cost = (df["Cost Basis"] * df["Shares"]).sum()
    total_gain = total_value - total_cost if total_value else 0
    total_gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Portfolio Value", f"${total_value:,.2f}" if total_value else "—")
    col2.metric("Total Gain/Loss", f"${total_gain:,.2f}", f"{total_gain_pct:+.2f}%")
    col3.metric("Positions", len(st.session_state.holdings))

    st.divider()

    # Format the dataframe for display
    st.dataframe(
        df.style.format({
            "Shares": "{:.4f}",
            "Cost Basis": "${:.2f}",
            "Current Price": "${:.2f}",
            "Market Value": "${:,.2f}",
            "Gain/Loss ($)": "${:+,.2f}",
            "Gain/Loss (%)": "{:+.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # --- What If: Scenario & Stress Testing ---
    st.divider()
    st.header("What If — Scenario Testing")

    scenario_tab, stress_tab = st.tabs(["Custom Scenario", "Stress Tests"])

    with scenario_tab:
        st.subheader("Uniform Market Move")
        pct_change = st.slider(
            "Simulate a market-wide move (%)",
            min_value=-50, max_value=50, value=0, step=1,
            format="%d%%",
        )

        st.subheader("Per-Ticker Overrides")
        st.caption("Set a custom % change for individual tickers (overrides the uniform move).")
        ticker_overrides = {}
        override_cols = st.columns(min(len(tickers), 4))
        for i, t in enumerate(tickers):
            with override_cols[i % len(override_cols)]:
                val = st.number_input(
                    f"{t} (%)", min_value=-100.0, max_value=200.0,
                    value=float(pct_change), step=1.0, key=f"override_{t}",
                )
                ticker_overrides[t] = val

        # Calculate scenario
        scenario_rows = []
        for h in st.session_state.holdings:
            price = price_data.get(h["ticker"])
            if price is None:
                continue
            move = ticker_overrides.get(h["ticker"], pct_change)
            new_price = price * (1 + move / 100)
            current_val = price * h["shares"]
            new_val = new_price * h["shares"]
            scenario_rows.append({
                "Ticker": h["ticker"],
                "Current Price": price,
                "Scenario Price": new_price,
                "Current Value": current_val,
                "Scenario Value": new_val,
                "Change ($)": new_val - current_val,
            })

        scenario_df = pd.DataFrame(scenario_rows)
        if not scenario_df.empty:
            new_total = scenario_df["Scenario Value"].sum()
            change_total = scenario_df["Change ($)"].sum()
            change_pct = (change_total / total_value * 100) if total_value > 0 else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Current Value", f"${total_value:,.2f}")
            c2.metric("Scenario Value", f"${new_total:,.2f}")
            c3.metric("Impact", f"${change_total:+,.2f}", f"{change_pct:+.2f}%")

            st.dataframe(
                scenario_df.style.format({
                    "Current Price": "${:.2f}",
                    "Scenario Price": "${:.2f}",
                    "Current Value": "${:,.2f}",
                    "Scenario Value": "${:,.2f}",
                    "Change ($)": "${:+,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    with stress_tab:
        st.subheader("Historical Stress Scenarios")
        st.caption(
            "Per-ticker drawdowns are computed from actual historical prices "
            "during each crisis window."
        )

        # Build a {ticker: current_value} lookup for weighting
        ticker_values = {}
        for h in st.session_state.holdings:
            p = price_data.get(h["ticker"])
            if p is not None:
                ticker_values.setdefault(h["ticker"], 0.0)
                ticker_values[h["ticker"]] += p * h["shares"]

        # Compute per-ticker drawdowns for every scenario
        stress_results = []
        scenario_details: dict[str, pd.DataFrame] = {}

        with st.spinner("Computing historical drawdowns..."):
            for name, (start, end) in STRESS_SCENARIOS.items():
                ticker_drawdowns: dict[str, float] = {}

                # Download & compute drawdown for each ticker
                for t in tickers:
                    prices = download_prices(t, start, end)
                    ticker_drawdowns[t] = peak_to_trough(prices)

                # Fallback: for tickers with no data, use the weighted-average
                # drawdown of those that do have data
                tickers_with_data = {
                    t: dd for t, dd in ticker_drawdowns.items() if dd != 0.0
                }
                if tickers_with_data:
                    total_w = sum(ticker_values.get(t, 0) for t in tickers_with_data)
                    if total_w > 0:
                        fallback_dd = sum(
                            dd * ticker_values.get(t, 0) / total_w
                            for t, dd in tickers_with_data.items()
                        )
                    else:
                        fallback_dd = sum(tickers_with_data.values()) / len(tickers_with_data)
                else:
                    fallback_dd = 0.0

                for t in ticker_drawdowns:
                    if ticker_drawdowns[t] == 0.0 and t not in tickers_with_data:
                        ticker_drawdowns[t] = fallback_dd

                # Aggregate portfolio-level results
                portfolio_stressed = 0.0
                detail_rows = []
                for t in tickers:
                    cur_val = ticker_values.get(t, 0.0)
                    dd = ticker_drawdowns[t]
                    stressed_val = cur_val * (1 + dd)
                    portfolio_stressed += stressed_val
                    detail_rows.append({
                        "Ticker": t,
                        "Drawdown": dd,
                        "Current Value": cur_val,
                        "Stressed Value": stressed_val,
                    })

                weighted_dd = (
                    (portfolio_stressed - total_value) / total_value * 100
                    if total_value > 0 else 0.0
                )

                stress_results.append({
                    "Scenario": name,
                    "Portfolio Drop": f"{weighted_dd:+.1f}%",
                    "Portfolio Value": portfolio_stressed,
                    "Loss ($)": portfolio_stressed - total_value,
                })
                scenario_details[name] = pd.DataFrame(detail_rows)

        stress_df = pd.DataFrame(stress_results)

        # Bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stress_df["Scenario"],
            y=stress_df["Portfolio Value"],
            marker_color=["#ef4444"] * len(stress_df),
            text=[f"${v:,.0f}" for v in stress_df["Portfolio Value"]],
            textposition="outside",
        ))
        fig.add_hline(
            y=total_value, line_dash="dash", line_color="#22c55e",
            annotation_text=f"Current: ${total_value:,.0f}",
        )
        fig.update_layout(
            yaxis_title="Portfolio Value ($)",
            xaxis_title="",
            height=400,
            margin=dict(t=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            stress_df.style.format({
                "Portfolio Value": "${:,.2f}",
                "Loss ($)": "${:+,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Per-ticker breakdown expanders
        for name, detail_df in scenario_details.items():
            with st.expander(f"{name} — per-ticker breakdown"):
                st.dataframe(
                    detail_df.style.format({
                        "Drawdown": "{:+.1%}",
                        "Current Value": "${:,.2f}",
                        "Stressed Value": "${:,.2f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

    # ------------------------------------------------------------------
    # Risk & Performance Analytics
    # ------------------------------------------------------------------
    st.divider()
    st.header("Risk & Performance Analytics")

    TRADING_DAYS = 252
    LOOKBACK_MAP = {"6 Months": 183, "1 Year": 365, "2 Years": 730}
    lookback_label = st.selectbox(
        "Lookback period", list(LOOKBACK_MAP.keys()), index=1,
    )
    cal_days = LOOKBACK_MAP[lookback_label]
    hist_start = (pd.Timestamp.now() - pd.Timedelta(days=cal_days)).strftime("%Y-%m-%d")
    hist_end = pd.Timestamp.now().strftime("%Y-%m-%d")

    with st.spinner("Downloading historical prices..."):
        hist_prices: dict[str, pd.Series] = {}
        for t in tickers:
            s = download_prices(t, hist_start, hist_end)
            if not s.empty:
                hist_prices[t] = s

    if not hist_prices:
        st.warning("No historical price data available for analytics.")
    else:
        # Align to common dates, forward-fill gaps, compute daily returns
        price_df = pd.DataFrame(hist_prices).ffill().dropna()
        daily_returns = price_df.pct_change().dropna()

        if len(daily_returns) < 2:
            st.warning("Not enough overlapping history for analytics.")
        else:
            # Portfolio weights from current market values
            w = pd.Series(
                {
                    t: (price_data.get(t) or 0)
                    * next(
                        (h["shares"] for h in st.session_state.holdings if h["ticker"] == t),
                        0,
                    )
                    for t in daily_returns.columns
                }
            )
            w = w / w.sum()

            port_returns = daily_returns.dot(w)

            perf_tab2, var_tab, mc_tab = st.tabs(
                ["Performance", "VaR / CVaR", "Monte Carlo"]
            )

            # ==============================================================
            # Performance tab
            # ==============================================================
            with perf_tab2:
                rf_annual = (
                    st.number_input(
                        "Risk-free rate (annual %)",
                        value=5.0,
                        step=0.25,
                        format="%.2f",
                        key="rf_rate",
                    )
                    / 100
                )
                rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1

                mean_daily = port_returns.mean()
                std_daily = port_returns.std()
                ann_return = (1 + mean_daily) ** TRADING_DAYS - 1
                ann_vol = std_daily * np.sqrt(TRADING_DAYS)

                sharpe = (
                    (ann_return - rf_annual) / ann_vol if ann_vol > 0 else 0.0
                )

                downside = port_returns[port_returns < rf_daily]
                downside_std = (
                    downside.std() if len(downside) > 1 else std_daily
                )
                sortino = (
                    (ann_return - rf_annual)
                    / (downside_std * np.sqrt(TRADING_DAYS))
                    if downside_std > 0
                    else 0.0
                )

                cum_returns = (1 + port_returns).cumprod()
                running_max = cum_returns.cummax()
                drawdown_series = (cum_returns - running_max) / running_max
                max_dd = float(drawdown_series.min())

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Ann. Return", f"{ann_return:+.2%}")
                m2.metric("Ann. Volatility", f"{ann_vol:.2%}")
                m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                m4.metric("Sortino Ratio", f"{sortino:.2f}")
                m5.metric("Max Drawdown", f"{max_dd:.2%}")

                st.divider()

                # Cumulative returns chart
                st.subheader("Cumulative Returns")
                cum_fig = go.Figure()
                cum_fig.add_trace(
                    go.Scatter(
                        x=cum_returns.index,
                        y=(cum_returns - 1) * 100,
                        mode="lines",
                        name="Portfolio",
                        line=dict(color="#3b82f6", width=2),
                    )
                )
                cum_fig.update_layout(
                    yaxis_title="Cumulative Return (%)",
                    height=350,
                    margin=dict(t=20),
                )
                st.plotly_chart(cum_fig, use_container_width=True)

                # Drawdown chart
                st.subheader("Drawdown")
                dd_fig = go.Figure()
                dd_fig.add_trace(
                    go.Scatter(
                        x=drawdown_series.index,
                        y=drawdown_series * 100,
                        mode="lines",
                        fill="tozeroy",
                        name="Drawdown",
                        line=dict(color="#ef4444", width=1),
                        fillcolor="rgba(239,68,68,0.2)",
                    )
                )
                dd_fig.update_layout(
                    yaxis_title="Drawdown (%)",
                    height=250,
                    margin=dict(t=20),
                )
                st.plotly_chart(dd_fig, use_container_width=True)

                # Daily returns distribution
                st.subheader("Daily Returns Distribution")
                dist_fig = go.Figure()
                dist_fig.add_trace(
                    go.Histogram(
                        x=port_returns * 100,
                        nbinsx=50,
                        marker_color="#3b82f6",
                        opacity=0.7,
                        name="Daily Returns",
                    )
                )
                dist_fig.update_layout(
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    height=300,
                    margin=dict(t=20),
                )
                st.plotly_chart(dist_fig, use_container_width=True)

            # ==============================================================
            # VaR / CVaR tab
            # ==============================================================
            with var_tab:
                st.subheader("Value at Risk & Conditional VaR")
                st.caption(
                    "Estimated 1-day potential loss at the selected confidence level."
                )

                confidence = st.selectbox(
                    "Confidence level",
                    [90.0, 95.0, 99.0],
                    index=1,
                    format_func=lambda x: f"{x:.0f}%",
                    key="var_confidence",
                )
                alpha = 1 - confidence / 100

                # --- Historical VaR / CVaR ---
                hist_q = float(port_returns.quantile(alpha))
                var_hist_pct = -hist_q
                cvar_hist_pct = -float(
                    port_returns[port_returns <= hist_q].mean()
                )
                var_hist_dollar = var_hist_pct * total_value
                cvar_hist_dollar = cvar_hist_pct * total_value

                # --- Parametric VaR / CVaR (normal assumption) ---
                Z_SCORES = {90.0: 1.2816, 95.0: 1.6449, 99.0: 2.3263}
                PHI_AT_Z = {90.0: 0.17550, 95.0: 0.10314, 99.0: 0.02665}
                z = Z_SCORES[confidence]

                var_param_pct = z * std_daily - mean_daily
                cvar_param_pct = (
                    std_daily * PHI_AT_Z[confidence] / alpha - mean_daily
                )
                var_param_dollar = var_param_pct * total_value
                cvar_param_dollar = cvar_param_pct * total_value

                st.markdown("#### Historical")
                hc1, hc2 = st.columns(2)
                hc1.metric(
                    f"VaR ({confidence:.0f}%)",
                    f"${var_hist_dollar:,.0f}",
                    f"{var_hist_pct:.2%} of portfolio",
                    delta_color="inverse",
                )
                hc2.metric(
                    f"CVaR ({confidence:.0f}%)",
                    f"${cvar_hist_dollar:,.0f}",
                    f"{cvar_hist_pct:.2%} of portfolio",
                    delta_color="inverse",
                )

                st.markdown("#### Parametric (Normal)")
                pc1, pc2 = st.columns(2)
                pc1.metric(
                    f"VaR ({confidence:.0f}%)",
                    f"${var_param_dollar:,.0f}",
                    f"{var_param_pct:.2%} of portfolio",
                    delta_color="inverse",
                )
                pc2.metric(
                    f"CVaR ({confidence:.0f}%)",
                    f"${cvar_param_dollar:,.0f}",
                    f"{cvar_param_pct:.2%} of portfolio",
                    delta_color="inverse",
                )

                # Returns histogram with VaR lines
                st.divider()
                var_fig = go.Figure()
                var_fig.add_trace(
                    go.Histogram(
                        x=port_returns * 100,
                        nbinsx=50,
                        marker_color="#3b82f6",
                        opacity=0.7,
                        name="Daily Returns",
                    )
                )
                var_fig.add_vline(
                    x=-var_hist_pct * 100,
                    line_dash="solid",
                    line_color="#ef4444",
                    annotation_text=f"Hist VaR: {-var_hist_pct * 100:.2f}%",
                )
                var_fig.add_vline(
                    x=-cvar_hist_pct * 100,
                    line_dash="dash",
                    line_color="#dc2626",
                    annotation_text=f"Hist CVaR: {-cvar_hist_pct * 100:.2f}%",
                )
                var_fig.update_layout(
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    height=350,
                    margin=dict(t=30),
                )
                st.plotly_chart(var_fig, use_container_width=True)

            # ==============================================================
            # Monte Carlo tab
            # ==============================================================
            with mc_tab:
                st.subheader("Monte Carlo Simulation")
                st.caption(
                    "Geometric Brownian Motion paths based on historical "
                    "mean return and volatility."
                )

                mc_c1, mc_c2 = st.columns(2)
                with mc_c1:
                    n_sims = st.selectbox(
                        "Simulations",
                        [1000, 5000, 10000],
                        index=1,
                        key="mc_sims",
                    )
                with mc_c2:
                    horizon = st.selectbox(
                        "Horizon (trading days)",
                        [21, 63, 126, 252],
                        index=0,
                        format_func=lambda d: {
                            21: "1 Month (21)",
                            63: "3 Months (63)",
                            126: "6 Months (126)",
                            252: "1 Year (252)",
                        }[d],
                        key="mc_horizon",
                    )

                rng = np.random.default_rng(42)
                mu = mean_daily
                sigma = std_daily

                rand = rng.standard_normal((n_sims, horizon))
                log_rets = (mu - 0.5 * sigma**2) + sigma * rand
                cum_log = np.cumsum(log_rets, axis=1)
                paths = total_value * np.exp(cum_log)

                terminal = paths[:, -1]
                mc_mean = float(np.mean(terminal))
                mc_median = float(np.median(terminal))
                mc_5 = float(np.percentile(terminal, 5))
                mc_95 = float(np.percentile(terminal, 95))

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Mean", f"${mc_mean:,.0f}")
                s2.metric("Median", f"${mc_median:,.0f}")
                s3.metric("5th Percentile", f"${mc_5:,.0f}")
                s4.metric("95th Percentile", f"${mc_95:,.0f}")

                # Terminal value histogram
                st.subheader("Terminal Value Distribution")
                mc_hist = go.Figure()
                mc_hist.add_trace(
                    go.Histogram(
                        x=terminal,
                        nbinsx=80,
                        marker_color="#8b5cf6",
                        opacity=0.7,
                    )
                )
                mc_hist.add_vline(
                    x=total_value,
                    line_dash="dash",
                    line_color="#22c55e",
                    annotation_text=f"Current: ${total_value:,.0f}",
                )
                mc_hist.add_vline(
                    x=mc_5,
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text=f"5th: ${mc_5:,.0f}",
                )
                mc_hist.add_vline(
                    x=mc_95,
                    line_dash="dash",
                    line_color="#3b82f6",
                    annotation_text=f"95th: ${mc_95:,.0f}",
                )
                mc_hist.update_layout(
                    xaxis_title="Portfolio Value ($)",
                    yaxis_title="Frequency",
                    height=400,
                    margin=dict(t=30),
                )
                st.plotly_chart(mc_hist, use_container_width=True)

                # Fan chart — percentile bands over time
                st.subheader("Simulation Paths (Percentile Bands)")
                pctl_levels = [5, 25, 50, 75, 95]
                days_ax = np.arange(1, horizon + 1)
                pctl_vals = np.percentile(paths, pctl_levels, axis=0)

                fan = go.Figure()
                # 5th–95th band
                fan.add_trace(
                    go.Scatter(
                        x=np.concatenate([days_ax, days_ax[::-1]]),
                        y=np.concatenate([pctl_vals[4], pctl_vals[0][::-1]]),
                        fill="toself",
                        fillcolor="rgba(139,92,246,0.1)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="5th – 95th",
                    )
                )
                # 25th–75th band
                fan.add_trace(
                    go.Scatter(
                        x=np.concatenate([days_ax, days_ax[::-1]]),
                        y=np.concatenate([pctl_vals[3], pctl_vals[1][::-1]]),
                        fill="toself",
                        fillcolor="rgba(139,92,246,0.25)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="25th – 75th",
                    )
                )
                # Median line
                fan.add_trace(
                    go.Scatter(
                        x=days_ax,
                        y=pctl_vals[2],
                        mode="lines",
                        name="Median",
                        line=dict(color="#8b5cf6", width=2),
                    )
                )
                fan.add_hline(
                    y=total_value,
                    line_dash="dash",
                    line_color="#22c55e",
                    annotation_text=f"Current: ${total_value:,.0f}",
                )
                fan.update_layout(
                    xaxis_title="Trading Days",
                    yaxis_title="Portfolio Value ($)",
                    height=400,
                    margin=dict(t=30),
                )
                st.plotly_chart(fan, use_container_width=True)

    # --- Portfolio Assistant Chatbot ---
    st.divider()
    st.header("Portfolio Assistant")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("Set OPENAI_API_KEY in your .env file to enable the Portfolio Assistant.")
    else:
        # Build portfolio context for the system prompt
        portfolio_lines = []
        for r in rows:
            line = (
                f"{r['Ticker']}: {r['Shares']} shares, "
                f"cost basis ${r['Cost Basis']:.2f}, "
                f"current price ${r['Current Price']:.2f}, "
                f"market value ${r['Market Value']:,.2f}, "
                f"gain/loss ${r['Gain/Loss ($)']:+,.2f} ({r['Gain/Loss (%)']:+.2f}%)"
            ) if r["Current Price"] is not None else f"{r['Ticker']}: price unavailable"
            portfolio_lines.append(line)

        portfolio_summary = "\n".join(portfolio_lines)
        system_prompt = (
            "You are a helpful portfolio advisor. The user has the following holdings:\n\n"
            f"{portfolio_summary}\n\n"
            f"Total portfolio value: ${total_value:,.2f}\n"
            f"Total cost basis: ${total_cost:,.2f}\n"
            f"Total gain/loss: ${total_gain:+,.2f} ({total_gain_pct:+.2f}%)\n\n"
            "Provide concise, actionable advice referencing their actual holdings. "
            "Remind the user this is not financial advice."
        )

        # Chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about your portfolio..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                client = OpenAI(api_key=api_key)
                with st.spinner("Thinking..."):
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        max_tokens=1024,
                        messages=[
                            {"role": "system", "content": system_prompt},
                        ] + [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat_messages
                        ],
                    )
                    reply = response.choices[0].message.content
                st.markdown(reply)

            st.session_state.chat_messages.append({"role": "assistant", "content": reply})

