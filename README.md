# Portfolio Tracker with Scenario & Stress Testing

This project is a Streamlit-based portfolio tracker that allows users to track equity holdings in real time and evaluate portfolio risk using custom scenarios and historical stress tests.

## Features

### Portfolio Tracking
- Add and remove equity holdings (persisted in SQLite)
- CSV bulk import with template download and validation
- Live price data pulled from Yahoo Finance
- Automatic calculation of:
  - Market value
  - Dollar gain/loss
  - Percentage return
- Summary metrics:
  - Total portfolio value
  - Total gain/loss
  - Number of positions

### What-If Scenario Testing
- Apply a uniform market-wide price shock (-50% to +50%)
- Override individual ticker returns
- Compare:
  - Current portfolio value
  - Scenario portfolio value
  - Dollar and percentage impact

### Historical Stress Tests
Simulate portfolio performance under major historical market events using actual per-ticker peak-to-trough drawdowns:
- 2008 Financial Crisis
- COVID Crash (March 2020)
- Dot-com Bust (2000-2002)
- Black Monday (1987)
- 2022 Bear Market

Per-ticker breakdown available in expandable panels.

### Risk & Performance Analytics
- **Performance**: Annualized return, volatility, Sharpe ratio, Sortino ratio, max drawdown
- **Charts**: Cumulative returns, drawdown, daily returns distribution
- **VaR / CVaR**: Historical and parametric (normal) at 90/95/99% confidence
- **Monte Carlo**: GBM simulation with configurable paths and horizon, terminal value histogram, percentile fan chart

### AI Risk Brief
Uses the Anthropic API (Claude) to generate a concise narrative risk summary from the computed metrics. Requires the `anthropic` Python package and an API key.

## AI Risk Brief — Exact Prompts

The prompts are defined as constants in `app.py` (`RISK_BRIEF_SYSTEM` and `RISK_BRIEF_USER_TEMPLATE`) and are viewable in-app via the "View prompt sent to Claude" expander. The call uses `temperature=0` for deterministic output.

### System Prompt

```
You are a portfolio risk analyst. Given portfolio data and computed risk metrics, produce a concise risk brief.

Structure your response in exactly three sections:

**Portfolio Overview**
2-3 sentences summarizing the portfolio composition, total value, and overall risk posture.

**Key Risk Findings**
3-5 bullet points citing specific numbers from the data. Cover concentration risk, volatility, drawdown exposure, and tail risk (VaR/CVaR). Compare Sharpe and Sortino ratios to characterize the return/risk profile.

**Recommendations**
2-4 bullet points with specific, actionable suggestions. Reference the data to justify each recommendation.

Rules:
- Cite exact numbers from the provided data.
- Keep total response under 300 words.
- Do not use hedging language ("consider", "might", "could"). State findings directly.
- Do not add disclaimers about not being financial advice.
```

### User Message Template

The `{placeholders}` are populated with live portfolio data before each call.

```
## Portfolio Holdings
{holdings_table}

## Performance Metrics ({lookback} lookback)
- Annualized Return: {ann_return}
- Annualized Volatility: {ann_vol}
- Sharpe Ratio: {sharpe}
- Sortino Ratio: {sortino}
- Max Drawdown: {max_dd}

## Value at Risk ({confidence}% confidence, 1-day)
- Historical VaR: ${var_hist_dollar} ({var_hist_pct} of portfolio)
- Historical CVaR: ${cvar_hist_dollar} ({cvar_hist_pct} of portfolio)
- Parametric VaR: ${var_param_dollar} ({var_param_pct} of portfolio)
- Parametric CVaR: ${cvar_param_dollar} ({cvar_param_pct} of portfolio)

## Monte Carlo Simulation ({mc_horizon}-day horizon, {mc_sims} simulations)
- Current portfolio value: ${current_value}
- Mean terminal value: ${mc_mean}
- Median terminal value: ${mc_median}
- 5th percentile: ${mc_5}
- 95th percentile: ${mc_95}

## Stress Test Results
{stress_table}
```

### API Call Parameters

| Parameter   | Value                        |
|-------------|------------------------------|
| Model       | `claude-sonnet-4-5-20250929` |
| Temperature | `0`                          |
| Max tokens  | `1024`                       |

## Tech Stack
- Python
- Streamlit
- Pandas / NumPy
- Plotly
- Yahoo Finance (yfinance)
- SQLite (persistence)
- Anthropic API (AI Risk Brief, optional)

## Running the App Locally

```bash
pip install -r requirements.txt

# Optional: enable AI Risk Brief
pip install anthropic
# Set your ANTHROPIC_API_KEY as an environment variable (see provider docs)

streamlit run app.py
```

Then open the app in your browser at: http://localhost:8501
