# Stock Solve

Stock Solve is a reproducible stock-selection backtest and paper-portfolio tracker. It compares a diversified momentum portfolio with Vanguard's `VFIAX` S&P 500 index fund.

The project starts with a clean $250,000 paper portfolio. It does not contain brokerage credentials, private account data, or the author's local portfolio history. It never places real brokerage orders.

> This software is for research and paper trading. It is not investment advice, and historical results do not guarantee future performance.

## Requirements

- Python 3.9 or newer
- Git
- Internet access for Yahoo Finance market data
- No API key is required

## Install

```bash
git clone https://github.com/mraftelis/Stock_Solve.git
cd Stock_Solve
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

`requirements-lock.txt` provides the tested, reproducible environment. `requirements.txt` contains broader direct dependency ranges for maintainers testing upgrades.

Verify the installation:

```bash
python -m unittest discover -s tests
```

## Run The Backtest

```bash
python -m stock_solve.backtest --assert-target
```

The first run downloads and caches adjusted daily prices in `data/adjusted_closes.csv`. Use `--no-cache` to force a fresh download.

Generated files are written to `outputs/`:

- `summary.json`
- `equity_curve.csv`
- `rebalances.csv`
- `equity_curve.png`

Both `data/*.csv` and `outputs/` are ignored by Git because market data and generated results can change between runs.

## Start A Clean Paper Portfolio

Reset to $250,000 cash and create a pending 20-stock order from the latest completed close:

```bash
python -m stock_solve.reset_paper_portfolio --capital 250000
```

The reset also establishes a new $250,000 `VFIAX` comparison basis. It does not fill stocks at the signal close.

At the next market open, run:

```bash
python -m stock_solve.track_paper_portfolio --rebalance
```

The tracker validates the pending target using that day's adjusted opening prices. It can cancel or change the target if the opening data no longer supports the close signal, then records the paper fill. Later runs update valuation using the latest available adjusted prices.

Open the generated dashboard after the first run:

```text
outputs/dashboard.html
```

The dashboard shows portfolio value, daily history, current positions, concentration, rebalance history, and normalized performance versus `VFIAX`.

For a research snapshot that does not use the staged next-open workflow, run:

```bash
python -m stock_solve.live_portfolio --no-cache
```

## Continuous Tracking

Run the tracker twice each trading weekday:

- Shortly after the market opens, to validate and execute an existing paper order.
- After the market closes, to update results and potentially stage the next weekly rebalance.

For a machine using the America/Chicago timezone, a cron example is:

```cron
40 8 * * 1-5 cd /absolute/path/to/Stock_Solve && .venv/bin/python -m stock_solve.track_paper_portfolio --rebalance
40 17 * * 1-5 cd /absolute/path/to/Stock_Solve && .venv/bin/python -m stock_solve.track_paper_portfolio --rebalance
```

Use absolute paths and configure the scheduler's timezone explicitly when the host is not set to America/Chicago. A sleeping or powered-off computer will miss local scheduled runs.

## Paper Rebalance Rules

- Signals use completed daily closes only; an in-progress intraday bar is excluded.
- A close signal cannot execute at that same close.
- Replacements are weekly-gated to limit churn.
- A pending order is validated at the next available open before a paper fill.
- Existing holdings remain eligible while ranked 30 or better.
- New candidates must rank 20 or better for two consecutive signals.
- A replacement must improve the risk-adjusted score by at least 3 percentage points.
- The portfolio holds 20 stocks in approximately equal weights.
- No stock may exceed 15% of portfolio value.
- The semiconductor/hardware risk group is capped at 8 of the 20 holdings.

The live ranking score is:

```text
126-day momentum
- 0.55 * annualized volatility
+ 0.75 * drawdown from the 126-day high
```

Because drawdown is zero at the recent high and negative below it, the final term penalizes stocks trading materially below their recent high.

## Historical Backtest Method

The historical strategy uses only data available before each rebalance:

1. Start with the historical Nasdaq-100 constituent list documented in the source code, rather than today's winners.
2. Use Yahoo Finance adjusted closing prices.
3. At each month-end, rank stocks by trailing 126-trading-day momentum through the prior trading day.
4. Require sufficient and current price history so stale or delisted prices cannot be selected.
5. Hold the top 20 stocks at equal target weights.
6. Charge 5 basis points on traded value at every rebalance.
7. Compare with `VFIAX` from 2016-06-01 through 2026-06-01, starting with $250,000.

The test suite verifies the minimum holding count, 15% position cap, selection behavior, completed-close handling, and open-validation logic.

## Last Verified Result

Fresh download and clean-clone verification performed on 2026-10-03:

| Metric | Strategy | VFIAX |
| --- | ---: | ---: |
| Backtest window | 2016-06-01 to 2026-06-01 | 2016-06-01 to 2026-06-01 |
| CAGR | 28.09% | 15.61% |
| Final value from $250,000 | $2,970,312 | $1,066,370 |
| Max drawdown | -31.32% | -33.83% |

The strategy exceeded `VFIAX` by 12.48 percentage points annualized, held at least 20 stocks, and had a maximum observed position weight of 10.17%.

Results can move when Yahoo Finance revises historical data. On the verification date, Yahoo returned no usable history for 27 of the 104 requested historical symbols, primarily because they were acquired or delisted. Production research should use paid point-in-time constituent and delisted-security total-return data.

## Repository Hygiene

- GitHub Actions runs the synthetic test suite on every push and pull request.
- Local market-data caches, paper holdings, history, pending orders, and dashboards are excluded from Git.
- The public repository always starts a new paper portfolio; it does not inherit the author's local tracking state.
- Dependency upgrades should be tested with `requirements.txt`, then frozen into `requirements-lock.txt` after the suite passes.

## License

MIT. See [LICENSE](LICENSE).
