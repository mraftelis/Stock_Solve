# Stock Solve

This repository contains a reproducible stock-picking backtest against `VFIAX`.

## Strategy

The implemented strategy is a monthly Nasdaq-100 momentum rotation:

1. Use a historical Nasdaq-100 constituent list published in a [Stack Overflow question asked on 2016-06-17](https://stackoverflow.com/questions/37872004/getyahoodata-ttr-and-getsymbols-quantomod-errors-when-importing-data-for-par), not a list of current winners.
2. Use Yahoo Finance adjusted closing prices.
3. At each month-end rebalance, rank stocks by trailing 126-trading-day adjusted-price momentum using only data available before the rebalance date.
4. Require enough price history ending on the prior trading day, so stale/delisted prices cannot be ranked.
5. Hold the top 20 stocks in equal weights.
6. Apply 5 bps transaction cost to traded value at every rebalance.
7. Compare the resulting portfolio to `VFIAX` from 2016-06-01 through 2026-06-01 with starting capital of $250,000.

Constraint checks are built into the backtest:

- At least 20 stocks must be held after every active rebalance.
- No stock can exceed 15% of portfolio value.
- The strategy must beat `VFIAX` by at least 1 percentage point annualized.

## Run

```bash
python3 -m stock_solve.backtest --assert-target
```

Outputs are written to `outputs/`:

- `summary.json`
- `equity_curve.csv`
- `rebalances.csv`
- `equity_curve.png`

## Paper portfolio

Create a current paper portfolio from the latest downloadable adjusted closes:

```bash
python3 -m stock_solve.live_portfolio --no-cache
```

Track the paper portfolio value and compare current holdings with the latest momentum top 20:

```bash
python3 -m stock_solve.track_paper_portfolio
```

Track and automatically apply paper add/drop changes when the momentum top 20 changes:

```bash
python3 -m stock_solve.track_paper_portfolio --rebalance
```

Reset the paper portfolio to $250,000 cash and stage fresh buys for the next validated open:

```bash
python3 -m stock_solve.reset_paper_portfolio --capital 250000
```

The reset also restarts the `VFIAX` comparison at the same reset value. Because `VFIAX` is a mutual fund and its NAV feed can lag daily stock closes, the benchmark reset price uses the latest available `VFIAX` adjusted close on or before the reset signal date.

Rebalance timing is intentionally split to avoid same-close execution bias:

- After the close on day D, the tracker may create a pending rebalance from the day D close signal.
- The paper rebalance is not applied at that close.
- At the next open, the tracker revalidates the rebalance using execution-day adjusted open prices.
- If the rebalance no longer makes sense at the open, it is cancelled.
- If the top-20 target changed at the open, the paper rebalance uses the open-validated target instead of the stale close target.
- Otherwise, the pending rebalance executes at the first available adjusted open after the signal date.
- Later tracking values use the latest available adjusted close.

Live stock-picking optimizations are designed to reduce concentration, sharp reversal exposure,
and boundary churn:

- Initial cash deployment buys the top 20 stocks by risk-adjusted 126-day momentum.
- The live ranking score is 126-day adjusted-price momentum minus a recent volatility penalty
  and a penalty for being below the 126-day high. This still uses only information available
  on the signal date.
- No single live risk group can occupy more than 8 of the 20 holdings. The current explicit group cap applies to semiconductor/hardware names because the live run showed that pure momentum could become a mostly single-factor semiconductor reversal bet.
- Existing holdings are kept while they remain ranked 30 or better.
- New replacement candidates must be ranked 20 or better.
- New replacement candidates must appear in the top 20 for 2 consecutive signals.
- A replacement must beat the dropped holding by at least 3 percentage points of risk-adjusted score.
- New rebalance signals are weekly-gated; daily checks still update valuation and can execute already-pending morning orders.

Generate the static dashboard:

```bash
python3 -m stock_solve.dashboard
```

The dashboard includes current paper-portfolio value, daily balance history, a normalized `VFIAX` comparison line, excess return versus `VFIAX`, current positions, rebalance history, cash, concentration, and max holding weight.

Paper portfolio outputs are written to `outputs/`:

- `paper_portfolio.csv`
- `paper_portfolio_summary.json`
- `paper_portfolio_tracked.csv`
- `paper_portfolio_tracking_summary.json`
- `paper_portfolio_tracking_history.csv`
- `dashboard.html`

## Verified result

Last verified on 2026-06-01 with Yahoo Finance adjusted closes:

| Metric | Strategy | VFIAX |
| --- | ---: | ---: |
| Backtest window | 2016-06-01 to 2026-06-01 | 2016-06-01 to 2026-06-01 |
| CAGR | 29.16% | 15.61% |
| Final value from $250,000 | $3,226,422 | $1,066,370 |
| Max drawdown | -31.10% | -33.83% |

The strategy exceeded `VFIAX` by 13.54 percentage points annualized. It held 20 stocks at every active rebalance, and the largest daily observed holding weight was 11.77%, below the 15% cap.

The requested historical universe contains 104 symbols. Yahoo Finance returned usable adjusted-close data for 81 of them and no usable data for 23 delisted/acquired symbols: ALXN, ATVI, CELG, CERN, CTRP, CTXS, DISCA, DISCK, DISH, ENDP, LLTC, LMCA, LVNTA, MXIM, MYL, QVCA, SRCL, SYMC, VIAB, WBA, WFM, XLNX, YHOO.

The factor choice is based on established quant research: intermediate-horizon cross-sectional momentum, commonly summarized as buying recent winners and avoiding recent losers. This implementation uses a long-only, monthly, equal-weight version to satisfy the portfolio constraints. Relevant references include Jegadeesh and Titman's 1993 momentum paper and Moskowitz, Ooi, and Pedersen's work on time-series momentum.

## Caveats

This is a historical backtest, not investment advice or a guarantee. The previous hindsight-selected current-stock universe has been removed. The remaining limitation is data availability: free Yahoo Finance data did not provide full historical adjusted closes for 23 delisted/acquired 2016 Nasdaq-100 symbols. A production-grade research version should use paid point-in-time constituent data and delisted-security total-return data before relying on the result for real capital.
