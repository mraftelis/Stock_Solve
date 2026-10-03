from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


STARTING_CAPITAL = 250_000.0
BENCHMARK = "VFIAX"
BACKTEST_START = "2016-06-01"
BACKTEST_END = "2026-06-02"  # yfinance end is exclusive; includes 2026-06-01.
DATA_START = "2015-06-01"
MIN_HOLDINGS = 20
MAX_POSITION_WEIGHT = 0.15
PICKS = 20
MOMENTUM_LOOKBACK_DAYS = 126
TRADE_COST_BPS = 5
SUCCESS_EXCESS_CAGR = 0.01

# Nasdaq-100 constituent list copied from a public Stack Overflow question
# asked on 2016-06-17. This replaces the previous hindsight universe of current
# winners with a universe an investor could have known near the start of the
# 10-year test. Delisted/acquired names are retained; if the free data source
# cannot return prices for a symbol, it remains in the requested universe but
# cannot receive a momentum score.
UNIVERSE = [
    "AAL",
    "AAPL",
    "ADBE",
    "ADI",
    "ADP",
    "ADSK",
    "AKAM",
    "ALXN",
    "AMAT",
    "AMGN",
    "AMZN",
    "ATVI",
    "BBBY",
    "BIDU",
    "BIIB",
    "BMRN",
    "CA",
    "CELG",
    "CERN",
    "CHKP",
    "CHTR",
    "CMCSA",
    "COST",
    "CSCO",
    "CSX",
    "CTRP",
    "CTSH",
    "CTXS",
    "DISCA",
    "DISCK",
    "DISH",
    "DLTR",
    "EA",
    "EBAY",
    "ENDP",
    "ESRX",
    "EXPE",
    "FAST",
    "FB",
    "FISV",
    "FOX",
    "FOXA",
    "GILD",
    "GOOG",
    "GOOGL",
    "HSIC",
    "INCY",
    "INTC",
    "INTU",
    "ILMN",
    "ISRG",
    "JD",
    "KHC",
    "LBTYA",
    "LBTYK",
    "LLTC",
    "LMCA",
    "LRCX",
    "LVNTA",
    "MAR",
    "MAT",
    "MDLZ",
    "MNST",
    "MSFT",
    "MU",
    "MXIM",
    "MYL",
    "NCLH",
    "NFLX",
    "NTAP",
    "NVDA",
    "NXPI",
    "ORLY",
    "PAYX",
    "PCAR",
    "PCLN",
    "PYPL",
    "QCOM",
    "QVCA",
    "REGN",
    "ROST",
    "SBAC",
    "SBUX",
    "SIRI",
    "SNDK",
    "SRCL",
    "STX",
    "SWKS",
    "SYMC",
    "TMUS",
    "TSCO",
    "TSLA",
    "TRIP",
    "TXN",
    "ULTA",
    "VIAB",
    "VOD",
    "VRSK",
    "VRTX",
    "WBA",
    "WDC",
    "WFM",
    "XLNX",
    "YHOO",
]


@dataclass(frozen=True)
class BacktestConfig:
    data_start: str = DATA_START
    start: str = BACKTEST_START
    end: str = BACKTEST_END
    benchmark: str = BENCHMARK
    starting_capital: float = STARTING_CAPITAL
    picks: int = PICKS
    momentum_lookback_days: int = MOMENTUM_LOOKBACK_DAYS
    min_holdings: int = MIN_HOLDINGS
    max_position_weight: float = MAX_POSITION_WEIGHT
    trade_cost_bps: float = TRADE_COST_BPS
    success_excess_cagr: float = SUCCESS_EXCESS_CAGR


@dataclass(frozen=True)
class Performance:
    final_value: float
    cagr: float
    annualized_volatility: float
    sharpe_0_rate: float
    max_drawdown: float


def download_adjusted_closes(
    tickers: Iterable[str], start: str, end: str, cache_path: Path | None = None
) -> pd.DataFrame:
    """Download adjusted closes and optionally cache them as CSV."""
    tickers = list(dict.fromkeys(tickers))
    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        missing = [ticker for ticker in tickers if ticker not in cached.columns]
        if not missing:
            return cached[tickers].sort_index()

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})

    closes = closes.dropna(axis=1, how="all").sort_index()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        closes.to_csv(cache_path)
    return closes


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def performance(equity: pd.Series) -> Performance:
    daily_returns = equity.pct_change().dropna()
    annualized_volatility = float(daily_returns.std() * np.sqrt(252))
    annualized_return = cagr(equity)
    sharpe = annualized_return / annualized_volatility if annualized_volatility else np.nan
    return Performance(
        final_value=float(equity.iloc[-1]),
        cagr=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_0_rate=float(sharpe),
        max_drawdown=max_drawdown(equity),
    )


def first_trading_days_after_month_end(index: pd.DatetimeIndex, start: str) -> list[pd.Timestamp]:
    month_ends = pd.Series(index=index, dtype=float).resample("M").last().index
    rebalancing_dates: list[pd.Timestamp] = []
    start_idx = index.searchsorted(pd.Timestamp(start))
    if start_idx < len(index):
        rebalancing_dates.append(index[start_idx])
    for month_end in month_ends:
        idx = index.searchsorted(month_end)
        if idx < len(index):
            date = index[idx]
            if date >= pd.Timestamp(start) and date not in rebalancing_dates:
                rebalancing_dates.append(date)
    return rebalancing_dates


def select_momentum_portfolio(
    prices_to_signal_date: pd.DataFrame, lookback_days: int, picks: int
) -> list[str]:
    current = prices_to_signal_date.iloc[-1]
    past = prices_to_signal_date.iloc[-lookback_days - 1]
    momentum = (current / past - 1).replace([np.inf, -np.inf], np.nan).dropna()

    valid_scores: dict[str, float] = {}
    for ticker, score in momentum.items():
        history = prices_to_signal_date[ticker].dropna()
        if (
            len(history) >= lookback_days + 1
            and pd.notna(current.get(ticker))
            and history.index[-1] == prices_to_signal_date.index[-1]
        ):
            valid_scores[ticker] = float(score)

    ranked = sorted(valid_scores, key=valid_scores.get, reverse=True)
    return ranked[:picks]


def run_backtest(
    prices: pd.DataFrame, universe: list[str], config: BacktestConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    benchmark_prices = prices[config.benchmark].dropna()
    stock_prices = prices[[ticker for ticker in universe if ticker in prices.columns]].copy()
    stock_returns = stock_prices.pct_change(fill_method=None).fillna(0)
    rebalancing_dates = set(first_trading_days_after_month_end(stock_prices.index, config.start))

    positions = pd.Series(0.0, index=stock_prices.columns)
    cash = config.starting_capital
    value = config.starting_capital
    previous_date: pd.Timestamp | None = None
    equity_rows = []
    rebalance_rows = []

    for date, daily_returns in stock_returns.loc[config.start :].iterrows():
        if previous_date is not None:
            positions = positions * (1 + daily_returns.fillna(0))
            value = float(positions.sum() + cash)

        if date in rebalancing_dates:
            signal_position = stock_prices.index.get_loc(date) - 1
            if signal_position >= config.momentum_lookback_days:
                price_history = stock_prices.iloc[: signal_position + 1]
                picks = select_momentum_portfolio(
                    price_history, config.momentum_lookback_days, config.picks
                )
                if len(picks) >= config.min_holdings:
                    target_weights = pd.Series(0.0, index=stock_prices.columns)
                    target_weights[picks] = 1 / len(picks)
                    target_positions = target_weights * value
                    traded_value = float((target_positions - positions).abs().sum())
                    transaction_cost = traded_value * config.trade_cost_bps / 10_000
                    value -= transaction_cost
                    positions = target_weights * value
                    cash = 0.0
                    turnover = traded_value / value if value else 0.0
                    rebalance_rows.append(
                        {
                            "date": date,
                            "signal_date": stock_prices.index[signal_position],
                            "holdings": len(picks),
                            "max_weight": float((positions / value).max()),
                            "turnover": turnover,
                            "transaction_cost": transaction_cost,
                            "tickers": ",".join(picks),
                        }
                    )

        daily_weights = positions / value if value else positions
        equity_rows.append(
            {
                "date": date,
                "strategy": value,
                "cash_weight": float(cash / value) if value else 0.0,
                "holdings": int((positions > 0).sum()),
                "max_weight": float(daily_weights.max()),
            }
        )
        previous_date = date

    equity = pd.DataFrame(equity_rows).set_index("date")
    benchmark_equity = benchmark_prices.loc[equity.index[0] : equity.index[-1]].reindex(equity.index).ffill()
    benchmark_equity = benchmark_equity / benchmark_equity.iloc[0] * config.starting_capital
    equity["benchmark"] = benchmark_equity
    equity = equity.dropna(subset=["benchmark"])

    rebalances = pd.DataFrame(rebalance_rows)
    strategy_perf = performance(equity["strategy"])
    benchmark_perf = performance(equity["benchmark"])
    summary = {
        "config": asdict(config),
        "data_start": str(prices.index.min().date()),
        "data_end": str(prices.index.max().date()),
        "requested_universe_size": len(universe),
        "priced_universe_size": int(len(stock_prices.dropna(axis=1, how="all").columns)),
        "unpriced_symbols": sorted(
            set(universe) - set(stock_prices.dropna(axis=1, how="all").columns)
        ),
        "backtest_start": str(equity.index.min().date()),
        "backtest_end": str(equity.index.max().date()),
        "strategy": asdict(strategy_perf),
        "benchmark": asdict(benchmark_perf),
        "excess_cagr": strategy_perf.cagr - benchmark_perf.cagr,
        "starting_capital": config.starting_capital,
        "average_turnover_per_rebalance": float(rebalances["turnover"].mean()),
        "max_turnover_per_rebalance": float(rebalances["turnover"].max()),
        "rebalance_count": int(len(rebalances)),
        "average_rebalances_per_year": float(len(rebalances) / 10),
        "min_holdings": int(equity.loc[equity["holdings"] > 0, "holdings"].min()),
        "max_observed_weight": float(equity["max_weight"].max()),
        "success": bool(
            strategy_perf.cagr - benchmark_perf.cagr >= config.success_excess_cagr
            and equity.loc[equity["holdings"] > 0, "holdings"].min() >= config.min_holdings
            and equity["max_weight"].max() <= config.max_position_weight
        ),
    }
    return equity, rebalances, summary


def write_outputs(
    equity: pd.DataFrame, rebalances: pd.DataFrame, summary: dict, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    equity.to_csv(output_dir / "equity_curve.csv")
    rebalances.to_csv(output_dir / "rebalances.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    equity[["strategy", "benchmark"]].plot(ax=ax)
    ax.set_title("Strategy vs VFIAX")
    ax.set_ylabel("Portfolio value ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.close(fig)


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_summary(summary: dict) -> None:
    print("Backtest window:", summary["backtest_start"], "to", summary["backtest_end"])
    print("Strategy CAGR:", format_pct(summary["strategy"]["cagr"]))
    print("VFIAX CAGR:", format_pct(summary["benchmark"]["cagr"]))
    print("Excess CAGR:", format_pct(summary["excess_cagr"]))
    print("Strategy final value:", f"${summary['strategy']['final_value']:,.0f}")
    print("VFIAX final value:", f"${summary['benchmark']['final_value']:,.0f}")
    print("Strategy max drawdown:", format_pct(summary["strategy"]["max_drawdown"]))
    print("VFIAX max drawdown:", format_pct(summary["benchmark"]["max_drawdown"]))
    print("Minimum holdings:", summary["min_holdings"])
    print("Maximum observed position weight:", format_pct(summary["max_observed_weight"]))
    print("Average turnover per monthly rebalance:", format_pct(summary["average_turnover_per_rebalance"]))
    print("Success:", summary["success"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest Stock Solve strategy against VFIAX.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache", type=Path, default=Path("data/adjusted_closes.csv"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--assert-target", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BacktestConfig()
    cache_path = None if args.no_cache else args.cache
    prices = download_adjusted_closes(
        UNIVERSE + [config.benchmark], config.data_start, config.end, cache_path
    )
    equity, rebalances, summary = run_backtest(prices, UNIVERSE, config)
    write_outputs(equity, rebalances, summary, args.output_dir)
    print_summary(summary)
    if args.assert_target and not summary["success"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
