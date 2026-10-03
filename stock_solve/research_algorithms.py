from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stock_solve.backtest import (
    BACKTEST_END,
    BACKTEST_START,
    BENCHMARK,
    DATA_START,
    MAX_POSITION_WEIGHT,
    MIN_HOLDINGS,
    STARTING_CAPITAL,
    TRADE_COST_BPS,
    UNIVERSE,
    download_adjusted_closes,
    first_trading_days_after_month_end,
    performance,
)


SEMICONDUCTOR_HARDWARE = {
    "ADI",
    "AMD",
    "AMAT",
    "INTC",
    "KLAC",
    "LRCX",
    "LLTC",
    "MCHP",
    "MPWR",
    "MU",
    "MXIM",
    "NVDA",
    "NXPI",
    "QCOM",
    "SNDK",
    "STX",
    "SWKS",
    "TXN",
    "WDC",
    "XLNX",
}


@dataclass(frozen=True)
class ResearchStrategy:
    name: str
    description: str
    lookback_days: int = 126
    skip_recent_days: int = 0
    picks: int = MIN_HOLDINGS
    volatility_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    max_semiconductor_count: int | None = None
    market_sma_days: int | None = None
    market_return_days: int | None = None
    target_volatility: float | None = None
    volatility_lookback_days: int = 63


def trading_days_after_week_end(index: pd.DatetimeIndex, start: str) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    start_ts = pd.Timestamp(start)
    start_position = index.searchsorted(start_ts)
    if start_position < len(index):
        dates.append(index[start_position])
    for position, date in enumerate(index[:-1]):
        if date >= start_ts and date.weekday() == 4:
            dates.append(index[position + 1])
    return sorted(set(dates))


def risk_group(ticker: str) -> str:
    return "semiconductor_hardware" if ticker in SEMICONDUCTOR_HARDWARE else ticker


def risk_group_counts(tickers: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ticker in tickers:
        group = risk_group(ticker)
        counts[group] = counts.get(group, 0) + 1
    return counts


def rank_tickers(prices_to_signal_date: pd.DataFrame, strategy: ResearchStrategy) -> pd.DataFrame:
    prices = prices_to_signal_date.dropna(axis=1, how="all").dropna(axis=0, how="all")
    latest = prices.iloc[-1]
    current_offset = strategy.skip_recent_days
    past_offset = strategy.lookback_days + strategy.skip_recent_days
    if len(prices) <= past_offset:
        return pd.DataFrame()

    ranking_current = prices.iloc[-current_offset - 1] if current_offset else prices.iloc[-1]
    ranking_past = prices.iloc[-past_offset - 1]
    momentum = (ranking_current / ranking_past - 1).replace([np.inf, -np.inf], np.nan)
    window = prices.iloc[-past_offset - 1 :]
    volatility = window.pct_change(fill_method=None).std() * np.sqrt(252)
    drawdown = (latest / window.cummax().max() - 1).replace([np.inf, -np.inf], np.nan)

    rows = []
    for ticker, value in momentum.dropna().items():
        history = prices[ticker].dropna()
        if (
            len(history) >= past_offset + 1
            and history.index[-1] == prices.index[-1]
            and pd.notna(latest.get(ticker))
        ):
            ticker_volatility = float(volatility.get(ticker, np.nan))
            ticker_drawdown = float(drawdown.get(ticker, np.nan))
            if np.isnan(ticker_volatility) or np.isnan(ticker_drawdown):
                continue
            score = (
                float(value)
                - strategy.volatility_penalty * ticker_volatility
                + strategy.drawdown_penalty * ticker_drawdown
            )
            rows.append(
                {
                    "ticker": ticker,
                    "momentum": float(value),
                    "annualized_volatility": ticker_volatility,
                    "drawdown_from_high": ticker_drawdown,
                    "score": score,
                }
            )

    ranked = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def select_tickers(ranked: pd.DataFrame, strategy: ResearchStrategy) -> list[str]:
    picks: list[str] = []
    skipped: list[str] = []
    counts: dict[str, int] = {}
    for ticker in ranked["ticker"].tolist():
        group = risk_group(ticker)
        if (
            strategy.max_semiconductor_count is not None
            and group == "semiconductor_hardware"
            and counts.get(group, 0) >= strategy.max_semiconductor_count
        ):
            skipped.append(ticker)
            continue
        picks.append(ticker)
        counts[group] = counts.get(group, 0) + 1
        if len(picks) == strategy.picks:
            break

    if len(picks) < strategy.picks:
        for ticker in ranked["ticker"].tolist():
            if ticker in picks:
                continue
            picks.append(ticker)
            if len(picks) == strategy.picks:
                break
    return picks


def market_allocation(
    benchmark_to_signal_date: pd.Series,
    stock_prices_to_signal_date: pd.DataFrame,
    picks: list[str],
    strategy: ResearchStrategy,
) -> float:
    allocation = 1.0
    benchmark = benchmark_to_signal_date.dropna()
    if strategy.market_sma_days:
        if len(benchmark) <= strategy.market_sma_days:
            return 0.0
        moving_average = benchmark.iloc[-strategy.market_sma_days:].mean()
        if benchmark.iloc[-1] < moving_average:
            allocation = 0.0
    if strategy.market_return_days:
        if len(benchmark) <= strategy.market_return_days:
            return 0.0
        if benchmark.iloc[-1] / benchmark.iloc[-strategy.market_return_days - 1] - 1 < 0:
            allocation = 0.0
    if allocation == 0.0 or strategy.target_volatility is None or not picks:
        return allocation

    returns = stock_prices_to_signal_date[picks].pct_change(fill_method=None).dropna(how="all")
    basket_returns = returns.tail(strategy.volatility_lookback_days).mean(axis=1).dropna()
    realized_volatility = float(basket_returns.std() * np.sqrt(252))
    if not realized_volatility or np.isnan(realized_volatility):
        return allocation
    return min(allocation, strategy.target_volatility / realized_volatility)


def run_research_backtest(
    prices: pd.DataFrame,
    universe: list[str],
    strategy: ResearchStrategy,
    start: str = BACKTEST_START,
    benchmark: str = BENCHMARK,
    starting_capital: float = STARTING_CAPITAL,
    trade_cost_bps: float = TRADE_COST_BPS,
    rebalance_frequency: str = "monthly",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    benchmark_prices = prices[benchmark].dropna()
    stock_prices = prices[[ticker for ticker in universe if ticker in prices.columns]].copy()
    stock_prices = stock_prices.dropna(axis=1, how="all").dropna(axis=0, how="all")
    stock_returns = stock_prices.pct_change(fill_method=None).fillna(0)
    if rebalance_frequency == "weekly":
        rebalance_dates = set(trading_days_after_week_end(stock_prices.index, start))
    else:
        rebalance_dates = set(first_trading_days_after_month_end(stock_prices.index, start))

    positions = pd.Series(0.0, index=stock_prices.columns)
    value = starting_capital
    cash = starting_capital
    previous_date: pd.Timestamp | None = None
    equity_rows = []
    rebalance_rows = []

    for date, daily_returns in stock_returns.loc[start:].iterrows():
        if previous_date is not None:
            positions = positions * (1 + daily_returns.fillna(0))
            value = float(positions.sum() + cash)

        if date in rebalance_dates:
            signal_position = stock_prices.index.get_loc(date) - 1
            if signal_position >= strategy.lookback_days + strategy.skip_recent_days:
                stock_history = stock_prices.iloc[: signal_position + 1]
                benchmark_history = benchmark_prices.loc[: stock_history.index[-1]]
                ranked = rank_tickers(stock_history, strategy)
                picks = select_tickers(ranked, strategy) if not ranked.empty else []
                if len(picks) >= strategy.picks:
                    allocation = market_allocation(benchmark_history, stock_history, picks, strategy)
                    target_weights = pd.Series(0.0, index=stock_prices.columns)
                    if allocation > 0:
                        target_weights[picks] = allocation / len(picks)
                    target_positions = target_weights * value
                    traded_value = float((target_positions - positions).abs().sum())
                    transaction_cost = traded_value * trade_cost_bps / 10_000
                    value -= transaction_cost
                    positions = target_weights * value
                    cash = float(value - positions.sum())
                    turnover = traded_value / value if value else 0.0
                    rebalance_rows.append(
                        {
                            "date": date,
                            "signal_date": stock_history.index[-1],
                            "strategy": strategy.name,
                            "holdings": len(picks) if allocation else 0,
                            "stock_slots": len(picks),
                            "cash_weight": cash / value if value else 0.0,
                            "equity_allocation": allocation,
                            "max_weight": float((positions / value).max()) if value else 0.0,
                            "turnover": turnover,
                            "transaction_cost": transaction_cost,
                            "risk_group_counts": json.dumps(risk_group_counts(picks), sort_keys=True),
                            "tickers": ",".join(picks),
                        }
                    )

        weights = positions / value if value else positions
        equity_rows.append(
            {
                "date": date,
                "strategy": value,
                "cash_weight": float(cash / value) if value else 0.0,
                "holdings": int((positions > 0).sum()),
                "max_weight": float(weights.max()) if len(weights) else 0.0,
            }
        )
        previous_date = date

    equity = pd.DataFrame(equity_rows).set_index("date")
    benchmark_equity = benchmark_prices.loc[equity.index[0] : equity.index[-1]].reindex(equity.index).ffill()
    benchmark_equity = benchmark_equity / benchmark_equity.iloc[0] * starting_capital
    equity["benchmark"] = benchmark_equity
    equity = equity.dropna(subset=["benchmark"])
    rebalances = pd.DataFrame(rebalance_rows)
    strategy_perf = performance(equity["strategy"])
    benchmark_perf = performance(equity["benchmark"])
    invested = equity[equity["holdings"] > 0]
    summary = {
        "strategy": strategy.name,
        "description": strategy.description,
        "config": asdict(strategy),
        "rebalance_frequency": rebalance_frequency,
        "backtest_start": str(equity.index.min().date()),
        "backtest_end": str(equity.index.max().date()),
        "strategy_performance": asdict(strategy_perf),
        "benchmark_performance": asdict(benchmark_perf),
        "excess_cagr": strategy_perf.cagr - benchmark_perf.cagr,
        "final_value_excess": strategy_perf.final_value - benchmark_perf.final_value,
        "average_cash_weight": float(equity["cash_weight"].mean()),
        "rebalance_count": int(len(rebalances)),
        "average_turnover_per_rebalance": float(rebalances["turnover"].mean()) if len(rebalances) else 0.0,
        "max_turnover_per_rebalance": float(rebalances["turnover"].max()) if len(rebalances) else 0.0,
        "min_active_holdings": int(invested["holdings"].min()) if len(invested) else 0,
        "max_observed_weight": float(equity["max_weight"].max()),
        "constraints_pass": bool(
            (len(invested) == 0 or invested["holdings"].min() >= MIN_HOLDINGS)
            and equity["max_weight"].max() <= MAX_POSITION_WEIGHT
        ),
        "beats_vfiax_by_one_percent_annually": bool(strategy_perf.cagr - benchmark_perf.cagr >= 0.01),
    }
    return equity, rebalances, summary


def default_strategies() -> list[ResearchStrategy]:
    return [
        ResearchStrategy(
            name="baseline_126_momentum",
            description="Original 126-trading-day adjusted-price momentum, equal-weight top 20.",
        ),
        ResearchStrategy(
            name="risk_adjusted_126",
            description="Current live ranking style: 126-day momentum penalized for volatility and drawdown.",
            volatility_penalty=0.55,
            drawdown_penalty=0.75,
            max_semiconductor_count=8,
        ),
        ResearchStrategy(
            name="risk_adjusted_252_skip21",
            description="Longer 12-1 style momentum signal, skipping the most recent trading month.",
            lookback_days=252,
            skip_recent_days=21,
            volatility_penalty=0.55,
            drawdown_penalty=0.75,
            max_semiconductor_count=8,
        ),
        ResearchStrategy(
            name="semiconductor_cap6",
            description="Risk-adjusted 126-day momentum with a stricter six-name semiconductor/hardware cap.",
            volatility_penalty=0.55,
            drawdown_penalty=0.75,
            max_semiconductor_count=6,
        ),
        ResearchStrategy(
            name="vol_target_18",
            description="Risk-adjusted momentum with equity exposure scaled to 18% realized basket volatility.",
            volatility_penalty=0.55,
            drawdown_penalty=0.75,
            max_semiconductor_count=8,
            target_volatility=0.18,
        ),
        ResearchStrategy(
            name="dual_momentum_cash",
            description="Risk-adjusted momentum only when VFIAX is above its 200-day average and has positive 126-day return; otherwise cash.",
            volatility_penalty=0.55,
            drawdown_penalty=0.75,
            max_semiconductor_count=8,
            market_sma_days=200,
            market_return_days=126,
        ),
    ]


def write_research_outputs(
    results: list[dict],
    equities: dict[str, pd.DataFrame],
    rebalances: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_dir / "strategy_comparison.csv", index=False)
    (output_dir / "strategy_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    for name, equity in equities.items():
        equity.to_csv(output_dir / f"{name}_equity.csv")
    for name, rebalance in rebalances.items():
        rebalance.to_csv(output_dir / f"{name}_rebalances.csv", index=False)


def flattened_summary(summary: dict) -> dict:
    strategy_perf = summary["strategy_performance"]
    benchmark_perf = summary["benchmark_performance"]
    return {
        "strategy": summary["strategy"],
        "description": summary["description"],
        "rebalance_frequency": summary["rebalance_frequency"],
        "final_value": strategy_perf["final_value"],
        "vfiax_final_value": benchmark_perf["final_value"],
        "final_value_excess": summary["final_value_excess"],
        "cagr": strategy_perf["cagr"],
        "vfiax_cagr": benchmark_perf["cagr"],
        "excess_cagr": summary["excess_cagr"],
        "annualized_volatility": strategy_perf["annualized_volatility"],
        "sharpe_0_rate": strategy_perf["sharpe_0_rate"],
        "max_drawdown": strategy_perf["max_drawdown"],
        "vfiax_max_drawdown": benchmark_perf["max_drawdown"],
        "average_cash_weight": summary["average_cash_weight"],
        "rebalance_count": summary["rebalance_count"],
        "average_turnover_per_rebalance": summary["average_turnover_per_rebalance"],
        "max_turnover_per_rebalance": summary["max_turnover_per_rebalance"],
        "min_active_holdings": summary["min_active_holdings"],
        "max_observed_weight": summary["max_observed_weight"],
        "constraints_pass": summary["constraints_pass"],
        "beats_vfiax_by_one_percent_annually": summary["beats_vfiax_by_one_percent_annually"],
    }


def print_results(results: list[dict]) -> None:
    ranked = sorted(results, key=lambda row: (row["excess_cagr"], -abs(row["max_drawdown"])), reverse=True)
    for row in ranked:
        print(
            row["strategy"],
            f"CAGR {row['cagr'] * 100:.2f}%",
            f"excess {row['excess_cagr'] * 100:.2f}%",
            f"maxDD {row['max_drawdown'] * 100:.2f}%",
            f"final ${row['final_value']:,.0f}",
            f"turnover {row['average_turnover_per_rebalance'] * 100:.1f}%",
            f"cash {row['average_cash_weight'] * 100:.1f}%",
            f"constraints {row['constraints_pass']}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare candidate Stock Solve algorithms.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research_algorithms"))
    parser.add_argument("--cache", type=Path, default=Path("data/adjusted_closes.csv"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--frequency", choices=["monthly", "weekly"], default="monthly")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_path = None if args.no_cache else args.cache
    prices = download_adjusted_closes(UNIVERSE + [BENCHMARK], DATA_START, BACKTEST_END, cache_path)
    results = []
    equities: dict[str, pd.DataFrame] = {}
    rebalances: dict[str, pd.DataFrame] = {}
    for strategy in default_strategies():
        equity, rebalance, summary = run_research_backtest(
            prices,
            UNIVERSE,
            strategy,
            rebalance_frequency=args.frequency,
        )
        row = flattened_summary(summary)
        results.append(row)
        equities[strategy.name] = equity
        rebalances[strategy.name] = rebalance
    write_research_outputs(results, equities, rebalances, args.output_dir)
    print_results(results)
    print("Wrote:", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
