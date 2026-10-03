from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stock_solve.backtest import (
    MAX_POSITION_WEIGHT,
    MIN_HOLDINGS,
    MOMENTUM_LOOKBACK_DAYS,
    STARTING_CAPITAL,
    download_adjusted_closes,
)


CURRENT_NASDAQ_100 = [
    "NVDA",
    "GOOGL",
    "GOOG",
    "AAPL",
    "MSFT",
    "AMZN",
    "AVGO",
    "TSLA",
    "META",
    "MU",
    "WMT",
    "AMD",
    "ASML",
    "INTC",
    "CSCO",
    "COST",
    "LRCX",
    "ARM",
    "PLTR",
    "NFLX",
    "AMAT",
    "TXN",
    "QCOM",
    "KLAC",
    "LIN",
    "PANW",
    "APP",
    "TMUS",
    "ADI",
    "STX",
    "PEP",
    "CRWD",
    "WDC",
    "AMGN",
    "MRVL",
    "GILD",
    "SHOP",
    "HON",
    "ISRG",
    "BKNG",
    "PDD",
    "VRTX",
    "SBUX",
    "ADBE",
    "CEG",
    "CDNS",
    "FTNT",
    "MAR",
    "SNPS",
    "INTU",
    "CMCSA",
    "ADP",
    "DDOG",
    "MNST",
    "MELI",
    "CSX",
    "NXPI",
    "ABNB",
    "MDLZ",
    "MPWR",
    "ROST",
    "ORLY",
    "DASH",
    "AEP",
    "CTAS",
    "WBD",
    "BKR",
    "REGN",
    "PCAR",
    "MSTR",
    "FANG",
    "MCHP",
    "FAST",
    "EA",
    "XEL",
    "FER",
    "ADSK",
    "ODFL",
    "EXC",
    "IDXX",
    "TTWO",
    "KDP",
    "CCEP",
    "ALNY",
    "PYPL",
    "TRI",
    "AXON",
    "WDAY",
    "PAYX",
    "ROP",
    "CPRT",
    "KHC",
    "DXCM",
    "GEHC",
    "TEAM",
    "CTSH",
    "INSM",
    "VRSK",
    "ZS",
    "CHTR",
    "CSGP",
]

RISK_GROUP_BY_TICKER = {
    "ADI": "semiconductor_hardware",
    "AMD": "semiconductor_hardware",
    "AMAT": "semiconductor_hardware",
    "ARM": "semiconductor_hardware",
    "ASML": "semiconductor_hardware",
    "INTC": "semiconductor_hardware",
    "KLAC": "semiconductor_hardware",
    "LRCX": "semiconductor_hardware",
    "MCHP": "semiconductor_hardware",
    "MPWR": "semiconductor_hardware",
    "MRVL": "semiconductor_hardware",
    "MU": "semiconductor_hardware",
    "NVDA": "semiconductor_hardware",
    "NXPI": "semiconductor_hardware",
    "QCOM": "semiconductor_hardware",
    "STX": "semiconductor_hardware",
    "TXN": "semiconductor_hardware",
    "WDC": "semiconductor_hardware",
}


@dataclass(frozen=True)
class PaperPortfolioConfig:
    capital: float = STARTING_CAPITAL
    lookback_days: int = MOMENTUM_LOOKBACK_DAYS
    picks: int = MIN_HOLDINGS
    max_position_weight: float = MAX_POSITION_WEIGHT
    data_start: str = "2025-01-01"
    data_end: str | None = None
    entry_rank: int = MIN_HOLDINGS
    keep_until_rank: int = 30
    confirmation_signals: int = 2
    replacement_momentum_hurdle: float = 0.03
    rebalance_weekday: int = 4
    max_risk_group_count: int = 8
    volatility_penalty: float = 0.55
    drawdown_penalty: float = 0.75


def momentum_table(
    prices: pd.DataFrame,
    universe: list[str],
    lookback_days: int,
    volatility_penalty: float = 0.55,
    drawdown_penalty: float = 0.75,
) -> pd.DataFrame:
    stock_prices = prices[[ticker for ticker in universe if ticker in prices.columns]].dropna(
        axis=1, how="all"
    )
    stock_prices = stock_prices.dropna(axis=0, how="all")
    if len(stock_prices) < lookback_days + 1:
        raise ValueError("Not enough price history for momentum ranking.")

    current = stock_prices.iloc[-1]
    past = stock_prices.iloc[-lookback_days - 1]
    momentum = (current / past - 1).replace([np.inf, -np.inf], np.nan).dropna()
    lookback_window = stock_prices.iloc[-lookback_days - 1 :]
    annualized_volatility = lookback_window.pct_change(fill_method=None).std() * np.sqrt(252)
    drawdown_from_high = (current / lookback_window.cummax().max() - 1).replace(
        [np.inf, -np.inf], np.nan
    )
    valid_scores: dict[str, float] = {}
    valid_momentum: dict[str, float] = {}
    valid_volatility: dict[str, float] = {}
    valid_drawdown: dict[str, float] = {}
    for ticker, score in momentum.items():
        history = stock_prices[ticker].dropna()
        if (
            len(history) >= lookback_days + 1
            and pd.notna(current.get(ticker))
            and history.index[-1] == stock_prices.index[-1]
        ):
            volatility = float(annualized_volatility.get(ticker, np.nan))
            drawdown = float(drawdown_from_high.get(ticker, np.nan))
            if np.isnan(volatility) or np.isnan(drawdown):
                continue
            valid_momentum[ticker] = float(score)
            valid_volatility[ticker] = volatility
            valid_drawdown[ticker] = drawdown
            valid_scores[ticker] = float(
                score - volatility_penalty * volatility + drawdown_penalty * drawdown
            )

    ranked = (
        pd.DataFrame(
            {
                "ticker": list(valid_scores),
                "lookback_momentum": [valid_momentum[ticker] for ticker in valid_scores],
                "annualized_volatility": [valid_volatility[ticker] for ticker in valid_scores],
                "drawdown_from_high": [valid_drawdown[ticker] for ticker in valid_scores],
                "ranking_score": list(valid_scores.values()),
            }
        )
        .sort_values("ranking_score", ascending=False)
        .reset_index(drop=True)
    )
    ranked["rank"] = ranked.index + 1
    return ranked


def risk_group(ticker: str) -> str:
    return RISK_GROUP_BY_TICKER.get(ticker, ticker)


def risk_group_counts(tickers: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ticker in tickers:
        group = risk_group(ticker)
        counts[group] = counts.get(group, 0) + 1
    return counts


def diversified_ranked_tickers(
    ranked_tickers: list[str],
    picks: int,
    max_risk_group_count: int,
    starting_tickers: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    target = list(dict.fromkeys(starting_tickers or []))
    skipped_for_group: list[str] = []
    counts = risk_group_counts(target)
    for ticker in ranked_tickers:
        if ticker in target:
            continue
        group = risk_group(ticker)
        if counts.get(group, 0) >= max_risk_group_count:
            skipped_for_group.append(ticker)
            continue
        target.append(ticker)
        counts[group] = counts.get(group, 0) + 1
        if len(target) == picks:
            break
    if len(target) < picks:
        for ticker in ranked_tickers:
            if ticker in target:
                continue
            target.append(ticker)
            if len(target) == picks:
                break
    return target[:picks], skipped_for_group


def confirmed_entry_tickers(
    prices: pd.DataFrame, universe: list[str], config: PaperPortfolioConfig
) -> set[str]:
    if config.confirmation_signals <= 1:
        latest = momentum_table(
            prices,
            universe,
            config.lookback_days,
            config.volatility_penalty,
            config.drawdown_penalty,
        )
        return set(latest.head(config.entry_rank)["ticker"])

    stock_prices = prices[[ticker for ticker in universe if ticker in prices.columns]].dropna(
        axis=1, how="all"
    )
    stock_prices = stock_prices.dropna(axis=0, how="all")
    confirmed: set[str] | None = None
    for offset in range(config.confirmation_signals):
        end = len(stock_prices) - offset
        if end <= config.lookback_days:
            break
        ranked = momentum_table(
            stock_prices.iloc[:end],
            universe,
            config.lookback_days,
            config.volatility_penalty,
            config.drawdown_penalty,
        )
        top = set(ranked.head(config.entry_rank)["ticker"])
        confirmed = top if confirmed is None else confirmed & top
    return confirmed or set()


def optimized_target_tickers(
    prices: pd.DataFrame,
    universe: list[str],
    config: PaperPortfolioConfig,
    current_tickers: list[str] | None = None,
) -> tuple[list[str], pd.DataFrame, dict]:
    current_tickers = list(dict.fromkeys(current_tickers or []))
    ranked = momentum_table(
        prices,
        universe,
        config.lookback_days,
        config.volatility_penalty,
        config.drawdown_penalty,
    )
    rank_by_ticker = ranked.set_index("ticker")["rank"].to_dict()
    score_by_ticker = ranked.set_index("ticker")["ranking_score"].to_dict()
    confirmed = confirmed_entry_tickers(prices, universe, config)
    ranked_tickers = ranked["ticker"].tolist()

    if not current_tickers:
        target, skipped_for_group = diversified_ranked_tickers(
            ranked_tickers, config.picks, config.max_risk_group_count
        )
        selection = {
            "mode": "initial_top_ranked",
            "kept": [],
            "replaced": [],
            "risk_group_counts": risk_group_counts(target),
            "skipped_for_risk_group_cap": skipped_for_group,
            "entry_candidates_confirmed": sorted(confirmed),
        }
        return target, ranked, selection

    target = [ticker for ticker in current_tickers if ticker in rank_by_ticker]
    replaced = []
    candidates = [
        ticker
        for ticker in ranked.head(config.entry_rank)["ticker"].tolist()
        if ticker not in target and ticker in confirmed
    ]

    for holding in list(target):
        rank = rank_by_ticker.get(holding, 10_000)
        if rank <= config.keep_until_rank:
            continue
        if not candidates:
            continue
        candidate = candidates[0]
        holding_score = score_by_ticker.get(holding, -float("inf"))
        candidate_score = score_by_ticker[candidate]
        if candidate_score >= holding_score + config.replacement_momentum_hurdle:
            index = target.index(holding)
            target[index] = candidate
            candidates.pop(0)
            replaced.append(
                {
                    "drop": holding,
                    "add": candidate,
                    "drop_rank": rank,
                    "add_rank": rank_by_ticker[candidate],
                    "drop_score": holding_score,
                    "add_score": candidate_score,
                }
            )

    risk_group_trimmed = []
    for group, count in risk_group_counts(target).items():
        while count > config.max_risk_group_count:
            group_tickers = [ticker for ticker in target if risk_group(ticker) == group]
            drop = max(group_tickers, key=lambda ticker: rank_by_ticker.get(ticker, 10_000))
            target.remove(drop)
            risk_group_trimmed.append(drop)
            count -= 1

    if len(target) < config.picks:
        for ticker in ranked_tickers:
            if ticker in target:
                continue
            if current_tickers and ticker not in confirmed:
                continue
            if risk_group_counts(target).get(risk_group(ticker), 0) >= config.max_risk_group_count:
                continue
            target.append(ticker)
            if len(target) == config.picks:
                break

    fallback_fill = []
    if len(target) < config.picks:
        for ticker in ranked_tickers:
            if ticker in target:
                continue
            if risk_group_counts(target).get(risk_group(ticker), 0) >= config.max_risk_group_count:
                continue
            target.append(ticker)
            fallback_fill.append(ticker)
            if len(target) == config.picks:
                break
    if len(target) < config.picks:
        for ticker in ranked_tickers:
            if ticker in target:
                continue
            target.append(ticker)
            fallback_fill.append(ticker)
            if len(target) == config.picks:
                break

    target = target[: config.picks]
    target = sorted(target, key=lambda ticker: rank_by_ticker.get(ticker, 10_000))
    selection = {
        "mode": "optimized_hysteresis",
        "kept": [ticker for ticker in target if ticker in current_tickers],
        "replaced": replaced,
        "fallback_fill": fallback_fill,
        "risk_group_counts": risk_group_counts(target),
        "risk_group_trimmed": risk_group_trimmed,
        "entry_candidates_confirmed": sorted(confirmed),
    }
    return target, ranked, selection


def build_paper_portfolio(
    prices: pd.DataFrame,
    universe: list[str],
    config: PaperPortfolioConfig,
    current_tickers: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    stock_prices = prices[[ticker for ticker in universe if ticker in prices.columns]].dropna(
        axis=1, how="all"
    )
    stock_prices = stock_prices.dropna(axis=0, how="all")
    picks, ranked, selection = optimized_target_tickers(stock_prices, universe, config, current_tickers)
    if len(picks) < MIN_HOLDINGS:
        raise ValueError(f"Only {len(picks)} valid picks found; need at least {MIN_HOLDINGS}.")

    signal_date = stock_prices.index[-1]
    latest = stock_prices.loc[signal_date, picks]
    ranked_by_ticker = ranked.set_index("ticker")
    selected_ranks = ranked_by_ticker.loc[picks]
    target_dollars = config.capital / len(picks)
    shares = np.floor(target_dollars / latest).astype(int)
    dollars = shares * latest
    weights = dollars / config.capital
    cash = config.capital - float(dollars.sum())

    portfolio = pd.DataFrame(
        {
            "ticker": picks,
            "signal_date": signal_date.date().isoformat(),
            "latest_adjusted_close": latest.values,
            "lookback_momentum": selected_ranks["lookback_momentum"].values,
            "annualized_volatility": selected_ranks["annualized_volatility"].values,
            "drawdown_from_high": selected_ranks["drawdown_from_high"].values,
            "ranking_score": selected_ranks["ranking_score"].values,
            "target_weight": 1 / len(picks),
            "shares": shares.values,
            "market_value": dollars.values,
            "actual_weight": weights.values,
        }
    )
    portfolio = portfolio.sort_values("ranking_score", ascending=False).reset_index(drop=True)
    summary = {
        "config": asdict(config),
        "source_universe": "Current Nasdaq-100 list from StockAnalysis.com page opened 2026-06-01",
        "requested_universe_size": len(universe),
        "priced_universe_size": int(len(stock_prices.columns)),
        "signal_date": signal_date.date().isoformat(),
        "capital": config.capital,
        "holdings": int(len(portfolio)),
        "cash_after_whole_share_rounding": cash,
        "cash_weight": cash / config.capital,
        "selection_rules": {
            "entry_rank": config.entry_rank,
            "keep_until_rank": config.keep_until_rank,
            "confirmation_signals": config.confirmation_signals,
            "replacement_momentum_hurdle": config.replacement_momentum_hurdle,
            "rebalance_weekday": config.rebalance_weekday,
            "max_risk_group_count": config.max_risk_group_count,
            "volatility_penalty": config.volatility_penalty,
            "drawdown_penalty": config.drawdown_penalty,
        },
        "selection": selection,
        "max_actual_weight": float(portfolio["actual_weight"].max()),
        "total_invested": float(portfolio["market_value"].sum()),
        "success_constraints": bool(
            len(portfolio) >= MIN_HOLDINGS and portfolio["actual_weight"].max() <= MAX_POSITION_WEIGHT
        ),
    }
    return portfolio, summary


def write_outputs(portfolio: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio.to_csv(output_dir / "paper_portfolio.csv", index=False)
    (output_dir / "paper_portfolio_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def print_portfolio(portfolio: pd.DataFrame, summary: dict) -> None:
    print("Signal date:", summary["signal_date"])
    print("Capital:", f"${summary['capital']:,.2f}")
    print("Holdings:", summary["holdings"])
    print("Total invested:", f"${summary['total_invested']:,.2f}")
    print("Cash after whole-share rounding:", f"${summary['cash_after_whole_share_rounding']:,.2f}")
    print("Max actual holding weight:", f"{summary['max_actual_weight'] * 100:.2f}%")
    print(portfolio.to_string(index=False, formatters={
        "latest_adjusted_close": "${:,.2f}".format,
        "lookback_momentum": "{:.2%}".format,
        "target_weight": "{:.2%}".format,
        "market_value": "${:,.2f}".format,
        "actual_weight": "{:.2%}".format,
    }))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a current paper portfolio.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache", type=Path, default=Path("data/current_nasdaq100_closes.csv"))
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PaperPortfolioConfig()
    cache_path = None if args.no_cache else args.cache
    prices = download_adjusted_closes(
        CURRENT_NASDAQ_100, config.data_start, config.data_end, cache_path
    )
    portfolio, summary = build_paper_portfolio(prices, CURRENT_NASDAQ_100, config)
    write_outputs(portfolio, summary, args.output_dir)
    from stock_solve.dashboard import generate_dashboard

    generate_dashboard(args.output_dir)
    print_portfolio(portfolio, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
