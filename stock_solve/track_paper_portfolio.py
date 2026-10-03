from __future__ import annotations

import argparse
import json
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from stock_solve.backtest import BENCHMARK, download_adjusted_closes
from stock_solve.live_portfolio import (
    CURRENT_NASDAQ_100,
    PaperPortfolioConfig,
    build_paper_portfolio,
)


PENDING_REBALANCE_FILE = "paper_portfolio_pending_rebalance.json"
MARKET_TIMEZONE = ZoneInfo("America/New_York")
MARKET_CLOSE_SETTLE_TIME = time(16, 15)


def load_current_portfolio(path: Path) -> tuple[pd.DataFrame, dict]:
    portfolio = pd.read_csv(path)
    summary_path = path.with_name("paper_portfolio_summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return portfolio, summary


def latest_available_prices_from_frame(prices: pd.DataFrame) -> pd.Series:
    usable = prices.dropna(axis=0, how="all")
    if usable.empty:
        return pd.Series(dtype=float)
    return usable.ffill().iloc[-1].dropna()


def completed_close_prices(
    prices: pd.DataFrame,
    now_utc: datetime | None = None,
) -> pd.DataFrame:
    """Exclude today's still-forming daily bar until after the market close settles."""
    usable = prices.dropna(axis=0, how="all")
    if usable.empty:
        return prices

    current_utc = now_utc or datetime.now(timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)
    market_now = current_utc.astimezone(MARKET_TIMEZONE)
    latest_date = pd.Timestamp(usable.index[-1]).date()
    if latest_date == market_now.date() and market_now.time() < MARKET_CLOSE_SETTLE_TIME:
        completed_mask = [
            pd.Timestamp(index_value).date() < market_now.date()
            for index_value in prices.index
        ]
        return prices.loc[completed_mask]
    return prices


def latest_prices(tickers: list[str]) -> pd.Series:
    prices = download_adjusted_closes(tickers, "2025-01-01", None, None)
    return latest_available_prices_from_frame(prices)


def download_adjusted_ohlc(tickers: list[str], start: str = "2025-01-01") -> dict[str, pd.DataFrame]:
    raw = yf.download(
        list(dict.fromkeys(tickers)),
        start=start,
        end=None,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return {}
    if isinstance(raw.columns, pd.MultiIndex):
        return {
            field: raw[field].dropna(axis=1, how="all").sort_index()
            for field in ["Open", "Close"]
            if field in raw.columns.get_level_values(0)
        }
    ticker = tickers[0]
    return {
        "Open": raw[["Open"]].rename(columns={"Open": ticker}).sort_index(),
        "Close": raw[["Close"]].rename(columns={"Close": ticker}).sort_index(),
    }


def load_pending_rebalance(output_dir: Path) -> dict | None:
    path = output_dir / PENDING_REBALANCE_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_pending_rebalance(pending: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PENDING_REBALANCE_FILE).write_text(json.dumps(pending, indent=2) + "\n")


def clear_pending_rebalance(output_dir: Path) -> None:
    path = output_dir / PENDING_REBALANCE_FILE
    if path.exists():
        path.unlink()


def pending_from_recommendation(
    recommended: pd.DataFrame,
    recommendation_summary: dict,
    current_tickers: list[str],
) -> dict | None:
    current_set = set(current_tickers)
    recommended_set = set(recommended["ticker"])
    adds = sorted(recommended_set - current_set)
    drops = sorted(current_set - recommended_set)
    if not adds and not drops:
        return None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_date": recommendation_summary["signal_date"],
        "target_tickers": recommended["ticker"].tolist(),
        "adds": adds,
        "drops": drops,
        "momentum": {
            ticker: float(momentum)
            for ticker, momentum in zip(recommended["ticker"], recommended["lookback_momentum"])
        },
        "ranking_score": {
            ticker: float(score)
            for ticker, score in zip(
                recommended["ticker"],
                recommended.get("ranking_score", recommended["lookback_momentum"]),
            )
        },
        "annualized_volatility": {
            ticker: float(volatility)
            for ticker, volatility in zip(
                recommended["ticker"],
                recommended.get("annualized_volatility", recommended["lookback_momentum"] * 0),
            )
        },
        "drawdown_from_high": {
            ticker: float(drawdown)
            for ticker, drawdown in zip(
                recommended["ticker"],
                recommended.get("drawdown_from_high", recommended["lookback_momentum"] * 0),
            )
        },
        "execution_rule": "weekly gated, validate at open, then execute at first available adjusted open after signal_date",
    }


def latest_recommendation_through(
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    capital: float,
    current_tickers: list[str],
) -> tuple[pd.DataFrame, dict]:
    signal_prices = prices.loc[:signal_date]
    return build_paper_portfolio(
        signal_prices,
        CURRENT_NASDAQ_100,
        PaperPortfolioConfig(capital=capital),
        current_tickers=current_tickers,
    )


def recommendation_at_open(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    execution_date: pd.Timestamp,
    capital: float,
    current_tickers: list[str],
) -> tuple[pd.DataFrame, dict]:
    close_history = closes.loc[closes.index < execution_date, CURRENT_NASDAQ_100]
    open_row = opens.loc[[execution_date], CURRENT_NASDAQ_100]
    open_signal_prices = pd.concat([close_history, open_row])
    recommended, summary = build_paper_portfolio(
        open_signal_prices,
        CURRENT_NASDAQ_100,
        PaperPortfolioConfig(capital=capital),
        current_tickers=current_tickers,
    )
    summary = dict(summary)
    summary["signal_date"] = execution_date.date().isoformat()
    summary["signal_price_type"] = "execution_open"
    return recommended, summary


def save_rebalanced_portfolio(
    recommended: pd.DataFrame, recommendation_summary: dict, output_dir: Path
) -> None:
    recommended.to_csv(output_dir / "paper_portfolio.csv", index=False)
    (output_dir / "paper_portfolio_summary.json").write_text(
        json.dumps(recommendation_summary, indent=2) + "\n"
    )


def is_rebalance_signal_day(signal_date: str, config: PaperPortfolioConfig) -> bool:
    return pd.Timestamp(signal_date).weekday() == config.rebalance_weekday


def same_target_tickers(left: list[str], right: list[str]) -> bool:
    return set(left) == set(right)


def execute_pending_rebalance_if_ready(
    portfolio: pd.DataFrame,
    initial_summary: dict,
    pending: dict | None,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict, dict | None, dict | None]:
    if not pending:
        return portfolio, initial_summary, None, None

    target_tickers = pending["target_tickers"]
    held = portfolio["ticker"].tolist()
    all_tickers = sorted(set(held + target_tickers + CURRENT_NASDAQ_100))
    ohlc = download_adjusted_ohlc(all_tickers, "2025-01-01")
    opens = ohlc.get("Open")
    closes = ohlc.get("Close")
    if opens is None or closes is None:
        return portfolio, initial_summary, None, None

    signal_date = pd.Timestamp(pending["signal_date"])
    execution_dates = opens.index[opens.index > signal_date]
    if execution_dates.empty:
        return portfolio, initial_summary, None, None

    execution_date = execution_dates[0]
    completed_closes = closes.index[closes.index < execution_date]
    validation_event = None
    if not completed_closes.empty and completed_closes[-1] > signal_date:
        validation_date = completed_closes[-1]
        validation_capital = float(initial_summary.get("capital", 250_000.0))
        recommended, recommendation_summary = latest_recommendation_through(
            closes[CURRENT_NASDAQ_100], validation_date, validation_capital, held
        )
        validation_pending = pending_from_recommendation(recommended, recommendation_summary, held)
        if validation_pending and not same_target_tickers(
            validation_pending["target_tickers"], target_tickers
        ):
            save_pending_rebalance(validation_pending, output_dir)
            validation_event = {
                "pending_rebalance_revalidated": False,
                "pending_rebalance_replaced": True,
                "stale_signal_date": pending["signal_date"],
                "validation_signal_date": recommendation_summary["signal_date"],
                "replacement_adds": validation_pending["adds"],
                "replacement_drops": validation_pending["drops"],
            }
            return portfolio, initial_summary, None, validation_event
        if validation_pending is None:
            clear_pending_rebalance(output_dir)
            validation_event = {
                "pending_rebalance_revalidated": False,
                "pending_rebalance_replaced": False,
                "pending_rebalance_cancelled": True,
                "stale_signal_date": pending["signal_date"],
                "validation_signal_date": recommendation_summary["signal_date"],
            }
            return portfolio, initial_summary, None, validation_event
        validation_event = {
            "pending_rebalance_revalidated": True,
            "pending_rebalance_replaced": False,
            "validation_signal_date": recommendation_summary["signal_date"],
        }

    open_prices = opens.loc[execution_date]
    if open_prices.reindex(target_tickers).isna().any() or (
        held and open_prices.reindex(held).isna().any()
    ):
        return portfolio, initial_summary, None, validation_event

    cash = float(initial_summary.get("cash_after_whole_share_rounding", 0.0))
    current_positions = portfolio.set_index("ticker")
    if current_positions.empty:
        held_open_value = 0.0
    else:
        held_open_value = float(
            (current_positions["shares"] * open_prices.reindex(current_positions.index)).sum()
        )
    execution_capital = held_open_value + cash

    open_recommended, open_recommendation_summary = recommendation_at_open(
        closes, opens, execution_date, execution_capital, held
    )
    open_pending = pending_from_recommendation(open_recommended, open_recommendation_summary, held)
    if open_pending is None:
        clear_pending_rebalance(output_dir)
        validation_event = {
            **(validation_event or {}),
            "open_validation_passed": False,
            "open_validation_cancelled": True,
            "open_validation_signal_date": open_recommendation_summary["signal_date"],
        }
        return portfolio, initial_summary, None, validation_event
    if not same_target_tickers(open_pending["target_tickers"], target_tickers):
        target_tickers = open_pending["target_tickers"]
        pending = open_pending
        validation_event = {
            **(validation_event or {}),
            "open_validation_passed": True,
            "open_validation_adjusted": True,
            "open_validation_signal_date": open_recommendation_summary["signal_date"],
            "open_validation_adds": open_pending["adds"],
            "open_validation_drops": open_pending["drops"],
        }
    else:
        validation_event = {
            **(validation_event or {}),
            "open_validation_passed": True,
            "open_validation_adjusted": False,
            "open_validation_signal_date": open_recommendation_summary["signal_date"],
        }

    target_dollars = execution_capital / len(target_tickers)
    target_prices = open_prices.reindex(target_tickers)
    if target_prices.isna().any():
        return portfolio, initial_summary, None, validation_event
    shares = np.floor(target_dollars / target_prices).astype(int)
    market_value = shares * target_prices
    remaining_cash = execution_capital - float(market_value.sum())
    latest_close_date = closes.dropna(axis=0, how="all").index[-1]
    latest_close = closes.loc[latest_close_date].reindex(target_tickers)

    rebalanced = pd.DataFrame(
        {
            "ticker": target_tickers,
            "signal_date": pending["signal_date"],
            "execution_date": execution_date.date().isoformat(),
            "execution_price_type": "next_open",
            "latest_adjusted_close": target_prices.values,
            "lookback_momentum": [pending["momentum"].get(ticker, 0.0) for ticker in target_tickers],
            "annualized_volatility": [
                pending.get("annualized_volatility", {}).get(ticker, 0.0)
                for ticker in target_tickers
            ],
            "drawdown_from_high": [
                pending.get("drawdown_from_high", {}).get(ticker, 0.0)
                for ticker in target_tickers
            ],
            "ranking_score": [
                pending.get("ranking_score", pending.get("momentum", {})).get(ticker, 0.0)
                for ticker in target_tickers
            ],
            "target_weight": 1 / len(target_tickers),
            "shares": shares.values,
            "market_value": market_value.values,
            "actual_weight": (market_value / execution_capital).values,
        }
    ).sort_values("ranking_score", ascending=False)

    new_summary = {
        "config": initial_summary.get("config", {}),
        "source_universe": initial_summary.get("source_universe"),
        "requested_universe_size": initial_summary.get("requested_universe_size"),
        "priced_universe_size": initial_summary.get("priced_universe_size"),
        "signal_date": pending["signal_date"],
        "execution_date": execution_date.date().isoformat(),
        "execution_price_type": "next_open",
        "capital": execution_capital,
        "holdings": int(len(rebalanced)),
        "cash_after_whole_share_rounding": remaining_cash,
        "cash_weight": remaining_cash / execution_capital if execution_capital else 0.0,
        "benchmark": initial_summary.get("benchmark"),
        "benchmark_reset_date": initial_summary.get("benchmark_reset_date"),
        "benchmark_reset_price": initial_summary.get("benchmark_reset_price"),
        "benchmark_reset_value": initial_summary.get("benchmark_reset_value"),
        "portfolio_reset_value": initial_summary.get(
            "portfolio_reset_value", initial_summary.get("benchmark_reset_value")
        ),
        "max_actual_weight": float(rebalanced["actual_weight"].max()),
        "total_invested": float(rebalanced["market_value"].sum()),
        "success_constraints": bool(
            len(rebalanced) >= 20 and rebalanced["actual_weight"].max() <= 0.15
        ),
    }
    save_rebalanced_portfolio(rebalanced, new_summary, output_dir)
    clear_pending_rebalance(output_dir)

    execution = {
        "executed_signal_date": pending["signal_date"],
        "execution_date": execution_date.date().isoformat(),
        "execution_price_type": "next_open",
        "validated_before_execution": bool(validation_event and validation_event.get("pending_rebalance_revalidated")),
        "validation_signal_date": validation_event.get("validation_signal_date") if validation_event else pending["signal_date"],
        "open_validation_passed": bool(validation_event and validation_event.get("open_validation_passed")),
        "open_validation_adjusted": bool(validation_event and validation_event.get("open_validation_adjusted")),
        "open_validation_signal_date": validation_event.get("open_validation_signal_date") if validation_event else None,
        "executed_adds": pending.get("adds", []),
        "executed_drops": pending.get("drops", []),
        "execution_capital": execution_capital,
        "cash_after_execution": remaining_cash,
        "latest_close_date_after_execution": latest_close_date.date().isoformat(),
        "latest_close_value_after_execution": float((shares * latest_close).sum() + remaining_cash),
    }
    return rebalanced, new_summary, execution, validation_event


def track_portfolio(portfolio_path: Path, output_dir: Path, rebalance: bool = False) -> dict:
    portfolio, initial_summary = load_current_portfolio(portfolio_path)
    pending_at_start = load_pending_rebalance(output_dir)
    portfolio, initial_summary, execution, validation_event = execute_pending_rebalance_if_ready(
        portfolio, initial_summary, pending_at_start, output_dir
    )
    held = portfolio["ticker"].tolist()
    prices = latest_prices((held if held else []) + [BENCHMARK])

    portfolio["current_adjusted_close"] = portfolio["ticker"].map(prices)
    portfolio["current_market_value"] = portfolio["shares"] * portfolio["current_adjusted_close"]
    initial_cash = float(initial_summary.get("cash_after_whole_share_rounding", 0.0))
    portfolio_value = float(portfolio["current_market_value"].sum() + initial_cash)
    capital = float(initial_summary.get("capital", 250_000.0))
    reset_basis = float(
        initial_summary.get(
            "portfolio_reset_value",
            initial_summary.get("benchmark_reset_value", capital),
        )
    )

    current_universe_prices = download_adjusted_closes(CURRENT_NASDAQ_100, "2025-01-01", None, None)
    latest_date = current_universe_prices.dropna(axis=0, how="all").index[-1].date().isoformat()
    signal_universe_prices = completed_close_prices(current_universe_prices)
    live_config = PaperPortfolioConfig(capital=portfolio_value)
    recommended, recommendation_summary = build_paper_portfolio(
        signal_universe_prices,
        CURRENT_NASDAQ_100,
        live_config,
        current_tickers=held,
    )
    current_set = set(held)
    recommended_set = set(recommended["ticker"])
    adds = sorted(recommended_set - current_set)
    drops = sorted(current_set - recommended_set)
    pending_created = False
    pending_refreshed = False
    pending_after_run = load_pending_rebalance(output_dir)
    rebalance_signal_allowed = (not held) or is_rebalance_signal_day(
        recommendation_summary["signal_date"], live_config
    )
    rebalance_deferred_until_weekly_gate = bool(
        rebalance and (adds or drops) and not pending_after_run and not rebalance_signal_allowed
    )

    if rebalance and (adds or drops) and not pending_after_run and rebalance_signal_allowed:
        pending = pending_from_recommendation(recommended, recommendation_summary, held)
        save_pending_rebalance(pending, output_dir)
        pending_after_run = pending
        pending_created = True
    elif (
        rebalance
        and (adds or drops)
        and pending_after_run
        and rebalance_signal_allowed
        and pending_after_run.get("signal_date") == recommendation_summary["signal_date"]
    ):
        pending = pending_from_recommendation(recommended, recommendation_summary, held)
        save_pending_rebalance(pending, output_dir)
        pending_after_run = pending
        pending_refreshed = True

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "latest_price_date": latest_date,
        "signal_price_date": recommendation_summary["signal_date"],
        "initial_signal_date": initial_summary.get("signal_date"),
        "initial_capital": capital,
        "portfolio_value": portfolio_value,
        "portfolio_return": portfolio_value / capital - 1,
        "portfolio_reset_value": reset_basis,
        "portfolio_return_since_reset": portfolio_value / reset_basis - 1 if reset_basis else 0.0,
        "portfolio_return_since_trade_basis": portfolio_value / capital - 1,
        "cash": initial_cash,
        "benchmark": initial_summary.get("benchmark"),
        "benchmark_reset_date": initial_summary.get("benchmark_reset_date"),
        "benchmark_reset_price": initial_summary.get("benchmark_reset_price"),
        "benchmark_reset_value": initial_summary.get("benchmark_reset_value"),
        "holdings": int(len(portfolio)),
        "max_current_weight": float(
            (portfolio["current_market_value"] / portfolio_value).max()
        ) if len(portfolio) and portfolio_value else 0.0,
        "recommended_signal_date": recommendation_summary["signal_date"],
        "recommended_adds": adds,
        "recommended_drops": drops,
        "rebalance_needed": bool(adds or drops),
        "rebalance_signal_allowed": rebalance_signal_allowed,
        "rebalance_deferred_until_weekly_gate": rebalance_deferred_until_weekly_gate,
        "rebalance_pending_created": pending_created,
        "rebalance_pending_refreshed": pending_refreshed,
        "pending_rebalance_revalidated": bool(
            validation_event and validation_event.get("pending_rebalance_revalidated")
        ),
        "pending_rebalance_replaced": bool(
            validation_event and validation_event.get("pending_rebalance_replaced")
        ),
        "pending_rebalance_cancelled": bool(
            validation_event and validation_event.get("pending_rebalance_cancelled")
        ),
        "open_validation_passed": bool(
            execution and execution.get("open_validation_passed")
        ),
        "open_validation_adjusted": bool(
            execution and execution.get("open_validation_adjusted")
        ),
        "open_validation_cancelled": bool(
            validation_event and validation_event.get("open_validation_cancelled")
        ),
        "open_validation_signal_date": (
            execution.get("open_validation_signal_date")
            if execution
            else (validation_event.get("open_validation_signal_date") if validation_event else None)
        ),
        "validation_signal_date": (
            validation_event.get("validation_signal_date") if validation_event else None
        ),
        "replacement_adds": (
            validation_event.get("replacement_adds", []) if validation_event else []
        ),
        "replacement_drops": (
            validation_event.get("replacement_drops", []) if validation_event else []
        ),
        "pending_rebalance": bool(pending_after_run),
        "pending_signal_date": pending_after_run.get("signal_date") if pending_after_run else None,
        "pending_adds": pending_after_run.get("adds", []) if pending_after_run else [],
        "pending_drops": pending_after_run.get("drops", []) if pending_after_run else [],
        "rebalance_executed": bool(execution),
        "executed_signal_date": execution.get("executed_signal_date") if execution else None,
        "execution_date": execution.get("execution_date") if execution else None,
        "execution_price_type": execution.get("execution_price_type") if execution else None,
        "validated_before_execution": execution.get("validated_before_execution") if execution else False,
        "executed_adds": execution.get("executed_adds", []) if execution else [],
        "executed_drops": execution.get("executed_drops", []) if execution else [],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    marked = portfolio.copy()
    marked["current_weight"] = marked["current_market_value"] / portfolio_value
    marked.to_csv(output_dir / "paper_portfolio_tracked.csv", index=False)
    (output_dir / "paper_portfolio_tracking_summary.json").write_text(
        json.dumps(snapshot, indent=2) + "\n"
    )

    history_path = output_dir / "paper_portfolio_tracking_history.csv"
    history_row = pd.DataFrame([snapshot])
    if history_path.exists():
        history = pd.read_csv(history_path)
        history = pd.concat([history, history_row], ignore_index=True)
    else:
        history = history_row
    history.to_csv(history_path, index=False)

    from stock_solve.dashboard import generate_dashboard

    generate_dashboard(output_dir)
    return snapshot


def print_snapshot(snapshot: dict) -> None:
    print("Latest price date:", snapshot["latest_price_date"])
    print("Initial signal date:", snapshot["initial_signal_date"])
    print("Portfolio value:", f"${snapshot['portfolio_value']:,.2f}")
    print(
        "Portfolio return since reset:",
        f"{snapshot.get('portfolio_return_since_reset', snapshot['portfolio_return']) * 100:.2f}%",
    )
    print("Portfolio return since trade basis:", f"{snapshot['portfolio_return'] * 100:.2f}%")
    print("Max current holding weight:", f"{snapshot['max_current_weight'] * 100:.2f}%")
    print("Rebalance needed:", snapshot["rebalance_needed"])
    print("Rebalance signal allowed:", snapshot["rebalance_signal_allowed"])
    print("Deferred until weekly gate:", snapshot["rebalance_deferred_until_weekly_gate"])
    print("Rebalance executed:", snapshot["rebalance_executed"])
    if snapshot["rebalance_executed"]:
        print("Executed at:", snapshot["execution_date"], snapshot["execution_price_type"])
        print("Validated before execution:", snapshot["validated_before_execution"])
        print("Open validation passed:", snapshot["open_validation_passed"])
        print("Open validation adjusted target:", snapshot["open_validation_adjusted"])
        print("Executed add:", ", ".join(snapshot["executed_adds"]) or "none")
        print("Executed drop:", ", ".join(snapshot["executed_drops"]) or "none")
    if snapshot["pending_rebalance_replaced"]:
        print("Pending rebalance replaced after validation")
        print("Replacement add:", ", ".join(snapshot["replacement_adds"]) or "none")
        print("Replacement drop:", ", ".join(snapshot["replacement_drops"]) or "none")
    if snapshot["pending_rebalance_cancelled"]:
        print("Pending rebalance cancelled after validation")
    if snapshot["open_validation_cancelled"]:
        print("Pending rebalance cancelled at open validation")
    if snapshot["pending_rebalance"]:
        print("Pending signal date:", snapshot["pending_signal_date"])
        print("Pending add:", ", ".join(snapshot["pending_adds"]) or "none")
        print("Pending drop:", ", ".join(snapshot["pending_drops"]) or "none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track the current paper portfolio.")
    parser.add_argument("--portfolio", type=Path, default=Path("outputs/paper_portfolio.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--rebalance",
        action="store_true",
        help="Overwrite the paper portfolio with current momentum picks when adds/drops are signaled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = track_portfolio(args.portfolio, args.output_dir, args.rebalance)
    print_snapshot(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
