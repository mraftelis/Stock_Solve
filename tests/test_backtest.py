import unittest
from datetime import datetime, timezone

import pandas as pd

from stock_solve.backtest import (
    BacktestConfig,
    first_trading_days_after_month_end,
    run_backtest,
    select_momentum_portfolio,
)
from stock_solve.live_portfolio import PaperPortfolioConfig, optimized_target_tickers
from stock_solve.research_algorithms import (
    ResearchStrategy,
    rank_tickers,
    run_research_backtest,
)
from stock_solve.track_paper_portfolio import (
    completed_close_prices,
    latest_available_prices_from_frame,
    same_target_tickers,
)


class BacktestTests(unittest.TestCase):
    def test_first_trading_days_after_month_end_uses_index_dates(self):
        index = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-29"])

        dates = first_trading_days_after_month_end(pd.DatetimeIndex(index), "2024-01-31")

        self.assertEqual(
            dates,
            [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        )

    def test_select_momentum_portfolio_ranks_by_adjusted_price_momentum(self):
        index = pd.bdate_range("2024-01-01", periods=6)
        prices = pd.DataFrame(
            {
                "AAA": [10, 11, 12, 13, 14, 15],
                "BBB": [10, 10, 10, 10, 10, 11],
                "CCC": [10, 9, 9, 9, 9, 9],
            },
            index=index,
        )

        picks = select_momentum_portfolio(prices, lookback_days=5, picks=2)

        self.assertEqual(picks, ["AAA", "BBB"])

    def test_run_backtest_enforces_holdings_and_position_caps_on_synthetic_data(self):
        index = pd.bdate_range("2022-01-03", periods=360)
        data = {}
        for i in range(25):
            data[f"S{i:02d}"] = [100 + i + day * (1 + i / 50) for day in range(len(index))]
        data["VFIAX"] = [100 + day for day in range(len(index))]
        prices = pd.DataFrame(data, index=index)
        config = BacktestConfig(
            data_start="2022-01-03",
            start="2022-06-01",
            end="2023-06-01",
            picks=20,
            momentum_lookback_days=60,
            min_holdings=20,
            max_position_weight=0.15,
        )

        _, _, summary = run_backtest(prices, [f"S{i:02d}" for i in range(25)], config)

        self.assertGreaterEqual(summary["min_holdings"], 20)
        self.assertLessEqual(summary["max_observed_weight"], 0.15)

    def test_optimized_target_tickers_fills_minimum_holdings_after_confirmation_filter(self):
        index = pd.bdate_range("2026-01-01", periods=10)
        universe = [f"S{i:02d}" for i in range(25)]
        prices = pd.DataFrame(
            {
                ticker: [100 + day * (1 + i / 10) for day in range(len(index))]
                for i, ticker in enumerate(universe)
            },
            index=index,
        )
        config = PaperPortfolioConfig(
            lookback_days=2,
            picks=20,
            entry_rank=1,
            keep_until_rank=30,
            confirmation_signals=2,
        )
        current_tickers = [f"S{i:02d}" for i in range(6, 25)]

        picks, _, selection = optimized_target_tickers(prices, universe, config, current_tickers)

        self.assertEqual(len(picks), 20)
        self.assertEqual(selection["fallback_fill"], ["S05"])

    def test_optimized_target_tickers_caps_semiconductor_hardware_group(self):
        index = pd.bdate_range("2026-01-01", periods=10)
        semis = [
            "MU",
            "INTC",
            "WDC",
            "STX",
            "ARM",
            "MRVL",
            "AMD",
            "AMAT",
            "LRCX",
            "KLAC",
        ]
        diversifiers = [f"S{i:02d}" for i in range(15)]
        universe = semis + diversifiers
        prices = pd.DataFrame(
            {
                ticker: [100 + day * (3 if ticker in semis else 1 + i / 20) for day in range(len(index))]
                for i, ticker in enumerate(universe)
            },
            index=index,
        )
        config = PaperPortfolioConfig(
            lookback_days=2,
            picks=20,
            max_risk_group_count=8,
        )

        picks, _, selection = optimized_target_tickers(prices, universe, config)

        self.assertEqual(len(picks), 20)
        self.assertEqual(selection["risk_group_counts"]["semiconductor_hardware"], 8)
        self.assertEqual(len([ticker for ticker in semis if ticker in picks]), 8)

    def test_latest_available_prices_forward_fills_single_ticker_gaps(self):
        prices = pd.DataFrame(
            {
                "AAA": [10.0, 11.0, None],
                "BBB": [20.0, None, 22.0],
            },
            index=pd.to_datetime(["2026-06-12", "2026-06-15", "2026-06-16"]),
        )

        latest = latest_available_prices_from_frame(prices)

        self.assertEqual(latest.to_dict(), {"AAA": 11.0, "BBB": 22.0})

    def test_completed_close_prices_excludes_current_intraday_bar(self):
        prices = pd.DataFrame(
            {"AAA": [10.0, 11.0]},
            index=pd.to_datetime(["2026-09-17", "2026-09-18"]),
        )

        completed = completed_close_prices(
            prices,
            datetime(2026, 9, 18, 13, 42, tzinfo=timezone.utc),
        )

        self.assertEqual(completed.index.tolist(), [pd.Timestamp("2026-09-17")])

    def test_completed_close_prices_keeps_current_bar_after_close_settles(self):
        prices = pd.DataFrame(
            {"AAA": [10.0, 11.0]},
            index=pd.to_datetime(["2026-09-17", "2026-09-18"]),
        )

        completed = completed_close_prices(
            prices,
            datetime(2026, 9, 18, 22, 41, tzinfo=timezone.utc),
        )

        self.assertEqual(completed.index.tolist(), prices.index.tolist())

    def test_same_target_tickers_ignores_ranking_order(self):
        self.assertTrue(same_target_tickers(["AAA", "BBB"], ["BBB", "AAA"]))
        self.assertFalse(same_target_tickers(["AAA", "BBB"], ["AAA", "CCC"]))

    def test_research_ranker_can_skip_recent_data_without_using_it_for_momentum(self):
        index = pd.bdate_range("2026-01-01", periods=8)
        prices = pd.DataFrame(
            {
                "AAA": [10, 10, 10, 10, 10, 10, 10, 99],
                "BBB": [10, 11, 12, 13, 14, 15, 16, 16],
            },
            index=index,
        )
        strategy = ResearchStrategy(
            name="skip_recent",
            description="test",
            lookback_days=3,
            skip_recent_days=1,
            picks=1,
        )

        ranked = rank_tickers(prices, strategy)

        self.assertEqual(ranked.iloc[0]["ticker"], "BBB")

    def test_research_backtest_cash_filter_can_relax_stock_holdings(self):
        index = pd.bdate_range("2026-01-01", periods=260)
        data = {f"S{i:02d}": [100 + day + i for day in range(len(index))] for i in range(25)}
        data["VFIAX"] = [100 - day / 10 for day in range(len(index))]
        prices = pd.DataFrame(data, index=index)
        strategy = ResearchStrategy(
            name="cash_filter",
            description="test",
            lookback_days=20,
            picks=20,
            market_sma_days=50,
            market_return_days=20,
        )

        _, _, summary = run_research_backtest(
            prices,
            [f"S{i:02d}" for i in range(25)],
            strategy,
            start="2026-03-02",
            rebalance_frequency="weekly",
        )

        self.assertEqual(summary["min_active_holdings"], 0)
        self.assertLessEqual(summary["max_observed_weight"], 0.15)


if __name__ == "__main__":
    unittest.main()
