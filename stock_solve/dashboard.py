from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from stock_solve.backtest import BENCHMARK, download_adjusted_closes


def money(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def latest_daily_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    history = history.copy()
    history["timestamp_utc"] = pd.to_datetime(history["timestamp_utc"], errors="coerce")
    history = history.sort_values("timestamp_utc")
    return history.groupby("latest_price_date", as_index=False).tail(1).reset_index(drop=True)


def add_benchmark_history(
    history: pd.DataFrame,
    basis: float,
    reset_date: str | None = None,
    reset_price: float | None = None,
) -> pd.DataFrame:
    if history.empty:
        return history
    plot = latest_daily_history(history)
    start = reset_date or str(plot["latest_price_date"].min())
    start = (pd.Timestamp(start) - pd.Timedelta(days=14)).date().isoformat()
    try:
        benchmark = download_adjusted_closes([BENCHMARK], start, None, None)
    except Exception:
        return plot

    if BENCHMARK not in benchmark.columns or benchmark[BENCHMARK].dropna().empty:
        return plot

    prices = benchmark[BENCHMARK].dropna()
    dates = pd.to_datetime(plot["latest_price_date"])
    aligned = prices.reindex(dates).ffill()
    if aligned.isna().any():
        aligned = prices.reindex(prices.index.union(dates)).sort_index().ffill().reindex(dates)
    if aligned.dropna().empty:
        return plot

    base_price = reset_price if reset_price else float(aligned.dropna().iloc[0])
    plot = plot.copy()
    plot["benchmark_value"] = aligned.values / base_price * basis
    return plot


def line_chart_svg(history: pd.DataFrame, width: int = 920, height: int = 320) -> str:
    if history.empty:
        return '<div class="empty">No tracking history yet.</div>'

    plot = latest_daily_history(history)
    values = plot["portfolio_value"].astype(float).tolist()
    benchmark_values = (
        plot["benchmark_value"].astype(float).tolist()
        if "benchmark_value" in plot.columns
        else []
    )
    labels = plot["latest_price_date"].astype(str).tolist()
    left, right, top, bottom = 72, 24, 24, 42
    inner_w = width - left - right
    inner_h = height - top - bottom
    all_values = values + benchmark_values
    min_v, max_v = min(all_values), max(all_values)
    if min_v == max_v:
        min_v *= 0.995
        max_v *= 1.005
    pad = (max_v - min_v) * 0.08
    min_v -= pad
    max_v += pad

    def x_at(i: int) -> float:
        if len(values) == 1:
            return left + inner_w / 2
        return left + i * inner_w / (len(values) - 1)

    def y_at(value: float) -> float:
        return top + (max_v - value) * inner_h / (max_v - min_v)

    points = " ".join(f"{x_at(i):.2f},{y_at(v):.2f}" for i, v in enumerate(values))
    benchmark_points = " ".join(
        f"{x_at(i):.2f},{y_at(v):.2f}" for i, v in enumerate(benchmark_values)
    )
    y_ticks = [min_v + (max_v - min_v) * i / 4 for i in range(5)]
    grid = []
    for tick in y_ticks:
        y = y_at(tick)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" class="grid" />'
        )
        grid.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" class="axis-label">{html.escape(money(tick))}</text>'
        )

    markers = []
    for i, (label, value) in enumerate(zip(labels, values)):
        x, y = x_at(i), y_at(value)
        markers.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" class="point" />')
        markers.append(
            f'<title>Portfolio {html.escape(label)}: {html.escape(money(value))}</title>'
        )
    benchmark_markers = []
    for i, (label, value) in enumerate(zip(labels, benchmark_values)):
        x, y = x_at(i), y_at(value)
        benchmark_markers.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.75" class="benchmark-point" />'
        )
        benchmark_markers.append(
            f'<title>{html.escape(BENCHMARK)} {html.escape(label)}: {html.escape(money(value))}</title>'
        )

    if len(labels) <= 8:
        label_indices = range(len(labels))
    else:
        label_indices = sorted({0, len(labels) - 1, len(labels) // 2})
    x_labels = []
    for i in label_indices:
        x_labels.append(
            f'<text x="{x_at(i):.2f}" y="{height-14}" text-anchor="middle" class="axis-label">{html.escape(labels[i])}</text>'
        )

    return f"""
    <svg class="balance-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Portfolio value by day">
      <rect x="0" y="0" width="{width}" height="{height}" class="chart-bg" />
      {''.join(grid)}
      <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis" />
      <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis" />
      <polyline points="{points}" fill="none" class="line" />
      {'<polyline points="' + benchmark_points + '" fill="none" class="benchmark-line" />' if benchmark_points else ''}
      {''.join(markers)}
      {''.join(benchmark_markers)}
      <g class="legend">
        <line x1="{left}" y1="{top+8}" x2="{left+34}" y2="{top+8}" class="line" />
        <text x="{left+42}" y="{top+12}" class="axis-label">Paper portfolio</text>
        <line x1="{left+170}" y1="{top+8}" x2="{left+204}" y2="{top+8}" class="benchmark-line" />
        <text x="{left+212}" y="{top+12}" class="axis-label">{html.escape(BENCHMARK)}</text>
      </g>
      {''.join(x_labels)}
    </svg>
    """


def metric(label: str, value: str, note: str = "") -> str:
    return f"""
    <section class="metric">
      <span>{html.escape(label)}</span>
      <strong>{html.escape(value)}</strong>
      <small>{html.escape(note)}</small>
    </section>
    """


def positions_table(positions: pd.DataFrame) -> str:
    if positions.empty:
        return '<div class="empty">No current positions found.</div>'

    rows = []
    value_sort_col = "current_market_value" if "current_market_value" in positions else "market_value"
    sorted_positions = positions.sort_values(value_sort_col, ascending=False)
    for _, row in sorted_positions.iterrows():
        current_price = row.get("current_adjusted_close", row.get("latest_adjusted_close", 0.0))
        market_value = row.get("current_market_value", row.get("market_value", 0.0))
        current_weight = row.get("current_weight", row.get("actual_weight", 0.0))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['ticker']))}</td>"
            f"<td class=\"num\">{int(row['shares'])}</td>"
            f"<td class=\"num\">{html.escape(money(float(current_price)))}</td>"
            f"<td class=\"num\">{html.escape(money(float(market_value)))}</td>"
            f"<td class=\"num\">{html.escape(pct(float(current_weight)))}</td>"
            f"<td class=\"num\">{html.escape(pct(float(row.get('lookback_momentum', 0.0))))}</td>"
            "</tr>"
        )
    return """
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th class="num">Shares</th>
          <th class="num">Price</th>
          <th class="num">Value</th>
          <th class="num">Weight</th>
          <th class="num">126D Momentum</th>
        </tr>
      </thead>
      <tbody>
    """ + "".join(rows) + """
      </tbody>
    </table>
    """


def rebalance_table(history: pd.DataFrame) -> str:
    if history.empty:
        return '<div class="empty">No rebalance checks found.</div>'
    checks = history.copy()
    masks = []
    for column in [
        "rebalance_needed",
        "rebalance_executed",
        "pending_rebalance",
        "rebalance_deferred_until_weekly_gate",
        "open_validation_cancelled",
        "open_validation_adjusted",
    ]:
        if column in checks:
            masks.append(checks[column].astype(str).str.lower() == "true")
    if not masks:
        return '<div class="empty">No rebalance checks found.</div>'
    mask = masks[0]
    for extra in masks[1:]:
        mask = mask | extra
    checks = checks[mask]
    if checks.empty:
        return '<div class="empty">No add/drop changes have been needed since the current tracker started.</div>'

    rows = []
    for _, row in checks.sort_values("timestamp_utc", ascending=False).iterrows():
        executed = str(row.get("rebalance_executed", False)).lower() == "true"
        pending = str(row.get("pending_rebalance", False)).lower() == "true"
        if executed:
            status = f"Executed {row.get('execution_date', '')} {row.get('execution_price_type', '')}"
            if str(row.get("open_validation_adjusted", False)).lower() == "true":
                status += " (open-adjusted)"
            drops = row.get("executed_drops", row.get("recommended_drops", "[]"))
            adds = row.get("executed_adds", row.get("recommended_adds", "[]"))
        elif str(row.get("open_validation_cancelled", False)).lower() == "true":
            status = "Cancelled at open validation"
            drops = row.get("recommended_drops", "[]")
            adds = row.get("recommended_adds", "[]")
        elif pending:
            status = "Pending next open"
            drops = row.get("pending_drops", row.get("recommended_drops", "[]"))
            adds = row.get("pending_adds", row.get("recommended_adds", "[]"))
        elif str(row.get("rebalance_deferred_until_weekly_gate", False)).lower() == "true":
            status = "Deferred until weekly gate"
            drops = row.get("recommended_drops", "[]")
            adds = row.get("recommended_adds", "[]")
        else:
            status = "Signal only"
            drops = row.get("recommended_drops", "[]")
            adds = row.get("recommended_adds", "[]")
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('latest_price_date', '')))}</td>"
            f"<td>{html.escape(str(drops))}</td>"
            f"<td>{html.escape(str(adds))}</td>"
            f"<td>{html.escape(status.strip())}</td>"
            "</tr>"
        )
    return """
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Drops</th>
          <th>Adds</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
    """ + "".join(rows) + """
      </tbody>
    </table>
    """


def generate_dashboard(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracked_path = output_dir / "paper_portfolio_tracked.csv"
    portfolio_path = output_dir / "paper_portfolio.csv"
    history_path = output_dir / "paper_portfolio_tracking_history.csv"
    summary_path = output_dir / "paper_portfolio_tracking_summary.json"

    positions = pd.read_csv(tracked_path if tracked_path.exists() else portfolio_path)
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    summary = load_json(summary_path)

    current_value = float(summary.get("portfolio_value", positions.get("market_value", pd.Series()).sum()))
    current_basis = float(summary.get("initial_capital", current_value))
    original_basis = float(history["initial_capital"].iloc[0]) if not history.empty else current_basis
    benchmark_reset_date = summary.get("benchmark_reset_date")
    benchmark_reset_price = summary.get("benchmark_reset_price")
    benchmark_reset_value = float(summary.get("benchmark_reset_value", original_basis))
    history = add_benchmark_history(
        history,
        benchmark_reset_value,
        benchmark_reset_date,
        benchmark_reset_price,
    )
    cash = float(summary.get("cash", 0.0))
    latest_price_date = str(summary.get("latest_price_date", "unknown"))
    max_weight = float(summary.get("max_current_weight", 0.0))
    return_current_basis = current_value / current_basis - 1 if current_basis else 0.0
    return_original_basis = float(
        summary.get(
            "portfolio_return_since_reset",
            current_value / original_basis - 1 if original_basis else 0.0,
        )
    )
    benchmark_value = (
        float(latest_daily_history(history)["benchmark_value"].iloc[-1])
        if not history.empty and "benchmark_value" in history.columns
        else None
    )
    benchmark_return = (
        benchmark_value / original_basis - 1
        if benchmark_value is not None and original_basis
        else None
    )
    excess_return = (
        return_original_basis - benchmark_return
        if benchmark_return is not None
        else None
    )
    rebalance_needed = bool(summary.get("rebalance_needed", False))
    rebalance_executed = bool(summary.get("rebalance_executed", False))
    pending_rebalance = bool(summary.get("pending_rebalance", False))
    rebalance_deferred = bool(summary.get("rebalance_deferred_until_weekly_gate", False))

    metrics = "".join(
        [
            metric("Current Value", money(current_value), f"Latest prices: {latest_price_date}"),
            metric("Return Since Reset", pct(return_original_basis), f"Original basis: {money(original_basis)}"),
            metric(
                f"{BENCHMARK} Value",
                money(benchmark_value) if benchmark_value is not None else "Unavailable",
                pct(benchmark_return) if benchmark_return is not None else "No benchmark data",
            ),
            metric(
                f"Excess vs {BENCHMARK}",
                pct(excess_return) if excess_return is not None else "Unavailable",
                "Since original reset",
            ),
            metric("Return Since Basis", pct(return_current_basis), f"Current basis: {money(current_basis)}"),
            metric("Cash", money(cash), pct(cash / current_value if current_value else 0.0)),
            metric("Max Holding", pct(max_weight), "15% cap"),
            metric(
                "Rebalance",
                "Pending"
                if pending_rebalance
                else ("Deferred" if rebalance_deferred else ("Needed" if rebalance_needed else "Not Needed")),
                "Executed at validated open"
                if rebalance_executed
                else (
                    "Waiting for next open"
                    if pending_rebalance
                    else ("Weekly gate" if rebalance_deferred else "No action")
                ),
            ),
        ]
    )

    largest = positions.copy()
    value_col = "current_market_value" if "current_market_value" in largest else "market_value"
    if largest.empty:
        position_note = "<li>No open stock positions; cash is waiting for a validated next-open paper execution.</li>"
        concentration_note = "<li>Top five concentration: <strong>0.00%</strong>.</li>"
    else:
        top_position = largest.sort_values(value_col, ascending=False).iloc[0]
        concentration = (
            largest.nlargest(5, value_col)[value_col].astype(float).sum() / current_value
            if current_value
            else 0.0
        )
        position_note = (
            f"<li>Largest position: <strong>{html.escape(str(top_position['ticker']))}</strong> "
            f"at {pct(float(top_position.get('current_weight', top_position.get('actual_weight', 0.0))))}.</li>"
        )
        concentration_note = f"<li>Top five concentration: <strong>{pct(concentration)}</strong>.</li>"

    rules = {}
    summary_config = load_json(output_dir / "paper_portfolio_summary.json").get("config", {})
    for key in [
        "entry_rank",
        "keep_until_rank",
        "confirmation_signals",
        "replacement_momentum_hurdle",
        "rebalance_weekday",
    ]:
        if key in summary_config:
            rules[key] = summary_config[key]

    rules_note = (
        "Entry rank <= {entry_rank}, keep until rank {keep_until_rank}, "
        "{confirmation_signals} confirming signals, {hurdle:.0%} momentum hurdle, "
        "weekly decision day {weekday}."
    ).format(
        entry_rank=rules.get("entry_rank", 20),
        keep_until_rank=rules.get("keep_until_rank", 30),
        confirmation_signals=rules.get("confirmation_signals", 2),
        hurdle=float(rules.get("replacement_momentum_hurdle", 0.03)),
        weekday=rules.get("rebalance_weekday", 4),
    )

    useful_notes = f"""
      <ul class="notes">
        {position_note}
        {concentration_note}
        <li>Benchmark: <strong>{html.escape(BENCHMARK)}</strong>, normalized to the reset date {html.escape(str(benchmark_reset_date or 'first tracked date'))} and reset value {money(benchmark_reset_value)}.</li>
        <li>Current rule: current Nasdaq-100 momentum with hysteresis. {html.escape(rules_note)}</li>
        <li>Paper rebalances are revalidated with execution-day open prices before a fill.</li>
        <li>This is paper tracking only; no real brokerage orders are placed.</li>
      </ul>
    """

    dashboard = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Solve Paper Portfolio</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --text: #18212f;
      --muted: #647084;
      --line: #d9dee7;
      --panel: #ffffff;
      --accent: #176b87;
      --benchmark: #8a4f14;
      --gain: #1f7a4d;
      --warn: #a05a00;
      --soft: #eef3f6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 24px 28px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0; font-size: 24px; letter-spacing: 0; }}
    header p {{ margin: 6px 0 0; color: var(--muted); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 22px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 98px;
    }}
    .metric span, .metric small {{ display: block; color: var(--muted); }}
    .metric strong {{ display: block; margin: 6px 0 4px; font-size: 24px; letter-spacing: 0; }}
    section.panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 14px 0;
      overflow: hidden;
    }}
    .panel h2 {{
      margin: 0;
      padding: 13px 16px;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
      letter-spacing: 0;
      background: var(--soft);
    }}
    .panel-body {{ padding: 16px; overflow-x: auto; }}
    .balance-chart {{ width: 100%; height: auto; display: block; }}
    .chart-bg {{ fill: #fff; }}
    .grid {{ stroke: #edf0f4; stroke-width: 1; }}
    .axis {{ stroke: #b7c0cd; stroke-width: 1; }}
    .axis-label {{ fill: var(--muted); font-size: 11px; }}
    .line {{ stroke: var(--accent); stroke-width: 3; }}
    .benchmark-line {{ stroke: var(--benchmark); stroke-width: 2.5; stroke-dasharray: 7 5; }}
    .point {{ fill: var(--gain); stroke: #fff; stroke-width: 2; }}
    .benchmark-point {{ fill: var(--benchmark); stroke: #fff; stroke-width: 1.5; }}
    .legend text {{ font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    tr:last-child td {{ border-bottom: 0; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .notes {{ margin: 0; padding-left: 20px; color: var(--text); }}
    .notes li {{ margin: 7px 0; }}
    .empty {{ color: var(--muted); padding: 10px 0; }}
    @media (max-width: 720px) {{
      header {{ padding: 18px; }}
      main {{ padding: 14px; }}
      .metric strong {{ font-size: 20px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Stock Solve Paper Portfolio</h1>
    <p>Generated from local tracking files. Latest price date: {html.escape(latest_price_date)}.</p>
  </header>
  <main>
    <div class="metrics">{metrics}</div>
    <section class="panel">
      <h2>Balance By Day</h2>
      <div class="panel-body">{line_chart_svg(history)}</div>
    </section>
    <section class="panel">
      <h2>Current Positions</h2>
      <div class="panel-body">{positions_table(positions)}</div>
    </section>
    <section class="panel">
      <h2>Rebalance Log</h2>
      <div class="panel-body">{rebalance_table(history)}</div>
    </section>
    <section class="panel">
      <h2>Useful Checks</h2>
      <div class="panel-body">{useful_notes}</div>
    </section>
  </main>
</body>
</html>
"""
    output_path = output_dir / "dashboard.html"
    output_path.write_text(dashboard)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Stock Solve paper dashboard.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = generate_dashboard(args.output_dir)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
