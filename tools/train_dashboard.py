#!/usr/bin/env python3
"""
Tiny browser dashboard for Parameter Golf training logs.

Usage:
    python3 tools/train_dashboard.py
    python3 tools/train_dashboard.py --host 127.0.0.1 --port 8765 --logs logs

Then open:
    http://127.0.0.1:8765

The dashboard polls /api/logs and parses logs/*.txt. It does not require MLX,
matplotlib, node, or any frontend build step.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


STEP_RE = re.compile(r"step:(?P<step>\d+)/(?P<total>\d+)(?P<body>.*)")
VAL_PROGRESS_RE = re.compile(r"val_progress:(?P<done>\d+)/(?P<total>\d+)")
METRIC_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*):(?P<value>-?\d+(?:\.\d+)?)")
BYTES_RE = re.compile(r"(?P<bytes>\d+) bytes")
LOG_LIMIT_BYTES = 2_000_000


def parse_metrics(text: str) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for match in METRIC_RE.finditer(text):
        raw = match.group("value")
        value: float | int
        if "." in raw:
            value = float(raw)
        else:
            value = int(raw)
        metrics[match.group("name")] = value
    return metrics


def read_tail(path: Path, limit: int = LOG_LIMIT_BYTES) -> str:
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > limit:
            handle.seek(size - limit)
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def parse_log(path: Path) -> dict[str, object]:
    stat = path.stat()
    text = read_tail(path)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    train_points: list[dict[str, object]] = []
    val_events: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    final_metrics: dict[str, float | int | str] = {}
    latest_step: int | None = None
    total_steps: int | None = None
    latest_val_progress: dict[str, int | float] | None = None
    status = "waiting"

    for line in lines:
        step_match = STEP_RE.search(line)
        if step_match:
            latest_step = int(step_match.group("step"))
            total_steps = int(step_match.group("total"))
            metrics = parse_metrics(step_match.group("body"))
            point: dict[str, object] = {
                "step": latest_step,
                "total": total_steps,
                "line": line,
            }
            point.update(metrics)
            train_points.append(point)
            if "val_loss" in metrics or "val_bpb" in metrics:
                val_events.append(point)
            status = "training"

        progress_match = VAL_PROGRESS_RE.search(line)
        if progress_match:
            done = int(progress_match.group("done"))
            total = int(progress_match.group("total"))
            latest_val_progress = {
                "done": done,
                "total": total,
                "fraction": done / total if total else 0,
            }
            status = "validating" if done < total else status

        if line.startswith("final_"):
            metrics = parse_metrics(line)
            final_metrics.update(metrics)
            final_metrics["kind"] = line.split()[0]
            status = "complete"

        if line.startswith("stopping_early:"):
            final_metrics["stop_reason"] = line
            status = "stopped"

        if line.startswith("saved_model:"):
            bytes_match = BYTES_RE.search(line)
            artifacts.append(
                {
                    "kind": "model",
                    "line": line,
                    "bytes": int(bytes_match.group("bytes")) if bytes_match else None,
                }
            )

        if line.startswith("serialized_model"):
            bytes_match = BYTES_RE.search(line)
            artifacts.append(
                {
                    "kind": "compressed",
                    "line": line,
                    "bytes": int(bytes_match.group("bytes")) if bytes_match else None,
                }
            )

    if latest_step is not None and total_steps and latest_step >= total_steps:
        status = "complete" if final_metrics else "trained"

    best_bpb = final_metrics.get("val_bpb")
    if best_bpb is None:
        for event in reversed(val_events):
            if "val_bpb" in event:
                best_bpb = event["val_bpb"]
                break

    latest_train_loss = None
    latest_tok_s = None
    latest_step_avg = None
    for point in reversed(train_points):
        latest_train_loss = point.get("train_loss", latest_train_loss)
        latest_tok_s = point.get("tok_s", latest_tok_s)
        latest_step_avg = point.get("step_avg", latest_step_avg)
        if latest_train_loss is not None:
            break

    return {
        "name": path.name,
        "path": str(path),
        "mtime": stat.st_mtime,
        "mtime_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "size": stat.st_size,
        "status": status,
        "latest_step": latest_step,
        "total_steps": total_steps,
        "progress": latest_step / total_steps if latest_step is not None and total_steps else None,
        "latest_train_loss": latest_train_loss,
        "latest_tok_s": latest_tok_s,
        "latest_step_avg": latest_step_avg,
        "latest_val_progress": latest_val_progress,
        "best_bpb": best_bpb,
        "final_metrics": final_metrics,
        "train_points": train_points[-300:],
        "val_events": val_events[-80:],
        "artifacts": artifacts,
        "tail": lines[-80:],
    }


def load_logs(log_dir: Path) -> dict[str, object]:
    files = sorted(log_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = []
    errors = []
    for path in files:
        try:
            runs.append(parse_log(path))
        except OSError as exc:
            errors.append({"name": path.name, "error": str(exc)})
    return {
        "generated_at": time.time(),
        "log_dir": str(log_dir),
        "runs": runs,
        "errors": errors,
    }


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Parameter Golf Training Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #071018;
      --panel: #0f1a24;
      --panel-2: #121f2b;
      --panel-3: #0b151e;
      --line: #263748;
      --line-soft: rgba(129, 157, 183, 0.16);
      --text: #dce7f3;
      --muted: #94a6ba;
      --muted-2: #607386;
      --blue: #4f8ef7;
      --purple: #b56dff;
      --green: #56c46a;
      --yellow: #e6bf45;
      --red: #f06158;
      --shadow: rgba(0, 0, 0, 0.34);
      --mono: "SFMono-Regular", ui-monospace, Menlo, Consolas, monospace;
      --sans: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: var(--sans);
      background:
        radial-gradient(circle at 16% 0%, rgba(61, 102, 142, .22), transparent 34rem),
        radial-gradient(circle at 84% 12%, rgba(86, 196, 106, .10), transparent 28rem),
        linear-gradient(180deg, #071018 0%, #0a121a 100%);
    }

    button, input, select {
      font: inherit;
      color: var(--text);
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(18, 31, 43, .92), rgba(10, 20, 29, .92));
      border-radius: 7px;
      outline: none;
    }

    button { cursor: pointer; transition: border-color .16s ease, background .16s ease, transform .16s ease; }
    button:hover { border-color: #4a6177; transform: translateY(-1px); }
    input, select, button { height: 36px; padding: 0 12px; }
    input { min-width: 260px; }
    select { min-width: 190px; }

    main { width: min(1920px, calc(100vw - 28px)); margin: 0 auto; padding: 16px 0 24px; }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }

    .controls { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; }
    .control-wrap { position: relative; display: inline-flex; align-items: center; }
    .control-wrap svg { position: absolute; left: 11px; color: var(--muted); }
    .control-wrap input { padding-left: 36px; }
    .control-wrap select { padding-left: 58px; appearance: none; }
    .control-label { position: absolute; left: 12px; color: var(--muted); font-size: 12px; pointer-events: none; }
    .refresh { display: inline-flex; align-items: center; gap: 8px; }
    .divider { width: 1px; height: 34px; background: var(--line); opacity: .7; }

    .switch-row { display: inline-flex; align-items: center; gap: 10px; color: var(--text); font-size: 14px; }
    .spinner { width: 22px; height: 22px; border-radius: 50%; border: 3px solid #34485e; border-top-color: #87aef8; }
    .switch { width: 41px; height: 22px; padding: 2px; border-radius: 999px; border: 1px solid #466178; background: #1f4a82; }
    .knob { display: block; width: 18px; height: 18px; border-radius: 999px; background: #e8f0ff; margin-left: 17px; box-shadow: 0 0 16px rgba(80, 142, 247, .55); }
    .refreshed { margin-left: auto; color: var(--muted); font-size: 13px; white-space: nowrap; }

    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(180deg, rgba(18, 31, 43, .86), rgba(12, 23, 33, .92));
      box-shadow: 0 14px 42px var(--shadow);
      overflow: hidden;
    }

    .runs-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .runs-table th { padding: 12px 16px; text-align: left; color: #eef4fb; font-size: 12px; font-weight: 700; border-bottom: 1px solid var(--line); white-space: nowrap; }
    .runs-table td { padding: 10px 16px; border-bottom: 1px solid var(--line-soft); color: #cad8e7; white-space: nowrap; }
    .runs-table tbody tr { cursor: pointer; transition: background .14s ease; }
    .runs-table tbody tr:hover, .runs-table tbody tr.active { background: rgba(111, 136, 160, .13); }
    .runs-table tbody tr:last-child td { border-bottom: 0; }
    .mono { font-family: var(--mono); }
    .num { text-align: right; font-variant-numeric: tabular-nums; }

    .status { display: inline-flex; align-items: center; gap: 8px; font-weight: 700; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--muted-2); }
    .status.training { color: var(--blue); }
    .status.training .dot { background: var(--blue); }
    .status.validating { color: var(--yellow); }
    .status.validating .dot { background: var(--yellow); }
    .status.trained, .status.complete { color: var(--green); }
    .status.trained .dot, .status.complete .dot { background: var(--green); }
    .status.stopped { color: var(--red); }
    .status.stopped .dot { background: var(--red); }

    .progress-cell { display: flex; align-items: center; gap: 10px; min-width: 118px; }
    .bar { flex: 0 0 78px; height: 10px; border: 1px solid #33475a; border-radius: 999px; overflow: hidden; background: #0a131c; }
    .fill { height: 100%; border-radius: inherit; background: var(--blue); }
    .fill.done { background: var(--green); }

    .summary-grid { display: grid; grid-template-columns: 1.05fr 1.05fr 2.35fr 1.45fr; gap: 12px; margin: 14px 0; }
    .tile { min-height: 94px; padding: 17px 20px; text-align: center; }
    .tile-title { color: var(--text); font-size: 14px; }
    .tile-value { margin-top: 8px; font-size: 28px; line-height: 1; font-weight: 700; font-variant-numeric: tabular-nums; }
    .tile-sub { margin-top: 8px; color: var(--muted); font-size: 13px; }
    .best { color: var(--green); font-size: 30px; }
    .best-row { display: grid; grid-template-columns: 1fr 1px 1fr; align-items: center; gap: 18px; }
    .vertical-line { height: 56px; background: var(--line); }
    .status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 24px; padding: 14px 28px; font-size: 13px; }
    .count-row { display: flex; align-items: center; justify-content: space-between; gap: 14px; }

    .dashboard-grid { display: grid; grid-template-columns: 410px minmax(0, 1fr) 620px; gap: 12px; align-items: start; }
    .details { padding: 18px; }
    .panel-title { margin: 0 0 16px; font-size: 16px; font-weight: 700; color: #f2f6fb; }
    .detail-row { display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 12px; margin: 0 0 11px; color: #cbd9e8; font-size: 13px; }
    .detail-row span:first-child { color: var(--muted); }
    .detail-row span:last-child { overflow: hidden; text-overflow: ellipsis; }
    .rule { height: 1px; background: var(--line); margin: 15px 0; }
    .open-log { display: inline-flex; align-items: center; gap: 8px; margin-top: 18px; text-decoration: none; color: var(--text); border: 1px solid var(--line); border-radius: 7px; padding: 9px 12px; font-size: 13px; }

    .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; grid-column: 2 / 4; }
    .chart-card { padding: 16px 18px 10px; min-height: 300px; }
    .chart-title { font-size: 16px; font-weight: 700; margin-bottom: 10px; }
    svg.chart { width: 100%; height: 238px; display: block; }
    .grid-line { stroke: rgba(143, 166, 188, .12); stroke-width: 1; }
    .axis { stroke: rgba(151, 173, 194, .24); stroke-width: 1; }
    .loss-line { fill: none; stroke: var(--blue); stroke-width: 2; }
    .bpb-line { fill: none; stroke: var(--purple); stroke-width: 2; }
    .chart-dot-loss { fill: var(--blue); }
    .chart-dot-bpb { fill: var(--purple); }
    .legend { display: flex; justify-content: center; align-items: center; gap: 8px; color: #c9d5e2; font-size: 12px; font-family: var(--mono); margin-top: 2px; }
    .legend-line { width: 18px; height: 2px; background: var(--blue); }
    .legend-line.purple { background: var(--purple); }
    .empty { color: var(--muted); padding: 84px 0; text-align: center; }

    .bottom-grid { display: grid; grid-template-columns: minmax(0, 1fr) 610px; gap: 12px; grid-column: 2 / 4; }
    .log-tail, .artifacts { padding: 16px; }
    .section-sub { color: var(--muted); font-weight: 400; }
    pre { margin: 0; max-height: 330px; overflow: auto; padding: 12px; border: 1px solid var(--line-soft); border-radius: 7px; background: rgba(4, 9, 13, .48); color: #d8e3ef; font-family: var(--mono); font-size: 12px; line-height: 1.55; white-space: pre-wrap; }
    .artifact-block { padding: 2px 0 16px; margin-bottom: 16px; border-bottom: 1px solid var(--line); }
    .artifact-block:last-child { border-bottom: 0; margin-bottom: 0; padding-bottom: 0; }
    .artifact-lines { display: grid; gap: 6px; color: #56df7e; font-family: var(--mono); font-size: 12px; line-height: 1.35; }
    .artifact-lines.compressed { color: #59d8e6; }
    .artifact-lines.final { color: #c77cff; }

    @media (max-width: 1280px) {
      .summary-grid, .dashboard-grid, .chart-grid, .bottom-grid { grid-template-columns: 1fr; }
      .chart-grid, .bottom-grid { grid-column: auto; }
      .runs-wrap { overflow-x: auto; }
      .refreshed { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <section class="topbar">
      <div class="controls">
        <label class="control-wrap">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m21 21-4.3-4.3m2.3-5.2a7.5 7.5 0 1 1-15 0 7.5 7.5 0 0 1 15 0Z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
          <input id="filter" placeholder="Filter runs by name..." autocomplete="off">
        </label>
        <label class="control-wrap">
          <span class="control-label">Sort by</span>
          <select id="sort">
            <option value="mtime">Newest</option>
            <option value="bpb">Best val_bpb</option>
            <option value="name">Name</option>
          </select>
        </label>
        <button id="refresh" class="refresh">↻ <span>Refresh Now</span></button>
        <span class="divider"></span>
        <span class="switch-row"><span class="spinner"></span> Auto-refresh <button id="pause" class="switch" aria-label="Pause auto refresh"><span class="knob"></span></button><span id="pauseLabel">Paused</span></span>
      </div>
      <div class="refreshed">Dashboard refreshed: <span id="updated">-</span></div>
    </section>

    <section class="panel runs-wrap">
      <table class="runs-table">
        <thead><tr>
          <th>Status</th><th>Log File</th><th>Last Modified</th><th>Path</th><th class="num">Size</th><th class="num">Latest Step</th><th class="num">Total Steps</th><th>Progress</th><th class="num">Train Loss (latest)</th><th class="num">Tokens / sec</th><th class="num">Step Avg (ms)</th><th class="num">Val Progress</th><th class="num">Best / Latest val_bpb</th>
        </tr></thead>
        <tbody id="runsBody"></tbody>
      </table>
    </section>

    <section class="summary-grid">
      <article class="panel tile"><div class="tile-title">Visible Runs</div><div class="tile-value" id="runCount">0</div><div class="tile-sub" id="totalRuns">of 0 total</div></article>
      <article class="panel tile"><div class="tile-title">Active-ish Runs</div><div class="tile-value" id="activeCount">0</div><div class="tile-sub">training or validating</div></article>
      <article class="panel tile best-row"><div><div class="tile-title">Best val_bpb</div><div class="tile-value best" id="bestBpb">-</div><div class="tile-sub mono" id="bestFile">-</div></div><span class="vertical-line"></span><div id="winnerNote" class="tile-sub">Lower is better.</div></article>
      <article class="panel status-grid" id="statusCounts"></article>
    </section>

    <section class="dashboard-grid">
      <aside class="panel details" id="details"></aside>
      <section class="chart-grid">
        <article class="panel chart-card"><div class="chart-title">Train Loss over Steps</div><div id="lossChart"></div><div class="legend"><span class="legend-line"></span> train_loss</div></article>
        <article class="panel chart-card"><div class="chart-title">Validation val_bpb over Validations</div><div id="bpbChart"></div><div class="legend"><span class="legend-line purple"></span> val_bpb</div></article>
      </section>
      <section class="bottom-grid">
        <article class="panel log-tail"><h3 class="panel-title">Last 80 Non-Empty Log Lines</h3><pre id="tail"></pre></article>
        <article class="panel artifacts" id="artifacts"></article>
      </section>
    </section>
  </main>

  <script>
    const state = { data: null, selected: null, paused: false };
    const els = {
      filter: document.getElementById("filter"), sort: document.getElementById("sort"), refresh: document.getElementById("refresh"), pause: document.getElementById("pause"), pauseLabel: document.getElementById("pauseLabel"), updated: document.getElementById("updated"), runsBody: document.getElementById("runsBody"), runCount: document.getElementById("runCount"), totalRuns: document.getElementById("totalRuns"), activeCount: document.getElementById("activeCount"), bestBpb: document.getElementById("bestBpb"), bestFile: document.getElementById("bestFile"), winnerNote: document.getElementById("winnerNote"), statusCounts: document.getElementById("statusCounts"), details: document.getElementById("details"), lossChart: document.getElementById("lossChart"), bpbChart: document.getElementById("bpbChart"), tail: document.getElementById("tail"), artifacts: document.getElementById("artifacts")
    };

    const statuses = ["training", "validating", "trained", "complete", "waiting", "stopped"];
    const nf0 = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
    const nf1 = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });

    function esc(value) { return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;"); }
    function metric(value, digits = 4) { if (value === null || value === undefined || value === "") return "-"; if (typeof value === "number") return value.toFixed(digits).replace(/\.?0+$/, ""); return String(value); }
    function comma(value) { return typeof value === "number" ? nf0.format(value) : "-"; }
    function pct(value) { return typeof value === "number" ? `${Math.max(0, Math.min(100, value * 100)).toFixed(1)}%` : "0.0%"; }
    function size(bytes) { if (!bytes) return "-"; const mb = bytes / 1024 / 1024; return `${nf1.format(mb)} MB`; }
    function timeOnly(text) { return text ? text.split(" ").at(-1) : "-"; }
    function relTime(ts) { const s = Math.max(0, Math.round(Date.now() / 1000 - ts)); if (s < 60) return `${s}s ago`; const m = Math.round(s / 60); if (m < 60) return `${m}m ago`; return `${Math.round(m / 60)}h ago`; }

    function filteredRuns() {
      if (!state.data) return [];
      const q = els.filter.value.trim().toLowerCase();
      let runs = state.data.runs.filter(run => run.name.toLowerCase().includes(q));
      if (els.sort.value === "bpb") runs = runs.slice().sort((a, b) => (a.best_bpb ?? Infinity) - (b.best_bpb ?? Infinity));
      else if (els.sort.value === "name") runs = runs.slice().sort((a, b) => a.name.localeCompare(b.name));
      else runs = runs.slice().sort((a, b) => b.mtime - a.mtime);
      return runs;
    }

    function statusHtml(status) { return `<span class="status ${esc(status)}"><span class="dot"></span>${esc(status)}</span>`; }
    function progressHtml(run) { const done = ["complete", "trained"].includes(run.status); return `<span class="progress-cell"><span class="bar"><span class="fill ${done ? "done" : ""}" style="width:${esc(pct(run.progress))}"></span></span><span>${esc(pct(run.progress))}</span></span>`; }

    function renderTable(runs) {
      if (!state.selected && runs[0]) state.selected = runs[0].name;
      if (runs.length && !runs.some(run => run.name === state.selected)) state.selected = runs[0].name;
      els.runsBody.innerHTML = runs.map(run => `<tr class="${run.name === state.selected ? "active" : ""}" data-run="${esc(run.name)}">
        <td>${statusHtml(run.status)}</td><td class="mono">${esc(run.name)}</td><td>${esc(timeOnly(run.mtime_text))}</td><td class="mono">${esc(run.path)}</td><td class="num">${esc(size(run.size))}</td><td class="num">${esc(comma(run.latest_step))}</td><td class="num">${esc(comma(run.total_steps))}</td><td>${progressHtml(run)}</td><td class="num">${esc(metric(run.latest_train_loss))}</td><td class="num">${esc(comma(run.latest_tok_s))}</td><td class="num">${esc(metric(run.latest_step_avg, 2))}</td><td class="num">${run.latest_val_progress ? `${esc(run.latest_val_progress.done)}/${esc(run.latest_val_progress.total)}` : "-"}</td><td class="num">${esc(metric(run.best_bpb))}</td>
      </tr>`).join("");
      els.runsBody.querySelectorAll("tr").forEach(row => row.addEventListener("click", () => { state.selected = row.dataset.run; render(); }));
    }

    function renderSummary(runs) {
      const best = runs.filter(run => typeof run.best_bpb === "number").sort((a, b) => a.best_bpb - b.best_bpb)[0];
      const active = runs.filter(run => ["training", "validating"].includes(run.status));
      els.runCount.textContent = runs.length;
      els.totalRuns.textContent = `of ${state.data?.runs.length ?? 0} total`;
      els.activeCount.textContent = active.length;
      els.bestBpb.textContent = best ? metric(best.best_bpb) : "-";
      els.bestFile.textContent = best ? best.name : "-";
      els.winnerNote.textContent = best ? `Selected ${best.status} run.` : "Lower is better.";
      els.statusCounts.innerHTML = statuses.map(status => {
        const count = runs.filter(run => run.status === status).length;
        return `<div class="count-row">${statusHtml(status)}<strong>${count}</strong></div>`;
      }).join("");
    }

    function chart(points, field, cls, dotCls, xLabelMax = null) {
      const usable = points.filter(point => typeof point[field] === "number" && typeof point.step === "number");
      if (usable.length < 2) return `<div class="empty">Not enough ${esc(field)} points yet.</div>`;
      const width = 760, height = 248, padL = 48, padR = 14, padT = 12, padB = 34;
      const xs = usable.map(p => p.step), ys = usable.map(p => p[field]);
      const minX = Math.min(...xs), maxX = xLabelMax || Math.max(...xs);
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      const yPad = (maxY - minY || 1) * .08; minY -= yPad; maxY += yPad;
      const x = v => padL + ((v - minX) / Math.max(1, maxX - minX)) * (width - padL - padR);
      const y = v => height - padB - ((v - minY) / Math.max(.000001, maxY - minY)) * (height - padT - padB);
      const d = usable.map((p, i) => `${i ? "L" : "M"} ${x(p.step).toFixed(1)} ${y(p[field]).toFixed(1)}`).join(" ");
      const last = usable.at(-1);
      const grid = [0,1,2,3,4].map(i => { const yy = padT + i * ((height - padT - padB) / 4); return `<line class="grid-line" x1="${padL}" y1="${yy}" x2="${width-padR}" y2="${yy}"></line>`; }).join("");
      return `<svg class="chart" viewBox="0 0 ${width} ${height}">${grid}<line class="axis" x1="${padL}" y1="${height-padB}" x2="${width-padR}" y2="${height-padB}"></line><line class="axis" x1="${padL}" y1="${padT}" x2="${padL}" y2="${height-padB}"></line><path class="${cls}" d="${d}"></path><circle class="${dotCls}" cx="${x(last.step)}" cy="${y(last[field])}" r="3"></circle><text x="${padL-8}" y="${padT+7}" text-anchor="end" fill="#a9b8c8" font-size="12">${esc(metric(maxY, 2))}</text><text x="${padL-8}" y="${height-padB+4}" text-anchor="end" fill="#a9b8c8" font-size="12">${esc(metric(minY, 2))}</text><text x="${padL}" y="${height-7}" fill="#a9b8c8" font-size="12">0</text><text x="${width-padR}" y="${height-7}" text-anchor="end" fill="#a9b8c8" font-size="12">${esc(xLabelMax ? xLabelMax : maxX)}</text></svg>`;
    }

    function artifactSection(title, lines, cls = "") {
      return `<div class="artifact-block"><h3 class="panel-title">${esc(title)} <span class="section-sub">(last 10)</span></h3><div class="artifact-lines ${esc(cls)}">${lines.length ? lines.slice(-10).map(line => `<div>${esc(line)}</div>`).join("") : `<div>-</div>`}</div></div>`;
    }

    function renderDetail(run) {
      if (!run) return;
      const artifacts = run.artifacts || [];
      const modelLines = artifacts.filter(a => a.kind === "model").map(a => a.line);
      const compressedLines = artifacts.filter(a => a.kind === "compressed").map(a => a.line);
      const finalLines = Object.entries(run.final_metrics || {}).map(([k, v]) => `${k}: ${v}`);
      const val = run.latest_val_progress;
      els.details.innerHTML = `<h3 class="panel-title">Run Details</h3>
        ${[["Log File", run.name], ["Status", statusHtml(run.status)], ["Last Modified", `${timeOnly(run.mtime_text)} (${relTime(run.mtime)} )`], ["Path", run.path], ["Size", size(run.size)]].map(([k,v]) => `<div class="detail-row"><span>${esc(k)}</span><span>${String(v).startsWith('<') ? v : esc(v)}</span></div>`).join("")}
        <div class="rule"></div>
        ${[["Latest Step", comma(run.latest_step)], ["Total Steps", comma(run.total_steps)], ["Progress", progressHtml(run)], ["Train Loss (latest)", metric(run.latest_train_loss)], ["Tokens / sec", comma(run.latest_tok_s)], ["Step Avg (ms)", metric(run.latest_step_avg, 2)], ["Val Progress", val ? `${val.done}/${val.total}` : "-"], ["Best / Latest val_bpb", metric(run.best_bpb)]].map(([k,v]) => `<div class="detail-row"><span>${esc(k)}</span><span>${String(v).startsWith('<') ? v : esc(v)}</span></div>`).join("")}
        <div class="rule"></div>
        <div class="detail-row"><span>Saved Model Artifacts</span><span>${modelLines.length}</span></div><div class="detail-row"><span>Compressed Artifacts</span><span>${compressedLines.length}</span></div>
        <div class="rule"></div><a class="open-log" href="/api/raw?name=${encodeURIComponent(run.name)}" target="_blank">↗ Open Log File</a>`;
      els.lossChart.innerHTML = chart(run.train_points || [], "train_loss", "loss-line", "chart-dot-loss", run.total_steps);
      els.bpbChart.innerHTML = chart(run.val_events || [], "val_bpb", "bpb-line", "chart-dot-bpb");
      els.tail.textContent = (run.tail || []).join("\n");
      els.artifacts.innerHTML = `${artifactSection("Saved Model Artifact Lines", modelLines)}${artifactSection("Compressed Serialized Artifact Lines", compressedLines, "compressed")}<div class="artifact-block"><h3 class="panel-title">Final Validation Metrics <span class="section-sub">(from final_* lines)</span></h3><div class="artifact-lines final">${finalLines.length ? finalLines.map(line => `<div>${esc(line)}</div>`).join("") : `<div>-</div>`}</div></div>`;
    }

    function render() {
      const runs = filteredRuns();
      renderTable(runs); renderSummary(runs); renderDetail(runs.find(run => run.name === state.selected) || runs[0]);
    }

    async function load() {
      if (state.paused) return;
      const response = await fetch(`/api/logs?ts=${Date.now()}`);
      state.data = await response.json();
      els.updated.textContent = new Date().toLocaleTimeString();
      render();
    }

    els.filter.addEventListener("input", render);
    els.sort.addEventListener("change", render);
    els.refresh.addEventListener("click", load);
    els.pause.addEventListener("click", () => { state.paused = !state.paused; els.pauseLabel.textContent = state.paused ? "Paused" : "Running"; if (!state.paused) load(); });
    load();
    setInterval(load, 2500);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    log_dir: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(HTML_PAGE)
            return

        if parsed.path == "/api/logs":
            query = parse_qs(parsed.query)
            log_dir = Path(query.get("logs", [str(self.log_dir)])[0]).expanduser()
            payload = load_logs(log_dir)
            self.send_json(payload)
            return

        if parsed.path == "/api/raw":
            query = parse_qs(parsed.query)
            name = Path(query.get("name", [""])[0]).name
            path = self.log_dir / name
            if not name or path.suffix != ".txt" or not path.exists():
                self.send_error(404, "Log file not found")
                return
            data = read_tail(path, 10_000_000).encode("utf-8", errors="replace")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404, "Not found")

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: object) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a live dashboard for Parameter Golf logs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("TRAIN_DASHBOARD_PORT", "8765")))
    parser.add_argument("--logs", default="logs", help="Directory containing *.txt training logs.")
    args = parser.parse_args()

    log_dir = Path(args.logs).expanduser().resolve()
    DashboardHandler.log_dir = log_dir
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving {html.escape(str(log_dir))} at {url}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
