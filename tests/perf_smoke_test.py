#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance smoke tests for llama-completion (llama.cpp).

Runs >=10 prompts of different lengths, parses common_perf_print lines,
and writes JSON + Markdown summary. Aligns with need.md (TTFT/TPOT/E2E).

TTFT note: llama-completion prints batch timings, not streaming TTFT.
We estimate: ttft_est_ms = prompt_eval_ms + (eval_ms / gen_runs) when gen_runs > 0.
E2E: wall-clock of the whole subprocess (load + prefill + full decode to n tokens).
TPOT: eval_ms / gen_runs (decode phase only).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class PerfRow:
    prompt_len_chars: int
    prompt_preview: str
    returncode: int
    wall_e2e_ms: float
    load_ms: float | None
    prompt_eval_ms: float | None
    prompt_tokens: int | None
    eval_ms: float | None
    gen_runs: int | None
    total_ms: float | None
    total_tokens: int | None
    tpot_ms: float | None
    ttft_est_ms: float | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_exe(root: Path) -> Path:
    if sys.platform == "win32":
        names = ("local_llm.exe", "llama-completion.exe")
        rel = ("build", "bin", "Release")
    else:
        names = ("local_llm", "llama-completion")
        rel = ("build", "bin")
    roots = (root, root / "llama.cpp")
    for base in roots:
        for name in names:
            cand = base.joinpath(*rel, name)
            if cand.is_file():
                return cand
    return root.joinpath(*rel, names[0])


def default_model(root: Path) -> Path:
    return root / "models" / "SmolLM2-135M-Instruct-Q4_0.gguf"


# At least 10 cases, increasing length (need.md: 多组不同长度输入)
def builtin_prompts() -> list[str]:
    base = "测"
    return [
        "Hi",
        "Hello, world.",
        "Write one sentence about rain.",
        "请用一句话介绍机器学习。",
        base * 8 + "短句padding",
        base * 32 + "中等长度输入用于prefill",
        base * 64 + "更长一些的上下文占位符重复字符",
        base * 128 + "继续加长以拉高prompt token数量观察prefill",
        "Summarize: " + ("alpha beta gamma delta epsilon " * 12),
        "翻译为英文：" + "人工智能正在改变软件工程。" * 4,
        "编号列表：" + "".join(f"{i}. item; " for i in range(1, 25)),
        base * 256 + "最后一组最长prompt用于压力prefill",
    ]


_RE_LOAD = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")
_RE_PROMPT = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
)
_RE_EVAL = re.compile(
    r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs"
)
_RE_TOTAL = re.compile(
    r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
)


def parse_perf(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if m := _RE_LOAD.search(text):
        out["load_ms"] = float(m.group(1))
    if m := _RE_PROMPT.search(text):
        out["prompt_eval_ms"] = float(m.group(1))
        out["prompt_tokens"] = int(m.group(2))
    if m := _RE_EVAL.search(text):
        out["eval_ms"] = float(m.group(1))
        out["gen_runs"] = int(m.group(2))
    if m := _RE_TOTAL.search(text):
        out["total_ms"] = float(m.group(1))
        out["total_tokens"] = int(m.group(2))
    ev = out.get("eval_ms")
    gr = out.get("gen_runs")
    if ev is not None and gr:
        out["tpot_ms"] = ev / gr
    else:
        out["tpot_ms"] = None
    pe = out.get("prompt_eval_ms")
    if pe is not None and out.get("tpot_ms") is not None:
        out["ttft_est_ms"] = pe + out["tpot_ms"]
    elif pe is not None:
        out["ttft_est_ms"] = pe
    else:
        out["ttft_est_ms"] = None
    return out


def run_case(
    exe: Path,
    model: Path,
    prompt: str,
    n_predict: int,
    ngl: int,
) -> tuple[int, str, float]:
    cmd = [
        str(exe),
        "-m",
        str(model),
        "-no-cnv",
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-ngl",
        str(ngl),
        "--perf",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, out, wall_ms


def build_row(prompt: str, rc: int, combined: str, wall_ms: float) -> PerfRow:
    p = parse_perf(combined)
    prev = prompt.replace("\n", " ")
    if len(prev) > 48:
        prev = prev[:45] + "..."
    return PerfRow(
        prompt_len_chars=len(prompt),
        prompt_preview=prev,
        returncode=rc,
        wall_e2e_ms=round(wall_ms, 3),
        load_ms=p.get("load_ms"),
        prompt_eval_ms=p.get("prompt_eval_ms"),
        prompt_tokens=p.get("prompt_tokens"),
        eval_ms=p.get("eval_ms"),
        gen_runs=p.get("gen_runs"),
        total_ms=p.get("total_ms"),
        total_tokens=p.get("total_tokens"),
        tpot_ms=round(p["tpot_ms"], 4) if p.get("tpot_ms") is not None else None,
        ttft_est_ms=round(p["ttft_est_ms"], 4) if p.get("ttft_est_ms") is not None else None,
    )


def write_charts_png(path: Path, rows: list[PerfRow]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    xs = list(range(1, len(rows) + 1))
    e2e = [r.wall_e2e_ms for r in rows]
    ttft = [r.ttft_est_ms if r.ttft_est_ms is not None else float("nan") for r in rows]
    tpot = [r.tpot_ms if r.tpot_ms is not None else float("nan") for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    titles = [
        "E2E wall latency (ms) — need.md 端到端",
        "TTFT estimate (ms) — prefill + 1 decode step",
        "TPOT (ms/token) — decode phase",
    ]
    series = [e2e, ttft, tpot]
    for ax, ser, tit in zip(axes, series, titles, strict=True):
        ax.bar(xs, ser, color="steelblue")
        ax.set_ylabel("ms")
        ax.set_title(tit)
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xlabel("case # (prompt length increases with index)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def write_markdown(path: Path, rows: list[PerfRow], meta: dict[str, Any]) -> None:
    lines = [
        "# perf_smoke_test",
        "",
        f"- generated: {meta['iso_time']}",
        f"- exe: `{meta['exe']}`",
        f"- model: `{meta['model']}`",
        f"- n_predict: {meta['n_predict']}, ngl: {meta['ngl']}",
    ]
    if meta.get("chart_png"):
        lines.extend(["", f"![perf charts]({meta['chart_png'].name})", ""])
    lines.extend(
        [
            "",
            "| # | len(ch) | wall_E2E_ms | TTFT_est_ms | TPOT_ms | gen_runs | prompt |",
            "|--:|--------:|------------:|------------:|--------:|---------:|--------|",
        ]
    )
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r.prompt_len_chars} | {r.wall_e2e_ms} | {r.ttft_est_ms} | "
            f"{r.tpot_ms} | {r.gen_runs} | {r.prompt_preview} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="llama-completion perf smoke tests")
    ap.add_argument("--root", type=Path, default=root, help="homework repo root")
    ap.add_argument("--exe", type=Path, default=None)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <root>/test_results",
    )
    ap.add_argument(
        "--append-text-report",
        type=Path,
        default=None,
        help="append a [C] section to an existing .txt report",
    )
    ap.add_argument("--no-charts", action="store_true", help="skip matplotlib PNG export")
    args = ap.parse_args()

    exe = args.exe or default_exe(args.root)
    model = args.model or default_model(args.root)
    out_dir = args.out_dir or (args.root / "test_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not exe.is_file():
        print(f"[perf_smoke_test] missing exe: {exe}", file=sys.stderr)
        return 2
    if not model.is_file():
        print(f"[perf_smoke_test] missing model: {model}", file=sys.stderr)
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prompts = builtin_prompts()
    meta = {
        "iso_time": datetime.now(timezone.utc).isoformat(),
        "exe": str(exe),
        "model": str(model),
        "n_predict": args.n_predict,
        "ngl": args.ngl,
        "prompt_count": len(prompts),
    }

    rows: list[PerfRow] = []
    for pr in prompts:
        rc, text, wall = run_case(exe, model, pr, args.n_predict, args.ngl)
        rows.append(build_row(pr, rc, text, wall))

    chart_path = out_dir / f"perf_smoke_{ts}.png"
    if not args.no_charts:
        if write_charts_png(chart_path, rows):
            meta["chart_png"] = chart_path
            print(f"[perf_smoke_test] wrote {chart_path}")
        else:
            print(
                "[perf_smoke_test] matplotlib not installed; "
                "pip install -r requirements.txt for charts",
                file=sys.stderr,
            )

    payload = {
        "meta": {**meta, "chart_png": str(meta["chart_png"]) if meta.get("chart_png") else None},
        "rows": [asdict(r) for r in rows],
    }
    json_path = out_dir / f"perf_smoke_{ts}.json"
    md_path = out_dir / f"perf_smoke_{ts}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, rows, meta)

    print(f"[perf_smoke_test] wrote {json_path}")
    print(f"[perf_smoke_test] wrote {md_path}")

    if args.append_text_report:
        block = []
        block.append("")
        block.append("[C] Python tests/perf_smoke_test.py (>=10 prompt lengths)")
        block.append(f"    JSON: {json_path}")
        block.append(f"    MD:   {md_path}")
        block.append("")
        block.append(md_path.read_text(encoding="utf-8"))
        with open(args.append_text_report, "a", encoding="utf-8", errors="replace") as f:
            f.write("\n".join(block))

    if any(r.returncode != 0 for r in rows):
        print("[perf_smoke_test] warning: some cases returned non-zero", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
