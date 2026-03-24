#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate model on appointment certificate dataset (need.md).
Labels: 1=合理(正例), 0=不合理(反例). Reports accuracy, precision, recall, F1 for class 合理.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_exe(root: Path) -> Path:
    for cand in (
        root / "build" / "bin" / "Release" / "local_llm.exe",
        root / "llama.cpp" / "build" / "bin" / "Release" / "local_llm.exe",
        root / "llama.cpp" / "build" / "bin" / "Release" / "llama-completion.exe",
    ):
        if cand.is_file():
            return cand
    return root / "build" / "bin" / "Release" / "local_llm.exe"


def default_model(root: Path) -> Path:
    return root / "models" / "SmolLM2-135M-Instruct-Q4_0.gguf"


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def build_prompt(text: str) -> str:
    return (
        "任务：判断下面这段「任命证书」在中文表达与常识逻辑上是否合理。"
        "若整体通顺且职务、时间、主体无明显矛盾，回答：合理。"
        "否则回答：不合理。只输出一个词：合理 或 不合理，不要解释。\n\n"
        f"文书：\n{text}\n\n答案："
    )


def parse_prediction(raw: str) -> int | None:
    """Return 1 if 合理, 0 if 不合理, None if unknown."""
    s = raw.replace(" ", "").lower()
    # Prefer longer match first
    if "不合理" in raw or "notreasonable" in s or "invalid" in s:
        return 0
    if "合理" in raw and "不合理" not in raw:
        return 1
    if re.search(r"\bok\b|\byes\b|\bvalid\b", s):
        return 1
    if re.search(r"\bno\b|\bfalse\b", s):
        return 0
    return None


def run_llm(exe: Path, model: Path, prompt: str, n_predict: int, ngl: int) -> str:
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
        "--no-warmup",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def extract_answer_tail(combined: str, prompt_marker: str = "答案：") -> str:
    if prompt_marker in combined:
        return combined.split(prompt_marker, 1)[-1][-400:]
    return combined[-400:]


@dataclass
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int

    def accuracy(self) -> float:
        t = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / t if t else 0.0

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate(
    rows: Iterator[dict],
    exe: Path,
    model: Path,
    n_predict: int,
    ngl: int,
    sleep_s: float,
    limit: int | None,
) -> tuple[Metrics, list[dict]]:
    m = Metrics(0, 0, 0, 0)
    details: list[dict] = []
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        y = int(row["label"])
        pr = build_prompt(row["text"])
        t0 = time.perf_counter()
        out = run_llm(exe, model, pr, n_predict, ngl)
        dt = time.perf_counter() - t0
        tail = extract_answer_tail(out)
        pred = parse_prediction(tail)
        details.append(
            {
                "id": row.get("id"),
                "label": y,
                "pred": pred,
                "latency_s": round(dt, 3),
                "tail": tail[:200],
            }
        )
        if pred is None:
            # 无法解析时按「未明确判定为合理」处理 -> 记为预测类 0
            if y == 1:
                m.fn += 1
            else:
                m.tn += 1
        elif pred == 1 and y == 1:
            m.tp += 1
        elif pred == 1 and y == 0:
            m.fp += 1
        elif pred == 0 and y == 0:
            m.tn += 1
        else:
            m.fn += 1
        if sleep_s > 0:
            time.sleep(sleep_s)
    return m, details


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=root / "data" / "appointment_cert_dataset.jsonl")
    ap.add_argument("--exe", type=Path, default=None)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--n-predict", type=int, default=8)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N samples")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--out-txt",
        type=Path,
        default=None,
        help="short human-readable summary (accuracy, P/R/F1)",
    )
    args = ap.parse_args()

    exe = args.exe or default_exe(root)
    model = args.model or default_model(root)
    if not exe.is_file():
        print(f"[accuracy_eval] missing exe: {exe}", file=sys.stderr)
        return 2
    if not model.is_file():
        print(f"[accuracy_eval] missing model: {model}", file=sys.stderr)
        return 2
    if not args.dataset.is_file():
        print(
            "[accuracy_eval] missing dataset. Run: python tests/gen_appointment_dataset.py",
            file=sys.stderr,
        )
        return 2

    rows = load_jsonl(args.dataset)
    metrics, details = evaluate(
        iter(rows), exe, model, args.n_predict, args.ngl, args.sleep, args.limit
    )

    report = {
        "exe": str(exe),
        "model": str(model),
        "samples": len(details),
        "confusion": {"tp": metrics.tp, "fp": metrics.fp, "tn": metrics.tn, "fn": metrics.fn},
        "accuracy": round(metrics.accuracy(), 4),
        "precision_positive": round(metrics.precision(), 4),
        "recall_positive": round(metrics.recall(), 4),
        "f1_positive": round(metrics.f1(), 4),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    outj = args.out_json
    if outj:
        outj.parent.mkdir(parents=True, exist_ok=True)
        outj.write_text(
            json.dumps({"metrics": report, "details": details}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[accuracy_eval] wrote {outj}", file=sys.stderr)

    if args.out_txt:
        c = report["confusion"]
        lines = [
            "任务：任命证书是否合理（正类=「合理」，数据集标签 1）",
            f"评测样本数: {len(details)}",
            f"准确率: {report['accuracy']}",
            f"精确率（正类为「合理」）: {report['precision_positive']}",
            f"召回率（正类为「合理」）: {report['recall_positive']}",
            f"F1分数（正类为「合理」）: {report['f1_positive']}",
            "混淆矩阵: "
            f"TP(真且判合理)={c['tp']} "
            f"FP(伪判成合理)={c['fp']} "
            f"TN(真不合理且判不合理)={c['tn']} "
            f"FN(真合理却未判合理)={c['fn']}",
        ]
        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        args.out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[accuracy_eval] wrote {args.out_txt}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
