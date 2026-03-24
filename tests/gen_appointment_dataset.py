#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate 100 positive + 100 negative synthetic 「任命证书」 samples (need.md)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _repo_root() -> Path:
    # tests/<this>.py under self-contained package root (folder f)
    return Path(__file__).resolve().parent.parent


def positive_cert(i: int) -> str:
    orgs = ["市第一人民医院", "某大学附属中学", "区市场监督管理局", "省科学技术协会", "集团有限公司人力资源部"]
    roles = ["科室主任", "项目经理", "教研组长", "高级工程师", "护士长", "团委副书记"]
    names = ["张伟", "李娜", "王强", "刘洋", "陈静", "赵敏", "孙磊", "周婷"]
    return (
        f"任命证书\n\n"
        f"兹任命 {names[i % len(names)]}{i % 10} 同志担任 {orgs[i % len(orgs)]} {roles[i % len(roles)]}，"
        f"任期自二〇{(20 + (i % 5)):02d}年{(1 + (i % 12)):02d}月{(1 + (i % 28)):02d}日起至"
        f"二〇{(22 + (i % 4)):02d}年{(1 + (i % 12)):02d}月止。\n\n"
        f"特此任命。\n\n"
        f"（公章）\n"
        f"二〇{(20 + (i % 5)):02d}年{(3 + (i % 9)):02d}月{(10 + (i % 18)):02d}日"
    )


def negative_cert(i: int) -> str:
    # 明显逻辑/表达问题：时间倒置、自我任命、职务矛盾、荒诞日期等
    org = f"某单位{i % 7}"
    flaws = [
        f"任命证书\n\n兹任命本人为本人上级领导，自即日起生效。\n\n{org}\n二〇三〇年一月一日",
        f"任命证书\n\n由于天气原因，任命张伟为去年已经撤销的职务「不存在科科长{i}」。\n\n"
        f"二〇二五年二月三十日\n{org}",
        f"任命证书\n\n任命：：：！！！？？？职务不明人员担任不明职务，无单位无日期。编号{i}",
        f"任命证书\n\n兹任命李娜同志担任总经理，同时担任反对担任总经理的监察员，"
        f"两职互相否定，自二〇二六年十三月四十五日起生效。\n\n{org}",
        f"任命证书\n\n该同志已于二〇二〇年退休，现任命其自二〇一〇年起全职在岗担任全职退休顾问（逻辑矛盾）。\n\n"
        f"二〇二四年某日 {org}",
        f"任命证书\n\n先落款后任命：\n（公章）二〇二一年一月一日\n\n兹任命王强为尚未成立的第{i}号幽灵部门负责人。",
        f"任命证书\n\n空白任命：兹任命        担任        ，日期留空，单位留空。编号{i}",
        f"任命证书\n\n circular: A任命B，B任命C，C任命A，三人互为唯一上级（编号{i}）。",
    ]
    return flaws[i % len(flaws)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = args.out or (_repo_root() / "data" / "appointment_cert_dataset.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    for i in range(100):
        rows.append({"id": f"pos_{i:03d}", "label": 1, "text": positive_cert(i)})
    for i in range(100):
        rows.append({"id": f"neg_{i:03d}", "label": 0, "text": negative_cert(i)})
    rng.shuffle(rows)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gen_appointment_dataset] wrote {len(rows)} lines -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
