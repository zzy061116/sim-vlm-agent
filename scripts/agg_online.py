# -*- coding: utf-8 -*-
"""聚合 6 次在线评测的 summary.json，输出 state vs old 对比表。"""
import glob
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

rows = []
for p in sorted(glob.glob(str(Path(__file__).resolve().parent.parent / "runs" / "stuck_*_s*" / "summary.json"))):
    rows.append((Path(p).parent.name, json.load(open(p, encoding="utf-8"))))

print(f"{'run':<18}{'detector':<10}{'success':<9}{'steps':<7}{'fallbacks':<11}{'consistent%':<13}{'elapsed'}")
for name, s in rows:
    print(f"{name:<18}{s['stuck_detector']:<10}{str(s['success']):<9}{s['steps']:<7}{s['fallbacks']:<11}{s['consistent_rate']:<13}{s['elapsed']}")

for det in ("state", "old"):
    group = [s for _, s in rows if s["stuck_detector"] == det]
    if group:
        print(f"\n{det}: 成功 {sum(1 for s in group if s['success'])}/{len(group)}, "
              f"总步数 {sum(s['steps'] for s in group)}, 总降级 {sum(s['fallbacks'] for s in group)}")
