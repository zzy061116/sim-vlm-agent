"""把一次运行的轨迹打印成表格，用来定位「模型为什么走错」。

    python scripts/inspect_trace.py            # 看最近一次
    python scripts/inspect_trace.py runs/xxx   # 看指定目录
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def latest_run() -> Path:
    runs = sorted((ROOT / "runs").glob("run_*")) + sorted((ROOT / "runs").glob("llm_*"))
    runs += sorted((ROOT / "runs").glob("offline_*"))
    dirs = [p for p in runs if p.is_dir() and (p / "trace.jsonl").exists()]
    if not dirs:
        raise SystemExit("没有找到轨迹，先跑一次 python run.py")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_raw = "--raw" in sys.argv
    target = Path(args[0]) if args else latest_run()
    lines = (target / "trace.jsonl").read_text(encoding="utf-8").splitlines()

    if show_raw:
        for line in lines[:3]:
            record = json.loads(line)
            print(f"--- STEP {record['step']} 原始输出 ---")
            print(record["observation"].get("raw", "")[:700])
        return

    header = (f"{'步':>2} {'位置':>9} {'朝向':>5} | "
              f"{'物体会':<6} {'VLM区':<14} {'换算方位':<12} {'真值':<12} {'距离':<5} {'堵':<6} | "
              f"{'动作':<13} 结果")
    print(target)
    print(header)
    print("-" * len(header))
    for line in lines:
        r = json.loads(line)
        obs = r["observation"]
        fused = r.get("fused", {})
        ver = r["verification"]
        plan = r["plan"]
        res = r["result"]
        status = "ok" if res["ok"] else res["reason"]
        pos = f"{r['pos_before']}->{r['pos_after']}"
        facing = f"{r['facing_before']}->{r['facing_after']}"
        obj = ",".join(sorted({o["color"][0] for o in obs.get("objects", [])})) or "-"
        print(f"{r['step']:>2} {pos:>9} {facing:>5} | "
              f"{obj:<6} {obs.get('target_region', '-'):<14} "
              f"{fused.get('target_bearing', '-'):<12} {ver.get('true_bearing', '-'):<12} "
              f"{fused.get('target_distance', '-'):<5} {str(fused.get('front_blocked')):<6} | "
              f"{plan['action']:<13} {status}")
        if r.get("fallback"):
            print(f"   ! {r['fallback']}")

    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    print("-" * len(header))
    print(f"结果 {summary['success']} | 步数 {summary['steps']} | "
          f"一致率 {summary['consistent_rate']}% | 降级 {summary['fallbacks']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
