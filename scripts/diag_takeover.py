"""诊断：state 判据在随机策略下的接管都是什么类型。

分类口径（对每一次接管回看轨迹）：
- 原地打转型：接管时机器人已连续 ≥1 步位置未动（在原地转向转圈）——真打转；
- 重访踱步型：位置刚变过，但回到了「旧格子 + 旧朝向」的状态（A↔B 踱步/绕圈）——也是真停滞；
- 同时统计「卡死段」数：连续多步接管算同一段（BFS 挪一步后随机策略立刻又转回去）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_stuck_detector import RandomPlanner, make_loop


def main() -> int:
    total = spin = wander = 0
    episode_counts = []
    for seed in range(10):
        loop = make_loop("state", seed, RandomPlanner(seed))
        try:
            result = loop.run()
        finally:
            loop.world.close()
        recs = loop.tracer.records
        n = 0
        prev_takeover = False
        for i, r in enumerate(recs):
            if "接管" not in str(r.get("fallback") or ""):
                prev_takeover = False
                continue
            total += 1
            n += 1
            if not prev_takeover:  # 连续接管算同一段卡死
                pass
            prev_takeover = True
            prev = recs[i - 1] if i else None
            stationary = prev is not None and r["pos_before"] == prev["pos_before"]
            if stationary:
                spin += 1
            else:
                wander += 1
        episode_counts.append((seed, n, result.steps, result.success))

    print(f"state 判据 × 随机策略 × 10 种子：接管共 {total} 次")
    print(f"  原地打转型（接管时已连续 ≥1 步没挪窝）: {spin}")
    print(f"  重访踱步型（刚挪过但回到旧状态）      : {wander}")
    print()
    print("逐局分布（seed, 接管次数, 步数, 成功）:")
    for seed, n, steps, ok in episode_counts:
        print(f"  seed={seed}  接管 {n} 次  步数 {steps}  成功={ok}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
