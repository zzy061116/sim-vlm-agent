"""批量在线评测：接管概率测量 + 逐次接管归因。

用法：python scripts/batch_eval.py <seed1> <seed2> ...
不给 seed 则用 10 个固定随机种子。

每个 seed 在线跑一局（真实 VLM+LLM），结束后输出：
- 成功/步数/接管次数/API 失败次数；
- 逐次接管的归因：类型（打转/绕圈）+ 接管时的方位/堵墙模式/模型当时的想法。
结果同时写入 runs/batch_eval_<时间戳>.json，供跨批次对比。
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from sim_vlm_agent.config import load_config
from sim_vlm_agent.loop import AgentLoop

DEFAULT_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 101]


def takeover_cases(records: list[dict]) -> list[dict]:
    """把每条「接管」记录提炼成归因案例。"""
    cases = []
    for r in records:
        fb = r.get("fallback") or ""
        if "接管" not in fb:
            continue
        fused = r["fused"]
        cases.append({
            "step": r["step"],
            "type": "loop" if "绕圈" in fb else "spin",
            "pos": r["pos_before"],
            "facing": r["facing_before"],
            "bearing": fused.get("target_bearing"),
            "blocked": (fused.get("front_blocked"), fused.get("left_blocked"),
                        fused.get("right_blocked")),
            "model_action": r["plan"]["action"],
            "thought": r["plan"]["thought"][:60],
            "guard": r["plan"].get("guard") or "",
        })
    return cases


def run_one(seed: int) -> dict:
    cfg = load_config()
    loop = AgentLoop(cfg, headless=True, label=f"eval_s{seed}", seed=seed)
    try:
        result = loop.run()
    finally:
        loop.world.close()
    recs = loop.tracer.records
    cases = takeover_cases(recs)
    s = result.summary
    return {
        "seed": seed,
        "success": result.success,
        "steps": s["steps"],
        "takeovers": s.get("takeovers", 0),
        "api_failures": s.get("api_failures", 0),
        "cases": cases,
        "run_dir": str(result.run_dir),
    }


def main() -> int:
    seeds = [int(a) for a in sys.argv[1:]] or DEFAULT_SEEDS
    results = []
    for i, seed in enumerate(seeds, 1):
        started = time.time()
        try:
            r = run_one(seed)
        except Exception as exc:
            r = {"seed": seed, "error": f"{type(exc).__name__}: {exc}"[:120]}
        r["elapsed"] = round(time.time() - started, 1)
        results.append(r)
        if "error" in r:
            print(f"[{i}/{len(seeds)}] seed={seed} 出错: {r['error']}", flush=True)
            continue
        print(f"[{i}/{len(seeds)}] seed={seed:<4d} 成功={r['success']} "
              f"步数={r['steps']:<3d} 接管={r['takeovers']} "
              f"api失败={r['api_failures']} 耗时={r['elapsed']}s", flush=True)

    ok = [r for r in results if r.get("success")]
    total_steps = sum(r["steps"] for r in results if "error" not in r)
    total_takeovers = sum(r["takeovers"] for r in results if "error" not in r)
    total_api = sum(r["api_failures"] for r in results if "error" not in r)
    spins = [c for r in results for c in r.get("cases", []) if c["type"] == "spin"]
    loops = [c for r in results for c in r.get("cases", []) if c["type"] == "loop"]

    print()
    print("=" * 70)
    print(f"成功 {len(ok)}/{len(results)}  总步数 {total_steps}  "
          f"总接管 {total_takeovers}  总 API 失败 {total_api}")
    print(f"接管归因：原地打转 {len(spins)} 次 / 宏观绕圈 {len(loops)} 次")

    def dist(items, key):
        d: dict[str, int] = {}
        for c in items:
            d[str(c[key])] = d.get(str(c[key]), 0) + 1
        return dict(sorted(d.items(), key=lambda kv: -kv[1]))

    if spins:
        print(f"打转型方位分布：{dist(spins, 'bearing')}")
        print(f"打型堵墙模式(前,左,右)：{dist(spins, 'blocked')}")
    if loops:
        print(f"绕圈型方位分布：{dist(loops, 'bearing')}")
        print(f"绕圈堵墙模式(前,左,右)：{dist(loops, 'blocked')}")

    print()
    print("接管案例明细：")
    for r in results:
        for c in r.get("cases", []):
            print(f"  seed={r['seed']} step={c['step']} [{c['type']}] "
                  f"pos={c['pos']} 朝向={c['facing']} 目标={c['bearing']} "
                  f"堵(前/左/右)={c['blocked']} 模型想: {c['model_action']} | {c['thought']}")

    out = Path("runs") / f"batch_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n明细已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
