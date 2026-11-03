"""连通性与能力自检：确认视觉模型、规划模型都能用，并打印它们的真实表现。

    python scripts/check_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_vlm_agent.config import load_config
from sim_vlm_agent.perception import Perception, fuse
from sim_vlm_agent.planner import Planner
from sim_vlm_agent.world import World


def main() -> int:
    cfg = load_config()
    print("=" * 60)
    print("模型自检")
    print("-" * 60)
    print(f"接口地址  : {cfg.base_url}")
    print(f"视觉模型  : {cfg.vlm_model}")
    print(f"规划模型  : {cfg.llm_model}")
    print(f"密钥      : {cfg.masked_key()}")
    print(f"离线模式  : {cfg.offline}")
    print("=" * 60)

    if not cfg.available:
        print("没有可用的密钥，程序会以离线模式运行（不调用任何模型）。")
        print("把 LLM_API_KEY 填进 .env 后重试。")
        return 0

    world = World(headless=True)
    truth = world.get_state()

    print("\n[1/2] 视觉模型感知当前画面 …")
    obs = Perception(cfg).perceive(world.get_screenshot())
    print(f"  耗时      : {obs.latency:.1f}s")
    print(f"  状态      : ok={obs.ok} source={obs.source}")
    if obs.error:
        print(f"  错误      : {obs.error}")
    print(f"  原始返回  : {(obs.raw or '(空)')[:360]}")
    print(f"  物体      : {json.dumps(obs.objects, ensure_ascii=False)}")
    print(f"  目标区域  : {obs.target_region}（visible={obs.target_visible}）")
    fused = fuse(obs, truth)
    print(f"  融合结果  : {fused.brief()}")
    print(f"  真实情况  : 方位 {truth['target']['bearing']} / 区域 {truth['target']['region']}")
    verdict = Perception.verify(obs, fused, truth)
    print(f"  一致性    : {verdict['consistent']}"
          f"（幻觉 {verdict['hallucinated'] or '无'} / 漏检 {verdict['missed'] or '无'}）")

    print("\n[2/2] 规划模型决策 …")
    plan = Planner(cfg).plan(obs, fused, truth, verdict, [])
    print(f"  耗时      : {plan.latency:.1f}s")
    print(f"  状态      : ok={plan.ok} source={plan.source}")
    if plan.error:
        print(f"  错误      : {plan.error}")
    print(f"  调用工具  : {plan.action}")
    print(f"  理由      : {plan.thought or '(空)'}")

    print("\n" + "=" * 60)
    ok = obs.ok and plan.ok
    print("结论：" + ("两个模型都可用，可以开始跑闭环。" if ok
                     else "有环节不可用，检查上面的错误信息。"))
    world.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
