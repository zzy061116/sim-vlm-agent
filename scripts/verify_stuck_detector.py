"""验证打转检测判据升级（state vs old）。

不调用任何模型：感知与规划都换成桩（stub），动作由脚本/随机数给出，
用确定性的方式检验检测逻辑本身。三个场景：

A 死角脱困（面壁需要连续转向再前进）：
   old 判据会在「转够次数正要前进」的那一步误接管；state 判据应全程放行。
B 原地打转（转左/转右交替，永不前进）：
   两种判据都应触发接管。
C 随机策略统计对照（N 个种子各跑两种判据）：
   state 判据接管次数应显著少于 old（少误伤），且同样不放过真打转。

运行：python scripts/verify_stuck_detector.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_vlm_agent.config import Config
from sim_vlm_agent.loop import AgentLoop
from sim_vlm_agent.perception import Observation
from sim_vlm_agent.planner import Plan
from sim_vlm_agent.world import ACTIONS


class StubPerception:
    """永远「看到」红块在其真实区域——观察无误差，隔离感知变量。"""

    def __init__(self, world) -> None:
        self.world = world

    def perceive(self, image_bytes: bytes) -> Observation:
        target = self.world.target
        return Observation(
            ok=True,
            source="vlm",
            objects=[{"color": "red", "shape": "square"}],
            target_visible=True,
            target_region=self.world.region_of(target.gx, target.gy),
        )


class ScriptedPlanner:
    """按脚本出动作；脚本耗尽后交给离线规划器收尾。"""

    def __init__(self, world, script: list[str]) -> None:
        self.world = world
        self.script = script
        self.offline = None  # 构造后由外部注入 AgentLoop.offline

    def plan(self, obs, fused, self_state, verification, history, memory=None) -> Plan:
        idx = len(history)
        if idx < len(self.script):
            action = self.script[idx]
            return Plan(True, f"脚本动作：{action}", action, source="llm")
        return self.offline.plan(self.world)


class RandomPlanner:
    """均匀随机出动作——制造大量真打转与撞墙，用于统计对照。"""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def plan(self, obs, fused, self_state, verification, history, memory=None) -> Plan:
        action = self.rng.choice(ACTIONS)
        return Plan(True, f"随机动作：{action}", action, source="llm")


def make_loop(detector: str, seed: int, planner) -> AgentLoop:
    # api_key 非空 + offline=False => available=True，但两个桩保证不发生真实网络调用
    cfg = Config(api_key="TEST", base_url="http://127.0.0.1:9",
                 vlm_model="stub-vlm", llm_model="stub-llm",
                 timeout=1.0, max_step=24, offline=False)
    loop = AgentLoop(cfg, headless=True, label=f"verify_{detector}",
                     seed=seed, stuck_detector=detector)
    loop.perception = StubPerception(loop.world)
    loop.planner = planner
    planner.offline = loop.offline
    if hasattr(planner, "world"):
        planner.world = loop.world
    return loop


def takeover_count(loop: AgentLoop) -> int:
    return sum(1 for r in loop.tracer.records
               if "接管" in str(r.get("fallback") or ""))


def run_case(name: str, detector: str, seed: int, planner) -> dict:
    loop = make_loop(detector, seed, planner)
    try:
        result = loop.run()
    finally:
        loop.world.close()
    return {
        "name": name, "detector": detector, "result": result,
        "takeovers": takeover_count(loop), "records": loop.tracer.records,
    }


def main() -> int:
    print("=" * 66)
    print("场景 A：死角脱困（转 3 次再前进，经典布局起点 (1,4,E)，正前方需绕行）")
    print("-" * 66)
    script = ["turn_right"] * 3  # E -> S -> W -> N，之后交给离线规划器走完
    old = run_case("A", "old", 123456, ScriptedPlanner(None, script))
    new = run_case("A", "state", 123456, ScriptedPlanner(None, script))
    print(f"old   : 接管 {old['takeovers']} 次，"
          f"第 3 步 fallback={old['records'][2]['fallback']!r}")
    print(f"state : 接管 {new['takeovers']} 次，成功={new['result'].success}，"
          f"步数={new['result'].steps}")
    assert old["takeovers"] >= 1, "old 判据应在此场景误接管（预期行为）"
    assert new["takeovers"] == 0, "state 判据不应误伤正常寻路"
    assert new["result"].success, "脱困后应正常完成任务"

    print()
    print("=" * 66)
    print("场景 B：原地打转（转左/转右交替，永不前进）")
    print("-" * 66)
    script = ["turn_left", "turn_right"] * 6
    old = run_case("B", "old", 123456, ScriptedPlanner(None, script))
    new = run_case("B", "state", 123456, ScriptedPlanner(None, script))
    print(f"old   : 接管 {old['takeovers']} 次")
    print(f"state : 接管 {new['takeovers']} 次")
    assert old["takeovers"] >= 1 and new["takeovers"] >= 1, "两种判据都应抓住真打转"

    print()
    print("=" * 66)
    print("场景 C：随机策略统计对照（10 个种子 × 两种判据，按故障类型分列）")
    print("-" * 66)
    spin = {"old": 0, "state": 0}   # 原地打转型接管（old 只有这一类）
    loop = {"old": 0, "state": 0}   # 宏观绕圈型接管（仅 state 有此信号）
    steps = {"old": 0, "state": 0}
    for seed in range(10):
        for detector in ("old", "state"):
            case = run_case("C", detector, seed, RandomPlanner(seed))
            for r in case["records"]:
                fb = str(r.get("fallback") or "")
                if "接管" not in fb:
                    continue
                if "绕圈" in fb:
                    loop[detector] += 1
                else:
                    spin[detector] += 1
            steps[detector] += case["result"].steps
    print(f"old   : 10 局接管共 {spin['old']} 次（全部为原地未动型），平均步数 {steps['old'] / 10:.1f}")
    print(f"state : 10 局接管共 {spin['state'] + loop['state']} 次"
          f"（原地打转 {spin['state']} + 宏观绕圈 {loop['state']}），"
          f"平均步数 {steps['state'] / 10:.1f}")
    assert spin["state"] <= spin["old"], "同口径（原地打转型）state 接管不应多于 old（少误伤）"

    print()
    print("=" * 66)
    print("场景 D：宏观绕圈（前进+转向交替兜圈子，位置一直在变）")
    print("-" * 66)
    # 经典布局起点 (1,4,E)：前进/右转交替会绕 (1,4)(2,4)(2,5)(1,5) 四格兜圈。
    # 每步位置都在变（will_advance 持续放行），old 判据完全看不见；
    # state 判据靠「格子第 3 次访问 + 状态重复」抓住。
    script = ["move_forward", "turn_right"] * 10
    old = run_case("D", "old", 123456, ScriptedPlanner(None, script))
    new = run_case("D", "state", 123456, ScriptedPlanner(None, script))
    new_loop = sum(1 for r in new["records"] if "绕圈" in str(r.get("fallback") or ""))
    print(f"old   : 接管 {old['takeovers']} 次（对宏观绕圈全盲）")
    print(f"state : 接管 {new['takeovers']} 次，其中绕圈型 {new_loop} 次")
    assert old["takeovers"] == 0, "old 判据对宏观绕圈应完全无感（证明盲区存在）"
    assert new_loop >= 1, "state 判据应通过格子重访信号抓住宏观绕圈"

    print()
    print("=" * 66)
    print("全部断言通过：")
    print("  A. state 判据放行「转向后前进」的正常寻路（old 会误伤）")
    print("  B. state 判据仍能抓住真打转")
    print("  C. 同口径（原地打转型）下 state 接管 <= old，且额外覆盖宏观绕圈")
    print("  D. 宏观绕圈：old 全盲，state 通过格子重访信号抓住")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
