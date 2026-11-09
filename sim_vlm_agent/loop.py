"""模块 4：主循环调度器。

while 未完成 and 轮次 < MAX_STEP:
    截图 -> VLM 感知 -> 校验 -> LLM 规划(Function Calling) -> 执行 -> 记录

几条防止跑飞的工程约束：
1. 轮次上限，绝不无限循环；
2. VLM 失败或观察长期不可信时，退回程序侧观察；
3. 打转检测触发时改由离线规划器走一步，把局面带开（默认「状态重复」判据，
   --stuck old 可退回旧的「连续位置未变」判据做对照）；
4. 任何一轮出错都不抛异常到顶层，全部记进轨迹。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .perception import Observation, Perception, fuse
from .planner import OfflinePlanner, Planner, TASK
from .trace import Tracer
from .world import World


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    reason: str
    run_dir: Path
    summary: dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    def __init__(
        self,
        cfg: Config,
        headless: bool = True,
        delay: float = 0.0,
        label: str = "run",
        seed: int | None = None,
        stuck_detector: str = "state",
    ) -> None:
        if stuck_detector not in ("state", "old"):
            raise ValueError(f"未知打转检测判据：{stuck_detector!r}")
        self.stuck_detector = stuck_detector
        self.cfg = cfg
        # seed=None：每次运行随机生成出生点/朝向/内墙；传 seed 可复现
        self.world = World(headless=headless, seed=seed)
        self.perception = Perception(cfg)
        self.planner = Planner(cfg)
        self.offline = OfflinePlanner()
        self.tracer = Tracer(label)
        self.delay = delay

    # ---------- 降级：用程序侧真值构造一份观察 ----------
    @staticmethod
    def _truth_observation(world: World, error: str) -> Observation:
        truth = world.get_state()
        return Observation(
            ok=False,
            source="fallback",
            objects=[{"color": o["color"], "shape": o["shape"]} for o in truth["objects"]],
            target_visible=True,
            target_region=truth["target"]["region"],
            error=error,
        )

    def run(self) -> EpisodeResult:
        cfg, world, tracer = self.cfg, self.world, self.tracer
        world.reset()

        history: list[dict[str, Any]] = []
        consistent = 0
        vlm_calls = 0
        fallbacks = 0
        takeovers = 0
        api_failures = 0
        success = False
        reason = "max_step_reached"
        still = 0
        last_pos: tuple[int, int] | None = None
        seen_states: set[tuple[int, int, str]] = set()
        state_stuck = 0
        # 撞墙记忆：真实撞过墙的 (格子 x, 格子 y, 移动方向)。只记事实，不做推断，
        # 与「传感器读数每步可见」互补——记忆管历史，传感管当下。
        known_blocked: set[tuple[int, int, str]] = set()
        pos_history: list[tuple[int, int]] = []
        # 格子访问计数：状态重复判据的盲区是「宏观绕圈」——位置一直在变、
        # 每步都是可执行前进（state_stuck 被持续清零），但格子集不扩张。
        # 同一格被访问 ≥3 次即视为绕圈信号。
        visits: dict[tuple[int, int], int] = {}

        while True:
            if world.done:
                success, reason = True, "grabbed_target"
                break
            if len(history) >= cfg.max_step:
                reason = "max_step_reached"
                break
            if world.poll_quit():
                reason = "quit_by_user"
                break

            step = len(history) + 1
            truth = world.get_state()
            image = world.get_screenshot()
            image_name = tracer.save_image(image, step)

            # 1) 感知
            note = ""
            if cfg.available:
                vlm_calls += 1
                obs = self.perception.perceive(image)
                if not obs.ok:
                    obs = self._truth_observation(world, obs.error)
                    note = "VLM 调用失败，退回程序侧观察"
                    api_failures += 1
            else:
                obs = self._truth_observation(world, "离线模式")
                note = "离线模式"

            fused = fuse(obs, truth)
            verification = Perception.verify(obs, fused, truth)
            if verification["consistent"]:
                consistent += 1

            # 1.5) 进展统计（先于规划：停滞计数与撞墙记忆要喂进 prompt）
            pos = (truth["agent"]["gx"], truth["agent"]["gy"])
            state = (pos[0], pos[1], truth["agent"]["dir"])
            still = still + 1 if last_pos == pos else 0
            last_pos = pos
            state_stuck = state_stuck + 1 if state in seen_states else 0
            seen_states.add(state)
            visits[pos] = visits.get(pos, 0) + 1
            memory = {
                "known_blocked": sorted(known_blocked),
                "pos_history": pos_history[-8:],
                "still": still,
            }

            # 2) 规划
            if cfg.available and obs.source == "vlm":
                plan = self.planner.plan(obs, fused, truth, verification, history, memory)
                if not plan.ok:
                    fallback_plan = self.offline.plan(world)
                    note = note or f"规划失败({plan.error})，改用离线规划"
                    api_failures += 1  # 规划调用失败/未返回工具调用，与接管分开算
                    plan = fallback_plan
            else:
                plan = self.offline.plan(world)
                note = note or "离线规划"

            # 3) 打转检测（两种判据，--stuck 选择；计数已在 1.5 提前算好）：
            #    old  ：连续 N 步位置未变即接管。缺陷：转向不改位置，三面被堵时
            #           「转 2~3 次再前进」的正常寻路会在第 3 步被误伤接管。
            #    state：进展 = 位置变化 或 出现新的 (位置,朝向) 状态。
            #           正常寻路时状态从不重复；真打转（原地转圈/来回踱步）因
            #           朝向只有 4 个取值，状态必然周期性重复。且本步计划是
            #           「可执行的前进」（move_forward 且前方无墙）时视为正在
            #           收敛，直接放行——这正是死角脱困的第 3 步。
            will_advance = plan.action == "move_forward" and not truth["front_blocked"]
            if will_advance:
                state_stuck = 0

            stuck_hit = False
            if self.stuck_detector == "old":
                if still >= 2 and plan.source != "offline":
                    stuck_hit = True
                    note = f"连续 {still + 1} 步位置未变，离线规划接管"
            elif plan.source != "offline" and (state_stuck >= 2
                    or (visits[pos] >= 3 and state_stuck >= 1)):
                # 两类停滞信号：
                #   a) 状态重复 >=2 次：原地打转/踱步；
                #   b) 格子第 3 次访问且当前状态也是重复：宏观绕圈（位置一直在变、
                #      每步都是可执行前进，但空间上在兜圈子）。必须叠加状态重复，
                #      否则出生点原地转三圈（状态全新）会被误伤。
                stuck_hit = True
                note = (f"状态 {state} 重复 {state_stuck + 1} 次，打转判定接管"
                        if state_stuck >= 2 else
                        f"格子 {pos} 第 {visits[pos]} 次访问，疑似绕圈，离线规划接管")
            if stuck_hit:
                plan_before_takeover = plan.to_dict()  # 保真：记下模型原本想做什么
                plan = self.offline.plan(world)

            if plan.source == "offline" and cfg.available:
                fallbacks += 1
                if stuck_hit:
                    takeovers += 1

            # 4) 执行
            result = world.exec_action(plan.action)
            after = world.get_state()

            record = {
                "step": step,
                "image": image_name,
                "observation": {**obs.to_dict(), "brief": obs.brief()},
                "fused": {**fused.to_dict(), "brief": fused.brief()},
                "verification": verification,
                "plan": plan.to_dict(),
                **({"plan_before_takeover": plan_before_takeover}
                   if stuck_hit else {}),
                "result": result.to_dict(),
                "facing_before": truth["agent"]["dir"],
                "facing_after": after["agent"]["dir"],
                "pos_before": f"{truth['agent']['gx']},{truth['agent']['gy']}",
                "pos_after": f"{after['agent']['gx']},{after['agent']['gy']}",
                "fallback": note,
            }
            tracer.add(record)
            history.append({
                "step": step,
                "action": plan.action,
                "ok": result.ok,
                "reason": result.reason,
            })

            # 5) 更新撞墙记忆与位置轨迹：只记「真实撞过」的 (格, 方向)，
            #    这是模型下一轮 prompt 里「必定可信」的那部分信息
            if (result.action == "move_forward" and not result.ok
                    and result.reason == "blocked_by_wall"):
                known_blocked.add(
                    (result.before["gx"], result.before["gy"], result.before["dir"]))
            pos_history.append((after["agent"]["gx"], after["agent"]["gy"]))

            if self.delay:
                time.sleep(self.delay)

        steps = len(history)
        summary = {
            "success": success,
            "reason": reason,
            "steps": steps,
            "task": TASK,
            "world": world.layout(),  # 随机布局的完整快照 + 复现种子
            "vlm_model": cfg.vlm_model if cfg.available else "offline",
            "llm_model": cfg.llm_model if cfg.available else "offline",
            "stuck_detector": self.stuck_detector,
            "wall_memory": len(known_blocked),
            "vlm_calls": vlm_calls,
            "consistent_rate": round(100 * consistent / steps, 1) if steps else 0.0,
            "fallbacks": fallbacks,
            "takeovers": takeovers,        # 打转/绕圈判定接管（安全网触发）
            "api_failures": api_failures,  # VLM/LLM 调用失败降级（基础设施问题）
            "final_state": world.get_state(),
        }
        run_dir = tracer.finish(summary)
        return EpisodeResult(success, steps, reason, run_dir, summary)
