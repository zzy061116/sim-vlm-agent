"""模块 4：主循环调度器。

while 未完成 and 轮次 < MAX_STEP:
    截图 -> VLM 感知 -> 校验 -> LLM 规划(Function Calling) -> 执行 -> 记录

几条防止跑飞的工程约束：
1. 轮次上限，绝不无限循环；
2. VLM 失败或观察长期不可信时，退回程序侧观察；
3. 连续无效动作就判定「卡住」，改由离线规划器走一步，把局面带开；
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
    ) -> None:
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
        success = False
        reason = "max_step_reached"
        still = 0
        last_pos: tuple[int, int] | None = None

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
            else:
                obs = self._truth_observation(world, "离线模式")
                note = "离线模式"

            fused = fuse(obs, truth)
            verification = Perception.verify(obs, fused, truth)
            if verification["consistent"]:
                consistent += 1

            # 2) 规划
            if cfg.available and obs.source == "vlm":
                plan = self.planner.plan(obs, fused, truth, verification, history)
                if not plan.ok:
                    fallback_plan = self.offline.plan(world)
                    note = note or f"规划失败({plan.error})，改用离线规划"
                    plan = fallback_plan
            else:
                plan = self.offline.plan(world)
                note = note or "离线规划"

            # 3) 打转检测：连续几步位置都没变（转向不算进展），就强制让离线规划带一步
            pos = (truth["agent"]["gx"], truth["agent"]["gy"])
            if last_pos == pos:
                still += 1
            else:
                still = 0
            last_pos = pos
            if still >= 2 and plan.source != "offline":
                plan = self.offline.plan(world)
                note = f"连续 {still + 1} 步位置未变，离线规划接管"

            if plan.source == "offline" and cfg.available:
                fallbacks += 1

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
            "vlm_calls": vlm_calls,
            "consistent_rate": round(100 * consistent / steps, 1) if steps else 0.0,
            "fallbacks": fallbacks,
            "final_state": world.get_state(),
        }
        run_dir = tracer.finish(summary)
        return EpisodeResult(success, steps, reason, run_dir, summary)
