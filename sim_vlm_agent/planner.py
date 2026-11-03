"""模块 3：LLM 规划 + Function-Calling。

输入：视觉观察（定性方位）+ 机器人自身里程计 + 最近几步动作
输出：一句思考 + 一次工具调用

刻意不让规划层看到目标的真实坐标：它必须依据「看见的东西」决策，
否则 VLM 就成了摆设，面试时一追问就穿。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Config
from .perception import FusedView, Observation

TASK = "在房间中找到红色方块，走到它的上下左右任意一格，然后抓取它。"

ACTION_DOC = {
    "move_forward": "沿当前朝向前进一格。前方是墙或边界时会失败，位置不变。",
    "turn_left": "原地左转 90 度（逆时针）。位置不变，只改朝向。",
    "turn_right": "原地右转 90 度（顺时针）。位置不变，只改朝向。",
    "grab_object": "抓取目标。只有当机器人已经站在红色方块旁边（距离 1 格内）才会成功。",
}

# 每个工具都强制带一个 reason 参数：模型走 tool_calls 时 content 通常为空，
# 把推理过程塞进参数里，轨迹报告才有「思考」可展示。
_REASON_PARAM = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "用一句中文说明为什么选这个动作，包含你观察到的目标方位",
        }
    },
    "required": ["reason"],
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": ACTION_DOC[name],
            "parameters": _REASON_PARAM,
        },
    }
    for name in ("move_forward", "turn_left", "turn_right", "grab_object")
]

SYSTEM_PROMPT = f"""你是一台在网格房间里行动的小机器人的决策模块。

任务：{TASK}

可用工具（每次只能调用一个）：
- move_forward：{ACTION_DOC['move_forward']}
- turn_left：{ACTION_DOC['turn_left']}
- turn_right：{ACTION_DOC['turn_right']}
- grab_object：{ACTION_DOC['grab_object']}

【动作语义，务必分清】
- turn_left / turn_right 只改变朝向，机器人位置完全不动。
- 只有 move_forward 会改变位置。
所以「目标在侧方就一直转向」是错的：转完必须前进，否则永远原地打转。

【决策表】从第一行往下逐行比对，命中哪一行就执行哪一行，
不允许用「距离太远」「想先看看别处」之类的理由跳过命中的行：

| 条件 | 必须调用 |
|---|---|
| already_beside_target 为 true | grab_object |
| target_bearing 是 here（目标就在身边，方向分辨不出来），且 front_blocked 为 false | move_forward |
| target_bearing 是 front / front-left / front-right，且 front_blocked 为 false | move_forward |
| front_blocked 为 true | 转向（往目标所在的那一侧转） |
| target_bearing 是 left | turn_left |
| target_bearing 是 right | turn_right |
| target_bearing 是 back-left | turn_left |
| target_bearing 是 back-right | turn_right |
| target_bearing 是 back | 任意转向一次 |
| 上面都不成立，且连续转向已 ≥ 2 次 | move_forward |

补充说明：
- 斜前方（front-left / front-right）不需要先把朝向转正，直接前进一格就行，
  走过去之后目标方位会自己变化。用 90 度转向去对准 45 度的目标，只会来回震荡。
- front_blocked 为 true 说明下一步就是墙，这时 move_forward 必然失败，必须先转向。
- 观察被判为不可信时，先做一个转向换视角，但下一步必须回到决策表。

【硬性约束】
- 只依据给你的观察判断，禁止假设画面里没提到的物体存在。
- 一步只做一个动作，必须通过工具调用表达。
- reason 里要写清：你看到什么，命中了决策表哪一行。"""


@dataclass
class Plan:
    ok: bool
    thought: str
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    source: str = "llm"  # llm | offline
    raw: str = ""
    error: str = ""
    guard: str = ""  # 动作校验层对模型输出做过的纠正
    latency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "thought": self.thought,
            "action": self.action,
            "args": self.args,
            "source": self.source,
            "error": self.error,
            "guard": self.guard,
            "latency": round(self.latency, 2),
        }


def _consecutive_turns(history: list[dict[str, Any]]) -> int:
    count = 0
    for item in reversed(history):
        if item.get("action") in ("turn_left", "turn_right"):
            count += 1
        else:
            break
    return count


def _history_text(history: list[dict[str, Any]], keep: int = 5) -> str:
    recent = history[-keep:]
    if not recent:
        return "（这是第一步，还没有动作历史）"
    lines = []
    for item in recent:
        mark = "成功" if item.get("ok") else f"失败({item.get('reason')})"
        lines.append(f"第{item['step']}步 {item['action']} -> {mark}")
    return "\n".join(lines)


class Planner:
    """真实模型规划器。模型不支持 tool_calls 时会返回 ok=False，由主循环降级。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def plan(
        self,
        obs: Observation,
        fused: FusedView,
        self_state: dict[str, Any],
        verification: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> Plan:
        started = time.time()
        user_prompt = self._build_prompt(obs, fused, self_state, verification, history)

        payload = {
            "model": self.cfg.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        try:
            response = httpx.post(
                f"{self.cfg.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                json=payload,
                timeout=self.cfg.timeout,
                trust_env=False,
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
        except Exception as exc:
            return Plan(False, "", "", source="llm",
                        error=f"{type(exc).__name__}: {exc}"[:180],
                        latency=time.time() - started)

        calls = message.get("tool_calls") or []
        thought = (message.get("content") or "").strip()
        if not calls:
            return Plan(False, thought, "", raw=thought[:400], source="llm",
                        error="模型没有返回工具调用", latency=time.time() - started)

        call = calls[0].get("function", {})
        name = str(call.get("name", "")).strip()
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        if name not in ACTION_DOC:
            return Plan(False, thought, name, args, raw=str(call)[:400], source="llm",
                        error=f"未知工具 {name!r}", latency=time.time() - started)

        reason = str(args.get("reason") or "").strip()
        plan = Plan(True, reason or thought or f"调用 {name}", name, args, source="llm",
                    raw=json.dumps(call, ensure_ascii=False)[:400],
                    latency=time.time() - started)
        return self._guard(plan, fused, history)

    @staticmethod
    def _guard(plan: Plan, fused: FusedView, history: list[dict[str, Any]]) -> Plan:
        """动作校验层。

        这几条判据在代码里就能算死，没必要让模型猜，也没必要等它执行失败再救：
        1. 已经站到目标旁边 → 该抓就抓，抓取只跟距离有关，跟朝向无关。
        2. 前方就是墙 → 前进必然失败，先把这一步改成转向（绕路），不要浪费一轮；
           此时也不再纠正转向方向：目标是绕过去，不是对准，硬掰只会来回抖。
        3. 目标在前半球（含斜前方）且前方无墙 → 接近策略就是往前，不是原地转。
           斜前方 45 度用 90 度转向永远对不准，转过去就跑到另一侧，会来回震荡。
        4. 已经进入两格内（near）且目标不在背后 → 同样往前挪。九宫格在这个尺度上
           已经没有分辨力，再让模型依据方位去转向只会原地踏步，不如先缩短距离。

        注意这里刻意「只拦不给方向」：能判死的只有前进/抓取这两个语义，
        转向方向留给模型，免得代码和模型互相纠正，把机器人卡在原地打转。
        """
        if not plan.ok:
            return plan

        # 抓取只看曼哈顿距离，跟朝向无关：站在旁边就该抓，再怎么转也是白转
        if fused.adjacent and plan.action != "grab_object":
            plan.guard = "已站在目标旁边，纠正为抓取"
            plan.action = "grab_object"
            return plan

        if fused.front_blocked:
            if plan.action == "move_forward":
                plan.guard = "前方贴墙，前进必然失败，改为左转绕行"
                plan.action = "turn_left"
            return plan

        front_hemi = fused.bearing in ("front", "front-left", "front-right", "here")
        # 距离档位只来自本体测距，九宫格给不了这么细；到这一步方向已经不重要了，
        # 目标就在两格内，除非它明确在背后，否则往前挪一格总不会更远。
        not_behind = fused.bearing not in ("back", "back-left", "back-right")
        if front_hemi or (fused.distance_band == "near" and not_behind):
            if plan.action != "move_forward":
                plan.guard = f"目标在{fused.bearing}（{fused.distance_band}）且前方无墙，纠正为前进"
                plan.action = "move_forward"
        return plan

    @staticmethod
    def _build_prompt(
        obs: Observation,
        fused: FusedView,
        self_state: dict[str, Any],
        verification: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> str:
        turns = _consecutive_turns(history)

        view = fused.to_dict()
        view["visible_objects"] = sorted({o["color"] for o in obs.objects})
        # 距离只用于判断何时抓取，放在这里容易诱导模型「因为太远所以先转向」
        view.pop("target_distance", None)
        # 只告知「观察可不可信」，不告诉它真实答案，避免泄题
        check = {
            "consistent": verification["consistent"],
            "hallucinated_colors": verification["hallucinated"],
            "missed_colors": verification["missed"],
        }
        return (
            f"【任务】{TASK}\n\n"
            f"【机器人自身状态（里程计）】\n"
            f"当前朝向：{self_state['agent']['dir']}（N 上 / E 右 / S 下 / W 左）\n"
            f"当前位置区域：{self_state['agent_region']}\n"
            f"已经连续转向：{turns} 次\n\n"
            f"【这一步的视觉观察】\n{json.dumps(view, ensure_ascii=False, indent=2)}\n\n"
            f"【观察可信度校验】\n{json.dumps(check, ensure_ascii=False)}\n"
            f"说明：consistent 为 false 表示这张图的描述与程序侧真值对不上，需要换个视角重新看。\n\n"
            f"【最近动作】\n{_history_text(history)}\n\n"
            f"请决定下一个动作，通过工具调用返回。"
        )


class OfflinePlanner:
    """不调用任何模型的确定性规划器：BFS 找路 + 贴墙转向。

    作用有两个：没配 key 时验证环境闭环；真实模型连续失败时兜底，
    保证演示不会卡死。
    """

    def plan(self, world: Any) -> Plan:
        started = time.time()
        gx, gy, facing = world.agent["gx"], world.agent["gy"], world.agent["dir"]

        if world.adjacent_to_target():
            return Plan(True, f"已经站在目标旁边（距离 {world.manhattan_to_target()}），直接抓取。",
                        "grab_object", source="offline", latency=time.time() - started)

        path = world.shortest_path((gx, gy), world.target_neighbours())
        if not path or len(path) < 2:
            return Plan(False, "找不到通往目标的路径。", "", source="offline",
                        error="no_path", latency=time.time() - started)

        nxt = path[1]
        need = None
        for name, (dx, dy) in {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}.items():
            if (gx + dx, gy + dy) == nxt:
                need = name
                break
        if need is None:
            return Plan(False, "路径相邻格异常。", "", source="offline",
                        error="bad_path", latency=time.time() - started)

        if need == facing:
            return Plan(True, f"目标下方路径下一格在正前（{need}），前进一格。",
                        "move_forward", source="offline", latency=time.time() - started)

        order = ["N", "E", "S", "W"]
        diff = (order.index(need) - order.index(facing)) % 4
        action = "turn_right" if diff <= 2 else "turn_left"
        steps = diff if diff <= 2 else 4 - diff
        return Plan(True, f"需要转向 {need}，当前朝 {facing}，{action}（差 {steps} 个 90 度）。",
                    action, source="offline", latency=time.time() - started)
