"""模块 2：VLM 视觉感知。

分工原则（踩过坑之后定下来的）：

- VLM 只回答「画面里有什么、红色方块落在画面哪个区域」。这是纯视觉判断，
  不需要任何视角变换，小模型也能答对。
- 机器人的朝向、正前方是否贴墙、目标距离档位，属于本体感知/局部地图，
  由环境侧提供，不让视觉模型去猜。
- 相对方位（目标在机器人的左前还是右后）由程序用「视觉的绝对区域 +
  本体朝向」做几何换算。

上一版让 VLM 直接输出「目标相对我朝哪个方向」，实测 glm-4v-flash 在 8 帧里
错 6 帧：它读不准橙色箭头指向，也做不好以自身为原点的坐标变换。把不该它
做的活拿走之后，观察才稳定下来。
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Config
from .world import REGION_LABELS, DIR_VEC, classify_bearing, region_to_rc

REGIONS = REGION_LABELS

# 画面九宫格 -> 绝对方向向量（x 向右为正，y 向下为正）
REGION_VEC = {
    "upper-left": (-1, -1), "upper-center": (0, -1), "upper-right": (1, -1),
    "middle-left": (-1, 0), "center": (0, 0), "middle-right": (1, 0),
    "lower-left": (-1, 1), "lower-center": (0, 1), "lower-right": (1, 1),
}

BEARING_RING = ("front", "front-right", "right", "back-right",
                "back", "back-left", "left", "front-left")

SYSTEM_PROMPT = """你是一台机器人的视觉模块。这一帧你只需要做一件事：在画面里找到红色方块。

画面说明：
- 这是一个 2D 俯视网格房间，深灰色方块是墙，浅色格子是空地。
- 画面里有一个橙色的圆形图标，那是机器人自己，它不是物体，绝不能当成目标。
- 除了红色方块，房间里可能还有蓝色方块、绿色圆形，它们只是干扰物。

只输出一个 JSON 对象，不要解释文字，不要 markdown 代码块：
{
  "red_present": true,
  "red_region": "<红色方块所在的九宫格区域>",
  "other_colors": ["画面上还能看到的其它颜色"]
}

字段说明：
- red_present：画面上有没有红色方块。有就 true，没有就 false。
- red_region：把画面横向三等分、纵向三等分，看红色方块中心落在哪一格。
  可选值只能是：%(regions)s
  看不见红色方块时填 unknown。
- other_colors：除红色以外，画面上还看得到的物体颜色。只能填 blue 或 green，
  一个都没有就填空数组 []。

硬性约束：
1. 只描述画面上真实存在的红色方块，绝对不允许编造。
2. 橙色是机器人自己，黄色只是描边，灰色是墙——这三种颜色都不算物体。
3. 不要输出任何坐标数字。""" % {"regions": " / ".join(REGIONS)}


def _bearing_close(seen: str, true: str, tolerance: int = 1) -> bool:
    """方位差异容忍一档：贴着 45 度分界的判断本来就会左右摇摆。"""
    if seen == "here" or true == "here":
        return seen == true
    if seen not in BEARING_RING or true not in BEARING_RING:
        return False
    diff = abs(BEARING_RING.index(seen) - BEARING_RING.index(true))
    return min(diff, len(BEARING_RING) - diff) <= tolerance


@dataclass
class Observation:
    """纯视觉结果：只有「看到了什么」和「目标在画面哪一格」。"""

    ok: bool
    source: str  # vlm | fallback
    objects: list[dict[str, Any]] = field(default_factory=list)
    target_visible: bool = False
    target_region: str = "unknown"
    raw: str = ""
    error: str = ""
    latency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source": self.source,
            "objects": self.objects,
            "target_visible": self.target_visible,
            "target_region": self.target_region,
            "raw": self.raw,
            "error": self.error,
            "latency": round(self.latency, 2),
        }

    def brief(self) -> str:
        colors = ", ".join(sorted({o.get("color", "?") for o in self.objects})) or "无"
        return (f"目标区={self.target_region} 可见={self.target_visible} 物体={colors}")


@dataclass
class FusedView:
    """视觉 + 本体感知融合后，规划层真正需要的那几个量。"""

    bearing: str
    distance_band: str
    front_blocked: bool
    adjacent: bool
    target_region: str
    target_visible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_bearing": self.bearing,
            "target_distance": self.distance_band,
            "front_blocked": self.front_blocked,
            "already_beside_target": self.adjacent,
            "target_region": self.target_region,
            "target_visible": self.target_visible,
        }

    def brief(self) -> str:
        return (f"目标={self.bearing}/{self.distance_band} 前方堵={self.front_blocked}"
                f" 画面区={self.target_region}")


def _relative_vector(agent_region: str, target_region: str) -> tuple[int, int] | None:
    """从「机器人所在区域」指向「目标所在区域」的粗略向量。

    不能用「目标区域相对画面中心」的向量：机器人往往不在画面正中，
    那样算出来的方向会整体偏一点。以机器人所在区域为原点才和「相对机器人」对得上。
    """
    here = region_to_rc(agent_region)
    there = region_to_rc(target_region)
    if here is None or there is None:
        return None
    dr, dc = there[0] - here[0], there[1] - here[1]
    if dr == 0 and dc == 0:
        return None  # 同一格：九宫格已经没有分辨力了
    return dc, dr  # (x 向右, y 向下)


def fuse(obs: Observation, truth: dict[str, Any]) -> FusedView:
    """用视觉给的区域 + 本体朝向，算出目标相对机器人的方位。

    九宫格是粗粒度的：它只能给出「大概在右前方」，给不出「还差两格」。
    同一区域时连方向都给不出（bearing=here），这时的收尾交给 planner._guard。
    """
    facing = truth["agent"]["dir"]
    bearing = "unknown"
    if obs.target_visible and obs.target_region in REGION_VEC:
        vector = _relative_vector(truth.get("agent_region", ""), obs.target_region)
        if vector is None:
            # 落在同一区域：方向未知，但可以确定离得很近
            bearing = "here"
        else:
            fx, fy = DIR_VEC[facing]
            rx, ry = -fy, fx
            dx, dy = vector
            bearing = classify_bearing(dx * fx + dy * fy, dx * rx + dy * ry)
    return FusedView(
        bearing=bearing,
        distance_band=truth["target"]["distance_band"],
        front_blocked=truth["front_blocked"],
        adjacent=truth["target"]["adjacent"],
        target_region=obs.target_region,
        target_visible=obs.target_visible,
    )


def _extract_json(text: str) -> dict[str, Any]:
    """模型偶尔会包一层 markdown 或加解释，这里做容错提取。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:idx + 1])
    raise ValueError("响应里找不到 JSON 对象")


class Perception:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def perceive(self, image_bytes: bytes) -> Observation:
        if not self.cfg.available:
            return self._fallback("离线模式未启用视觉模型")

        started = time.time()
        payload = {
            "model": self.cfg.vlm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                       + base64.b64encode(image_bytes).decode()
                            },
                        },
                        {"type": "text", "text": "描述这一帧画面，按要求输出 JSON。"},
                    ],
                },
            ],
            "temperature": 0.0,
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
            body = response.json()
            content = (body["choices"][0]["message"].get("content") or "").strip()
            data = _extract_json(content)
        except Exception as exc:  # 网络、HTTP、解析都走同一条降级路径
            return self._fallback(f"{type(exc).__name__}: {exc}"[:180], time.time() - started)

        region = str(data.get("red_region", "unknown")).strip().lower()
        red_present = bool(data.get("red_present"))
        objects: list[dict[str, Any]] = []
        if red_present:
            objects.append({"color": "red", "shape": "square"})
        others = data.get("other_colors")
        if isinstance(others, list):
            for item in others:
                color = str(item).strip().lower()
                if color in ("blue", "green"):  # 词表外的颜色一律丢弃，避免把机器人/墙当物体
                    objects.append({"color": color, "shape": "unknown"})

        obs = Observation(
            ok=True,
            source="vlm",
            objects=objects,
            target_visible=red_present and region in REGIONS,
            target_region=region if region in REGIONS else "unknown",
            raw=content[:1200],
            latency=time.time() - started,
        )
        return obs

    def _fallback(self, reason: str, elapsed: float = 0.0) -> Observation:
        return Observation(ok=False, source="fallback", error=reason, latency=elapsed)

    # ---------- 与真值对账 ----------
    @staticmethod
    def verify(obs: Observation, fused: FusedView, truth: dict[str, Any]) -> dict[str, Any]:
        """多报=幻觉，少报=漏检，方位偏差 >1 档=不可信。"""
        real = set(truth["colors_present"])
        seen = {o["color"] for o in obs.objects}
        hallucinated = sorted(seen - real)
        missed = sorted(real - seen)
        if fused.bearing == "here":
            # 目标与机器人同处一个九宫格：这是表示精度的上限，不是判断错误
            bearing_ok = truth["target"]["distance"] <= 2
        else:
            bearing_ok = _bearing_close(fused.bearing, truth["target"]["bearing"])
        # 任务的成败只取决于有没有找到红块、方位对不对；
        # 漏报干扰物记一笔就够了，不该因此判整帧观察不可信。
        red_found = "red" in seen
        consistent = red_found and bearing_ok and not hallucinated
        return {
            "consistent": consistent,
            "hallucinated": hallucinated,
            "missed": missed,
            "bearing_ok": bearing_ok,
            "seen_bearing": fused.bearing,
            "true_bearing": truth["target"]["bearing"],
            "true_region": truth["target"]["region"],
        }
