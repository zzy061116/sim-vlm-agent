"""探测：视觉模型能不能回答「红块在机器人的左上/右下」这类相对判断。

    python scripts/probe_relative.py

同一帧里同时问两种问法，逐帧和真值对账，看哪一种问法更靠得住：
  A. 九宫格区域（红块在画面哪一格）
  B. 相对机器人本体的比较（红块在机器人上面还是下面、左边还是右边）

区域法再分两种换算口径：以画面中心为原点（绝对）、以机器人所在区域为原点（相对）。
输出三种口径的方位准确率，用来决定感知层该问哪一个、该怎么换算。
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_vlm_agent.config import load_config
from sim_vlm_agent.perception import REGION_VEC, REGIONS, _bearing_close, _extract_json
from sim_vlm_agent.world import DIR_VEC, REGION_LABELS, classify_bearing, region_to_rc

PROMPT = """你是一台机器人的视觉模块。画面是 2D 俯视网格房间：深灰色方块是墙，浅色格子是空地。

画面里有两个关键色块：
- 橙色圆形图标，带一个深橙色箭头，这是机器人自己。
- 一个红色方块，这是要找的目标。

只输出一个 JSON 对象，不要解释，不要 markdown：
{
  "red_present": true,
  "red_region": "<红色方块所在的九宫格区域>",
  "red_vs_robot": {"vertical": "above/below/same", "horizontal": "left/right/same"},
  "other_colors": ["除红色以外还能看到的物体颜色"]
}

字段说明：
- red_region：把画面横向三等分、纵向三等分，红色方块中心落在哪一格。可选值只能是：%(regions)s
- red_vs_robot：只比较两个色块中心点的位置。红色方块中心比机器人中心更靠画面上方填 above，
  更靠下方填 below，差不多高填 same；同理横向填 left / right / same。
- other_colors：只能填 blue 或 green，没有就填空数组。

硬性约束：橙色是机器人自己，黄色只是描边，灰色是墙，这三种都不算物体。不要输出坐标数字。""" % {
    "regions": " / ".join(REGIONS)
}

# 覆盖真实路线会经过的位置与朝向
POSES = [
    (1, 5, "E"), (3, 5, "E"), (6, 5, "E"), (7, 5, "N"),
    (7, 4, "N"), (7, 4, "E"), (7, 3, "N"), (6, 4, "W"),
    (7, 4, "S"), (5, 4, "N"), (2, 3, "N"), (2, 2, "E"),
]


def ask(cfg, image: bytes) -> dict:
    payload = {
        "model": cfg.vlm_model,
        "messages": [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64,"
                                          + base64.b64encode(image).decode()}},
                    {"type": "text", "text": "按要求输出 JSON。"},
                ],
            },
        ],
        "temperature": 0.0,
    }
    response = httpx.post(
        f"{cfg.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        json=payload,
        timeout=cfg.timeout,
        trust_env=False,
    )
    response.raise_for_status()
    content = (response.json()["choices"][0]["message"].get("content") or "").strip()
    return _extract_json(content)


def bearing_from_compare(compare: dict, facing: str) -> str:
    """把「相对机器人上下左右」的比较结果换算成八方位。"""
    vertical = str(compare.get("vertical", "same")).strip().lower()
    horizontal = str(compare.get("horizontal", "same")).strip().lower()
    dx = {"left": -1, "right": 1}.get(horizontal, 0)
    dy = {"above": -1, "below": 1}.get(vertical, 0)
    if dx == 0 and dy == 0:
        return "here"
    fx, fy = DIR_VEC[facing]
    rx, ry = -fy, fx
    return classify_bearing(dx * fx + dy * fy, dx * rx + dy * ry)


def bearing_from_region(region: str, agent_region: str, facing: str) -> str:
    """区域法一：直接用目标区域相对画面中心的向量（线上的做法）。"""
    vector = REGION_VEC.get(region)
    return _to_bearing(vector, facing)


def bearing_from_region_relative(region: str, agent_region: str, facing: str) -> str:
    """区域法二：以机器人所在区域为原点，算目标区域的相对向量。"""
    here, there = region_to_rc(agent_region), region_to_rc(region)
    if here is None or there is None:
        return "unknown"
    dr, dc = there[0] - here[0], there[1] - here[1]
    if dr == 0 and dc == 0:
        return "here"
    return _to_bearing((dc, dr), facing)


def _to_bearing(vector: tuple[int, int] | None, facing: str) -> str:
    if vector is None:
        return "unknown"
    fx, fy = DIR_VEC[facing]
    rx, ry = -fy, fx
    dx, dy = vector
    return classify_bearing(dx * fx + dy * fy, dx * rx + dy * ry)


def main() -> int:
    cfg = load_config()
    if not cfg.available:
        print("没有可用密钥，先把 LLM_API_KEY 写进 .env。")
        return 1

    from sim_vlm_agent.world import World

    world = World(headless=True, classic=True)  # 探测脚本固定用经典布局，保证与历史实验可比
    rows = []
    for gx, gy, facing in POSES:
        world.agent.update(gx=gx, gy=gy, dir=facing)
        truth = world.get_state()
        image = world.get_screenshot()

        try:
            data = ask(cfg, image)
        except Exception as exc:  # 探测脚本，单帧失败就跳过
            print(f"  ({gx},{gy},{facing}) 调用失败 {type(exc).__name__}: {exc}")
            continue

        region = str(data.get("red_region", "unknown")).strip().lower()
        if region not in REGION_LABELS:
            region = "unknown"
        compare = data.get("red_vs_robot") or {}
        b_region = bearing_from_region(region, truth["agent_region"], facing)
        b_relative = bearing_from_region_relative(region, truth["agent_region"], facing)
        b_compare = bearing_from_compare(compare, facing)
        true_bearing = truth["target"]["bearing"]
        rows.append((gx, gy, facing, region, b_region, b_relative, b_compare, true_bearing))
        print(f"  ({gx},{gy}) {facing} 区域 {region:<13} 区域法 {b_region:<11}"
              f" 相对法 {b_relative:<11} 比较法 {b_compare:<11} 真值 {true_bearing}")

    print("-" * 78)
    if not rows:
        print("一帧都没问成，检查网络与密钥。")
        world.close()
        return 1

    n = len(rows)
    exact_region = sum(1 for r in rows if r[4] == r[7])
    close_region = sum(1 for r in rows if _bearing_close(r[4], r[7]))
    exact_relative = sum(1 for r in rows if r[5] == r[7])
    close_relative = sum(1 for r in rows if _bearing_close(r[5], r[7]))
    exact_cmp = sum(1 for r in rows if r[6] == r[7])
    close_cmp = sum(1 for r in rows if _bearing_close(r[6], r[7]))
    exact_zone = sum(1 for r in rows if r[3] == "middle-right")  # 真值区域恒为 middle-right
    print(f"帧数 {n}")
    print(f"区域判断本身：{exact_zone}/{n} 与真值区域一致")
    print(f"九宫格绝对区域法：完全命中 {exact_region}/{n}（{100*exact_region/n:.0f}%），"
          f"容差一档 {close_region}/{n}（{100*close_region/n:.0f}%）")
    print(f"九宫格相对区域法：完全命中 {exact_relative}/{n}（{100*exact_relative/n:.0f}%），"
          f"容差一档 {close_relative}/{n}（{100*close_relative/n:.0f}%）")
    print(f"相对比较法　　　：完全命中 {exact_cmp}/{n}（{100*exact_cmp/n:.0f}%），"
          f"容差一档 {close_cmp}/{n}（{100*close_cmp/n:.0f}%）")
    print("-" * 78)
    print(json.dumps({"frames": n, "zone_exact": exact_zone,
                      "abs_exact": exact_region, "abs_close": close_region,
                      "rel_exact": exact_relative, "rel_close": close_relative,
                      "cmp_exact": exact_cmp, "cmp_close": close_cmp},
                     ensure_ascii=False))
    world.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
