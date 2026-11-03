"""探测：能不能让视觉模型给出更细的画面网格（判断感知分辨率的上限）。

    python scripts/probe_grid.py

九宫格在 9x7 的房间里一格就是 3x3 格，太粗了，闭环拿不到有效梯度。
这里同一帧问两套网格，和真值对账：
  A. 3x3（线上用的）
  B. 5x4（把横向五等分、纵向四等分）

比较两套网格换算出的方位准确率，决定是提升感知分辨率，还是缩小世界尺度。
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_vlm_agent.config import load_config
from sim_vlm_agent.perception import REGIONS, _bearing_close, _extract_json
from sim_vlm_agent.world import DIR_VEC, GRID_W, GRID_H, REGION_LABELS, classify_bearing

PROMPT = """你是一台机器人的视觉模块。画面是 2D 俯视网格房间：深灰色方块是墙，浅色格子是空地，
顶部的深色横条是状态栏，不属于房间。

画面里有一个橙色的圆形图标（带深橙色箭头），那是机器人自己；还有一个红色方块，是目标。

只输出一个 JSON 对象，不要解释，不要 markdown：
{
  "red_region": "<红色方块所在的九宫格区域>",
  "red_grid": {"col": 1, "row": 1}
}

字段说明：
- red_region：忽略顶部状态栏，把房间画面横向三等分、纵向三等分，红色方块中心落在哪一格。
  可选值只能是：%(regions)s
- red_grid：同样忽略顶部状态栏，把房间画面横向**五等分**、纵向**四等分**，
  红色方块中心落在第几列（1-5）、第几行（1-4），从上往下、从左往右数。

硬性约束：橙色是机器人自己，灰色是墙，都不算物体。不要输出小数。""" % {
    "regions": " / ".join(REGIONS)
}

POSES = [
    (1, 5, "E"), (3, 5, "E"), (6, 5, "E"), (7, 5, "N"),
    (7, 4, "N"), (6, 4, "W"), (5, 4, "N"), (2, 3, "N"),
    (2, 2, "E"), (5, 2, "N"), (1, 1, "S"), (4, 1, "W"),
]

COL_BUCKETS, ROW_BUCKETS = 5, 4


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
    content = (response.json()["choices"][0]["message"]["content"] or "").strip()
    return _extract_json(content)


def bearing_from_vec(dx: float, dy: float, facing: str) -> str:
    if abs(dx) < 0.2 and abs(dy) < 0.2:
        return "here"
    fx, fy = DIR_VEC[facing]
    rx, ry = -fy, fx
    return classify_bearing(dx * fx + dy * fy, dx * rx + dy * ry)


def bearing_from_fine(data: dict, agent, facing: str) -> str:
    grid = data.get("red_grid") or {}
    try:
        col = int(grid.get("col"))
        row = int(grid.get("row"))
    except (TypeError, ValueError):
        return "unknown"
    if not (1 <= col <= COL_BUCKETS and 1 <= row <= ROW_BUCKETS):
        return "unknown"
    # 网格中心 -> 世界坐标（以格为单位），再减去机器人所在的格中心
    tx = (col - 0.5) / COL_BUCKETS * GRID_W
    ty = (row - 0.5) / ROW_BUCKETS * GRID_H
    return bearing_from_vec(tx - (agent["gx"] + 0.5), ty - (agent["gy"] + 0.5), facing)


def bearing_from_coarse(region: str, agent_region: str, facing: str) -> str:
    rows = {"upper": 0, "middle": 1, "lower": 2}
    cols = {"left": 0, "center": 1, "right": 2}
    if region not in REGION_LABELS or agent_region not in REGION_LABELS:
        return "unknown"

    def rc(name: str) -> tuple[int, int]:
        if name == "center":
            return 1, 1
        row, col = name.split("-")
        return rows[row], cols[col]

    ar, ac = rc(agent_region)
    tr, tc = rc(region)
    dr, dc = tr - ar, tc - ac
    if dr == 0 and dc == 0:
        return "here"
    return bearing_from_vec(dc, dr, facing)


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
        try:
            data = ask(cfg, world.get_screenshot())
        except Exception as exc:
            print(f"  ({gx},{gy},{facing}) 调用失败 {type(exc).__name__}: {exc}")
            continue

        region = str(data.get("red_region", "unknown")).strip().lower()
        if region not in REGION_LABELS:
            region = "unknown"
        b_coarse = bearing_from_coarse(region, truth["agent_region"], facing)
        b_fine = bearing_from_fine(data, truth["agent"], facing)
        true_bearing = truth["target"]["bearing"]
        rows.append((region, b_coarse, b_fine, true_bearing))
        print(f"  ({gx},{gy}) {facing} 区域 {region:<13} 九宫格 {b_coarse:<11}"
              f" 5x4 {b_fine:<11} 真值 {true_bearing}")

    if not rows:
        print("一帧都没问成，检查网络与密钥。")
        world.close()
        return 1

    n = len(rows)
    c_exact = sum(1 for r in rows if r[1] == r[3])
    c_close = sum(1 for r in rows if _bearing_close(r[1], r[3]))
    f_exact = sum(1 for r in rows if r[2] == r[3])
    f_close = sum(1 for r in rows if _bearing_close(r[2], r[3]))
    print("-" * 78)
    print(f"帧数 {n}")
    print(f"3x3 九宫格：完全命中 {c_exact}/{n}（{100*c_exact/n:.0f}%），"
          f"容差一档 {c_close}/{n}（{100*c_close/n:.0f}%）")
    print(f"5x4 细网格：完全命中 {f_exact}/{n}（{100*f_exact/n:.0f}%），"
          f"容差一档 {f_close}/{n}（{100*f_close/n:.0f}%）")
    world.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
