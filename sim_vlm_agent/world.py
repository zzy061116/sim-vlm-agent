"""模块 1：Pygame 2D 仿真环境。

职责边界：只负责「世界是什么样」和「动作怎么改变世界」，不认识任何模型。
对外三个接口：
    get_screenshot()  当前画面 PNG 字节
    exec_action(name) 执行动作，返回执行结果
    get_state()       程序侧真实状态（ground truth，用于校验与降级）
"""

from __future__ import annotations

import io
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pygame
from PIL import Image

# ---------- 世界尺寸 ----------
# 6x6 不是随手定的：感知层只可靠到「九宫格」，一格里就是 2x2 个格子，
# 区域之间的相对向量才有 1~2 格的分辨力。房间再大，目标就会长期粘在同一个
# 区域里，闭环等于没有反馈（9x7 实测就是这样卡住的）。
GRID_W, GRID_H = 6, 6
CELL = 64
HEADER = 48
VIEW_W, VIEW_H = GRID_W * CELL, GRID_H * CELL
WIN_W, WIN_H = VIEW_W, VIEW_H + HEADER

# 朝向：0=N 1=E 2=S 3=W（屏幕坐标 y 向下）
DIR_VEC = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
DIR_ORDER = ["N", "E", "S", "W"]

ACTIONS = ("move_forward", "turn_left", "turn_right", "grab_object")

# 九宫格方位词表，和 VLM prompt 里的取值完全一致
REGION_LABELS = (
    "upper-left", "upper-center", "upper-right",
    "middle-left", "center", "middle-right",
    "lower-left", "lower-center", "lower-right",
)


BEARING_SECTORS = (
    (-180, -157.5, "back"), (-157.5, -112.5, "back-left"),
    (-112.5, -67.5, "left"), (-67.5, -22.5, "front-left"),
    (-22.5, 22.5, "front"), (22.5, 67.5, "front-right"),
    (67.5, 112.5, "right"), (112.5, 157.5, "back-right"),
    (157.5, 180, "back"),
)


def classify_bearing(forward: float, right: float) -> str:
    """把「前方分量 / 右方分量」分类成八方位。

    环境算真值时用它，程序把视觉的绝对方位换算成相对方位时也用它，
    保证两边口径完全一致。
    """
    if forward == 0 and right == 0:
        return "here"
    angle = math.degrees(math.atan2(right, forward))
    for low, high, name in BEARING_SECTORS:
        if low <= angle < high:
            return name
    return "back"


def region_to_rc(region: str) -> tuple[int, int] | None:
    """九宫格方位 -> (行号, 列号)，用于给观测校验设置容差。"""
    rows = {"upper": 0, "middle": 1, "lower": 2}
    cols = {"left": 0, "center": 1, "right": 2}
    if region == "center":
        return 1, 1
    if region not in REGION_LABELS:
        return None
    row, col = region.split("-")
    return rows[row], cols[col]

# ---------- 配色 ----------
C_BG = (247, 248, 250)
C_GRID = (226, 229, 235)
C_WALL = (90, 96, 106)
C_WALL_EDGE = (66, 71, 80)
C_AGENT = (240, 140, 30)
C_AGENT_ARROW = (196, 92, 8)
C_AGENT_EDGE = (255, 255, 255)
C_OBJECTS = {"red": (220, 53, 47), "blue": (52, 120, 220), "green": (46, 160, 90)}
C_HEADER = (31, 41, 55)
C_TEXT = (255, 255, 255)


@dataclass
class SceneObject:
    name: str
    color: str
    shape: str  # square | circle
    gx: int
    gy: int

    def to_dict(self) -> dict[str, Any]:
        return {"color": self.color, "shape": self.shape, "name": self.name}


@dataclass
class StepResult:
    action: str
    ok: bool
    reason: str
    before: dict[str, Any]
    after: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "ok": self.ok,
            "reason": self.reason,
            "before": self.before,
            "after": self.after,
        }


class World:
    """一个 6x6 的网格房间：外墙固定，内墙、出生点与朝向每次随机生成。

    seed=None（默认）每次随机；seed=<int> 复现布局；classic=True 用最早的
    固定布局（L 形隔墙 + 出生 (1,4,E)），供 probe_* 探测脚本复现实验条件。
    硬约束：出生点 BFS 可达目标、与目标曼哈顿距离 >= 3；物体位置固定。
    """

    CLASSIC_INNER_WALLS = ((2, 2), (2, 3), (3, 4))
    CLASSIC_START = (1, 4, "E")

    def __init__(self, headless: bool = True, seed: int | None = None,
                 classic: bool = False) -> None:
        self.headless = headless
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.font.init()

        self.screen: pygame.Surface | None = None
        if not headless:
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))
            pygame.display.set_caption("Sim-VLM-Agent")

        self.surface = pygame.Surface((WIN_W, WIN_H))
        self.font = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        self.readable = False
        self.quit_requested = False
        self.step_index = 0

        # 物体固定：任务始终是「找到红色方块」，随机的是出发条件和路障
        self.objects: list[SceneObject] = [
            SceneObject("目标红色方块", "red", "square", 4, 1),
            SceneObject("蓝色方块", "blue", "square", 4, 4),
            SceneObject("绿色圆块", "green", "circle", 3, 3),
        ]
        self.target = self.objects[0]

        # 外墙：房间边界
        outer: set[tuple[int, int]] = set()
        for x in range(GRID_W):
            outer.update({(x, 0), (x, GRID_H - 1)})
        for y in range(GRID_H):
            outer.update({(0, y), (GRID_W - 1, y)})

        if classic:
            self.seed: int | None = None
            self.walls = frozenset(outer | set(self.CLASSIC_INNER_WALLS))
            self.start = self.CLASSIC_START
        else:
            # None 时也生成具体种子记下来：summary 可查、可复现
            actual_seed = seed if seed is not None else random.randrange(10 ** 9)
            self.seed = actual_seed
            self.walls, self.start = self._random_layout(random.Random(actual_seed), outer)

        self.reset()

    # ---------- 生命周期 ----------
    def reset(self) -> None:
        gx, gy, d = self.start
        self.agent = {"gx": gx, "gy": gy, "dir": d}
        self.done = False
        self.grabbed = False
        self.step_index = 0
        self.readable = False

    # ---------- 随机布局 ----------
    def _random_layout(self, rng: random.Random,
                       outer: set[tuple[int, int]]) -> tuple[frozenset, tuple[int, int, str]]:
        """随机出生点/朝向/内墙，并保证任务可解（出生点 BFS 可达目标）。

        内墙 2~3 堵：6x6 内部只有 16 格，再多会把路堵死，24 步上限兜不住。
        """
        interior = [(x, y) for x in range(1, GRID_W - 1) for y in range(1, GRID_H - 1)]
        obj_cells = {(o.gx, o.gy) for o in self.objects}

        for _ in range(500):
            sx, sy = rng.choice(interior)
            if (sx, sy) in obj_cells:
                continue
            if abs(sx - self.target.gx) + abs(sy - self.target.gy) < 3:
                continue  # 出生就贴脸没有演示意义
            facing = rng.choice(DIR_ORDER)
            candidates = [c for c in interior if c not in obj_cells and c != (sx, sy)]
            inner = set(rng.sample(candidates, rng.randint(2, 3)))
            walls = frozenset(outer | inner)
            if self._layout_solvable(walls, (sx, sy)):
                return walls, (sx, sy, facing)

        # 500 次采样全部无解的概率近似为零；真到这一步回退经典布局
        return frozenset(outer | set(self.CLASSIC_INNER_WALLS)), self.CLASSIC_START

    def _layout_solvable(self, walls: frozenset, start_cell: tuple[int, int]) -> bool:
        """BFS：出生点能否走到目标的任一相邻格。"""
        goals = {(self.target.gx + dx, self.target.gy + dy) for dx, dy in DIR_VEC.values()}
        queue = deque([start_cell])
        seen = {start_cell}
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) in goals:
                return True
            for dx, dy in DIR_VEC.values():
                nxt = (cx + dx, cy + dy)
                nx, ny = nxt
                if nxt in seen or not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                    continue
                if nxt in walls:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return False

    def layout(self) -> dict[str, Any]:
        """布局快照：写进 summary，配合 seed 可完整复现本次地图。"""
        inner = sorted(
            [w for w in self.walls
             if 0 < w[0] < GRID_W - 1 and 0 < w[1] < GRID_H - 1]
        )
        return {
            "grid": [GRID_W, GRID_H],
            "seed": self.seed,
            "start": list(self.start),
            "inner_walls": [list(w) for w in inner],
            "objects": [{"name": o.name, "color": o.color, "shape": o.shape,
                         "pos": [o.gx, o.gy]} for o in self.objects],
        }

    def close(self) -> None:
        pygame.quit()

    def poll_quit(self) -> bool:
        """窗口模式下让用户能关窗口；headless 永远返回 False。"""
        if self.screen is None:
            return False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit_requested = True
        return self.quit_requested

    # ---------- 几何 ----------
    def is_wall(self, gx: int, gy: int) -> bool:
        return (gx, gy) in self.walls

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < GRID_W and 0 <= gy < GRID_H

    def _front_cell(self) -> tuple[int, int]:
        dx, dy = DIR_VEC[self.agent["dir"]]
        return self.agent["gx"] + dx, self.agent["gy"] + dy

    def manhattan_to_target(self) -> int:
        return abs(self.agent["gx"] - self.target.gx) + abs(self.agent["gy"] - self.target.gy)

    def adjacent_to_target(self) -> bool:
        return self.manhattan_to_target() <= 1

    def front_blocked(self) -> bool:
        """正前方紧邻格是不是墙——本体近距离传感/局部地图就能拿到的量。"""
        nx, ny = self._front_cell()
        return (not self.in_bounds(nx, ny)) or self.is_wall(nx, ny)

    def _side_cell(self, side: int) -> tuple[int, int]:
        """side=+1 右侧，side=-1 左侧：朝向的垂直方向偏转一格。"""
        dx, dy = DIR_VEC[self.agent["dir"]]
        sx, sy = -dy * side, dx * side
        return self.agent["gx"] + sx, self.agent["gy"] + sy

    def left_blocked(self) -> bool:
        """左侧紧邻格是不是墙——对应真机的左向测距/红外。"""
        nx, ny = self._side_cell(-1)
        return (not self.in_bounds(nx, ny)) or self.is_wall(nx, ny)

    def right_blocked(self) -> bool:
        """右侧紧邻格是不是墙——对应真机的右向测距/红外。"""
        nx, ny = self._side_cell(1)
        return (not self.in_bounds(nx, ny)) or self.is_wall(nx, ny)

    def distance_band(self) -> str:
        """粗粒度距离档位，视觉上本来也只能估到这个精度。"""
        distance = self.manhattan_to_target()
        return "near" if distance <= 2 else ("mid" if distance <= 4 else "far")

    def region_of(self, gx: int, gy: int) -> str:
        """把格子映射到九宫格方位标签。

        必须用格子「中心」的像素位置来三分，而不是格子索引：
        视觉模型看的是画面像素，两套口径不一致会把边界上的物体判错。
        """
        cx, cy = gx + 0.5, gy + 0.5
        col = "left" if cx < GRID_W / 3 else ("right" if cx >= 2 * GRID_W / 3 else "center")
        row = "upper" if cy < GRID_H / 3 else ("lower" if cy >= 2 * GRID_H / 3 else "middle")
        if col == "center" and row == "middle":
            return "center"
        if row == "middle":
            return f"middle-{col}"
        if col == "center":
            return f"{row}-center"
        return f"{row}-{col}"

    def bearing_of(self, gx: int, gy: int) -> str:
        """目标相对 Agent 朝向的八方位（front / front-right / right / …）。"""
        dx = gx - self.agent["gx"]
        dy = gy - self.agent["gy"]
        fx, fy = DIR_VEC[self.agent["dir"]]
        rx, ry = -fy, fx
        return classify_bearing(dx * fx + dy * fy, dx * rx + dy * ry)

    # ---------- 动作 ----------
    def exec_action(self, action: str) -> StepResult:
        before = dict(self.agent)
        action = (action or "").strip()

        if self.done:
            return StepResult(action, False, "episode_already_finished", before, dict(self.agent))

        if action == "move_forward":
            nx, ny = self._front_cell()
            if not self.in_bounds(nx, ny) or self.is_wall(nx, ny):
                return StepResult(action, False, "blocked_by_wall", before, dict(self.agent))
            self.agent["gx"], self.agent["gy"] = nx, ny
            self.step_index += 1
            return StepResult(action, True, "moved", before, dict(self.agent))

        if action in ("turn_left", "turn_right"):
            idx = DIR_ORDER.index(self.agent["dir"])
            self.agent["dir"] = DIR_ORDER[(idx + (3 if action == "turn_left" else 1)) % 4]
            self.step_index += 1
            return StepResult(action, True, "turned", before, dict(self.agent))

        if action == "grab_object":
            if not self.adjacent_to_target():
                return StepResult(action, False, "target_too_far", before, dict(self.agent))
            self.grabbed = True
            self.done = True
            self.step_index += 1
            return StepResult(action, True, "grabbed_target", before, dict(self.agent))

        return StepResult(action, False, "unknown_action", before, dict(self.agent))

    # ---------- 观测 ----------
    def get_state(self) -> dict[str, Any]:
        """程序侧真实状态。规划层用它做校验与降级，不参与「看图」。"""
        return {
            "agent": dict(self.agent),
            "agent_region": self.region_of(self.agent["gx"], self.agent["gy"]),
            "front_blocked": self.front_blocked(),
            "left_blocked": self.left_blocked(),
            "right_blocked": self.right_blocked(),
            "target": {
                "color": self.target.color,
                "shape": self.target.shape,
                "region": self.region_of(self.target.gx, self.target.gy),
                "bearing": self.bearing_of(self.target.gx, self.target.gy),
                "distance": self.manhattan_to_target(),
                "distance_band": self.distance_band(),
                "adjacent": self.adjacent_to_target(),
            },
            "objects": [
                {**o.to_dict(), "region": self.region_of(o.gx, o.gy),
                 "bearing": self.bearing_of(o.gx, o.gy)}
                for o in self.objects
            ],
            "colors_present": sorted({o.color for o in self.objects}),
            "done": self.done,
            "grabbed": self.grabbed,
            "step_index": self.step_index,
        }

    # ---------- 渲染 ----------
    def _cell_rect(self, gx: int, gy: int) -> pygame.Rect:
        return pygame.Rect(gx * CELL, HEADER + gy * CELL, CELL, CELL)

    def render(self) -> pygame.Surface:
        s = self.surface
        s.fill(C_BG)

        # 网格
        for gx in range(GRID_W + 1):
            x = gx * CELL
            pygame.draw.line(s, C_GRID, (x, HEADER), (x, HEADER + VIEW_H))
        for gy in range(GRID_H + 1):
            y = HEADER + gy * CELL
            pygame.draw.line(s, C_GRID, (0, y), (VIEW_W, y))

        # 墙
        for gx, gy in self.walls:
            rect = self._cell_rect(gx, gy)
            pygame.draw.rect(s, C_WALL, rect)
            pygame.draw.rect(s, C_WALL_EDGE, rect, 2)

        # 物体。目标不加额外装饰——黄色描边会被视觉模型误认成「黄色物体」
        for obj in self.objects:
            rect = self._cell_rect(obj.gx, obj.gy)
            color = C_OBJECTS[obj.color]
            size = 18 if obj.color == "red" else 20
            if obj.shape == "square":
                pygame.draw.rect(s, color, rect.inflate(-size, -size), border_radius=4)
                pygame.draw.rect(s, (255, 255, 255), rect.inflate(-size, -size), 2, border_radius=4)
            else:
                pygame.draw.circle(s, color, rect.center, 15)
                pygame.draw.circle(s, (255, 255, 255), rect.center, 15, 2)

        # Agent：橙色圆 + 圆外的粗箭头，保证视觉模型能看出朝向
        rect = self._cell_rect(self.agent["gx"], self.agent["gy"])
        cx, cy = rect.center
        dx, dy = DIR_VEC[self.agent["dir"]]
        px, py = -dy, dx  # 朝向的垂直方向，用来撑开箭头底边
        tip = (cx + dx * 30, cy + dy * 30)
        left = (cx + dx * 15 + px * 11, cy + dy * 15 + py * 11)
        right = (cx + dx * 15 - px * 11, cy + dy * 15 - py * 11)
        pygame.draw.polygon(s, C_AGENT_ARROW, [tip, left, right])
        pygame.draw.circle(s, C_AGENT, (cx, cy), 15)
        pygame.draw.circle(s, C_AGENT_EDGE, (cx, cy), 15, 2)

        # 顶栏
        pygame.draw.rect(s, C_HEADER, pygame.Rect(0, 0, WIN_W, HEADER))
        label = f"STEP {self.step_index:02d}   DIR {self.agent['dir']}"
        s.blit(self.font.render(label, True, C_TEXT), (14, 13))

        if self.screen is not None:
            self.screen.blit(s, (0, 0))
            pygame.display.flip()
        return s

    def get_screenshot(self, path: str | os.PathLike | None = None) -> bytes:
        """返回当前画面的 PNG 字节；可选同时落盘一份供轨迹报告使用。"""
        surf = self.render()
        raw = pygame.image.tobytes(surf, "RGB")
        image = Image.frombytes("RGB", surf.get_size(), raw)
        if path is not None:
            image.save(str(path), "PNG")
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue()

    # ---------- 给离线规划器用 ----------
    def walkable(self, gx: int, gy: int) -> bool:
        return self.in_bounds(gx, gy) and not self.is_wall(gx, gy)

    def shortest_path(self, start: tuple[int, int], goals: set[tuple[int, int]]) -> list[tuple[int, int]]:
        """BFS 最短路径，返回包含起点与终点的格子序列；无解返回空列表。"""
        from collections import deque

        if start in goals:
            return [start]
        queue = deque([start])
        prev = {start: None}
        while queue:
            cur = queue.popleft()
            cx, cy = cur
            for dx, dy in DIR_VEC.values():
                nxt = (cx + dx, cy + dy)
                if nxt in prev or not self.walkable(*nxt):
                    continue
                prev[nxt] = cur
                if nxt in goals:
                    path = [nxt]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                queue.append(nxt)
        return []

    def target_neighbours(self) -> set[tuple[int, int]]:
        cells = set()
        for dx, dy in DIR_VEC.values():
            nx, ny = self.target.gx + dx, self.target.gy + dy
            if self.walkable(nx, ny):
                cells.add((nx, ny))
        return cells
