"""命令行入口。

    python run.py                 # 无窗口跑一遍完整闭环
    python run.py --render        # 开窗口实时看
    python run.py --offline       # 不调用模型，只验证环境与循环
    python run.py --max-step 30
"""

from __future__ import annotations

import argparse
import sys

from sim_vlm_agent.config import load_config
from sim_vlm_agent.loop import AgentLoop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sim-VLM-Agent 仿真具身智能原型")
    parser.add_argument("--render", action="store_true", help="打开 Pygame 窗口实时观看")
    parser.add_argument("--offline", action="store_true", help="不调用任何模型")
    parser.add_argument("--max-step", type=int, default=None, help="最大循环轮次")
    parser.add_argument("--delay", type=float, default=0.0, help="每步之间停顿秒数")
    parser.add_argument("--label", default="run", help="本次运行的标签，用于轨迹目录名")
    parser.add_argument("--seed", type=int, default=None,
                        help="世界随机种子：复现某次运行的布局；缺省每次随机生成")
    parser.add_argument("--stuck", choices=("state", "old"), default="state",
                        help="打转检测判据：state=状态重复+可执行前进放行（默认）；"
                             "old=连续 3 步位置未变（旧版，供对照）")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.offline:
        cfg = load_config(OFFLINE="1")
    if args.max_step:
        cfg = load_config(MAX_STEP=str(args.max_step))

    print("=" * 62)
    print("Sim-VLM-Agent  感知 -> 规划 -> 行动")
    print("-" * 62)
    print(f"模式      : {'离线（不调用模型）' if not cfg.available else '在线（真实模型）'}")
    if cfg.available:
        print(f"视觉模型  : {cfg.vlm_model}")
        print(f"规划模型  : {cfg.llm_model}")
        print(f"接口地址  : {cfg.base_url}")
        print(f"密钥      : {cfg.masked_key()}")
    print(f"最大轮次  : {cfg.max_step}")
    print(f"打转检测  : {args.stuck}")
    print("=" * 62)

    loop = AgentLoop(
        cfg,
        headless=not args.render,
        delay=args.delay if args.render else 0.0,
        label=args.label,
        seed=args.seed,
        stuck_detector=args.stuck,
    )
    layout = loop.world.layout()
    print(f"世界种子  : {layout['seed']}")
    print(f"出生点    : ({layout['start'][0]},{layout['start'][1]}) 朝向 {layout['start'][2]}")
    print(f"内墙      : {['%d,%d' % tuple(w) for w in layout['inner_walls']]}")
    print("-" * 62)
    try:
        result = loop.run()
    finally:
        loop.world.close()

    print("-" * 62)
    print(f"结果      : {'成功' if result.success else '失败'}（{result.reason}）")
    print(f"步数      : {result.steps}")
    print(f"观察一致率: {result.summary['consistent_rate']}%")
    print(f"降级次数  : {result.summary['fallbacks']}")
    print(f"总耗时    : {result.summary['elapsed']} s")
    print(f"轨迹目录  : {result.run_dir}")
    print(f"演示报告  : {result.run_dir / 'report.html'}")
    print("=" * 62)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
