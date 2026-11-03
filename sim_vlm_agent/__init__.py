"""Sim-VLM-Agent：基于视觉大模型的仿真具身智能原型。

感知 -> 规划 -> 行动闭环。四大模块：
    world       2D 仿真环境
    perception  VLM 视觉感知
    planner     LLM 规划 + Function-Calling
    loop        主循环调度器
"""

__version__ = "0.1.0"
