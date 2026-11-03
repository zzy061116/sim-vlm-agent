"""配置读取：支持 .env 文件与环境变量，缺少 key 时自动降级为离线模式。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


@dataclass(frozen=True)
class Config:
    api_key: str
    base_url: str
    vlm_model: str
    llm_model: str
    timeout: float
    max_step: int
    offline: bool

    @property
    def available(self) -> bool:
        """是否具备调用真实模型的条件。"""
        return bool(self.api_key) and not self.offline

    def masked_key(self) -> str:
        if not self.api_key:
            return "(空)"
        return f"{self.api_key[:4]}…{self.api_key[-4:]}（{len(self.api_key)} 位）"


def load_config(env_path: Path | None = None, **overrides: object) -> Config:
    file_values = _read_env_file(env_path or ROOT / ".env")

    def pick(key: str, default: str) -> str:
        # 显式覆盖 > 进程环境变量 > .env > 默认值
        if key in overrides and overrides[key] is not None:
            return str(overrides[key])
        return os.environ.get(key) or file_values.get(key) or default

    api_key = pick("LLM_API_KEY", "")
    return Config(
        api_key=api_key,
        base_url=pick("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").rstrip("/"),
        vlm_model=pick("VLM_MODEL", "glm-4v-flash"),
        llm_model=pick("LLM_MODEL", "glm-4-flash"),
        timeout=float(pick("TIMEOUT", "60")),
        max_step=int(pick("MAX_STEP", "24")),
        offline=pick("OFFLINE", "0") in ("1", "true", "True", "yes") or not api_key,
    )
