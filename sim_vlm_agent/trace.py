"""模块 5：轨迹记录。

每轮落盘一张截图 + 一条 JSONL，跑完生成一份可直接打开的 report.html
和一段 run.gif——面试演示时比讲十句话有用。
"""

from __future__ import annotations

import html
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

CSS = """
:root{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--ok:#16a34a;--bad:#dc2626;--warn:#d97706}
*{box-sizing:border-box}
body{margin:0;background:#f1f5f9;color:var(--ink);
  font:14px/1.7 "Microsoft YaHei","PingFang SC",-apple-system,Segoe UI,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:32px 20px 64px}
header{background:var(--ink);color:#fff;border-radius:14px;padding:26px 28px;margin-bottom:22px}
header h1{margin:0 0 6px;font-size:20px;letter-spacing:.4px}
header p{margin:0;color:#94a3b8;font-size:13px}
.badge{display:inline-block;padding:3px 12px;border-radius:999px;font-size:12px;
  font-weight:700;margin-right:8px}
.badge.ok{background:rgba(22,163,74,.18);color:#4ade80}
.badge.bad{background:rgba(220,38,38,.18);color:#f87171}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:12px;margin:18px 0 26px}
.stat{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.stat b{display:block;font-size:20px;line-height:1.3}
.stat span{color:var(--muted);font-size:12px}
.step{background:#fff;border:1px solid var(--line);border-radius:12px;
  padding:16px;margin-bottom:14px;display:grid;
  grid-template-columns:200px 1fr;gap:18px}
.step.fail{border-left:3px solid var(--bad)}
.step.done{border-left:3px solid var(--ok)}
.step img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}
.idx{font-size:12px;color:var(--muted);letter-spacing:.6px;margin-bottom:6px}
.act{font-weight:700;font-size:15px;margin:2px 0 8px}
.act code{background:#f1f5f9;padding:2px 8px;border-radius:6px;font-size:13px}
.thought{color:#334155;margin:0 0 6px}
.meta{color:var(--muted);font-size:12px}
.tag{display:inline-block;font-size:11px;padding:1px 8px;border-radius:5px;
  margin-right:6px;border:1px solid var(--line);color:var(--muted)}
.tag.hot{color:var(--warn);border-color:#fcd34d;background:#fffbeb}
.tag.ok{color:var(--ok);border-color:#bbf7d0;background:#f0fdf4}
details{margin-top:8px}summary{cursor:pointer;color:var(--muted);font-size:12px}
pre{background:#0f172a;color:#cbd5e1;padding:12px;border-radius:8px;
  overflow-x:auto;font-size:12px;line-height:1.6}
footer{color:var(--muted);font-size:12px;text-align:center;margin-top:30px}
@media(max-width:640px){.step{grid-template-columns:1fr}}
"""


def _json_block(data: Any) -> str:
    return f"<pre>{html.escape(json.dumps(data, ensure_ascii=False, indent=2))}</pre>"


class Tracer:
    def __init__(self, label: str = "run") -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = RUNS_DIR / f"{label}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.started = time.time()
        self._jsonl = self.run_dir / "trace.jsonl"

    def save_image(self, image_bytes: bytes, step: int) -> str:
        name = f"step_{step:02d}.png"
        (self.run_dir / name).write_bytes(image_bytes)
        return name

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        with self._jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finish(self, summary: dict[str, Any]) -> Path:
        elapsed = time.time() - self.started
        # 原地写回，调用方拿到的 summary 也带耗时，避免两边字段不一致
        summary["elapsed"] = round(elapsed, 2)
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.run_dir / "report.html").write_text(
            self._render_report(summary), encoding="utf-8")
        self._make_gif()
        return self.run_dir

    # ---------- GIF ----------
    def _make_gif(self) -> None:
        frames = []
        for record in self.records:
            path = self.run_dir / record["image"]
            if path.exists():
                frames.append(Image.open(path).convert("RGB"))
        if len(frames) < 2:
            return
        try:
            frames[0].save(
                self.run_dir / "run.gif", save_all=True, append_images=frames[1:],
                duration=1100, loop=0, optimize=True,
            )
        except Exception:
            pass  # GIF 只是锦上添花，失败不影响主流程

    # ---------- HTML ----------
    def _render_report(self, summary: dict[str, Any]) -> str:
        ok = summary.get("success")
        badge = ('<span class="badge ok">任务成功</span>' if ok
                 else '<span class="badge bad">任务失败</span>')
        stats = [
            ("步数", summary.get("steps", 0)),
            ("耗时", f"{summary.get('elapsed', 0)} s"),
            ("视觉模型", summary.get("vlm_model", "-")),
            ("观察一致率", f"{summary.get('consistent_rate', 0)}%"),
            ("降级次数", summary.get("fallbacks", 0)),
        ]
        stat_html = "".join(
            f'<div class="stat"><b>{html.escape(str(value))}</b>'
            f'<span>{html.escape(name)}</span></div>'
            for name, value in stats
        )

        pieces = []
        for record in self.records:
            plan = record.get("plan", {})
            result = record.get("result", {})
            ver = record.get("verification", {})
            obs = record.get("observation", {})
            cls = "done" if result.get("reason") == "grabbed_target" else (
                "" if result.get("ok") else "fail")

            tags = []
            tags.append(f'<span class="tag">来源 {html.escape(plan.get("source", "?"))}</span>')
            if ver.get("consistent"):
                tags.append('<span class="tag ok">观察一致</span>')
            else:
                tags.append('<span class="tag hot">观察存疑</span>')
            if ver.get("hallucinated"):
                tags.append(f'<span class="tag hot">幻觉 {html.escape("、".join(ver["hallucinated"]))}</span>')
            if ver.get("missed"):
                tags.append(f'<span class="tag hot">漏检 {html.escape("、".join(ver["missed"]))}</span>')
            if record.get("fallback"):
                tags.append(f'<span class="tag hot">降级：{html.escape(str(record["fallback"]))}</span>')
            if plan.get("guard"):
                tags.append(f'<span class="tag hot">校验纠正：{html.escape(str(plan["guard"]))}</span>')

            status = "成功" if result.get("ok") else f"失败（{result.get('reason')}）"
            pieces.append(f"""
      <section class="step {cls}">
        <div>
          <img src="{html.escape(record['image'])}" alt="step {record.get('step')}">
          <div class="idx">STEP {record.get('step'):02d}</div>
          <div class="meta">{html.escape(obs.get('brief', ''))}</div>
        </div>
        <div>
          <div class="act">{html.escape(result.get('action', '-'))}
            <code>{html.escape(status)}</code></div>
          <p class="thought">{html.escape(plan.get('thought') or '（模型未给出理由）')}</p>
          <div>{''.join(tags)}</div>
          <div class="meta" style="margin-top:6px">
            朝向 {html.escape(str(record.get('facing_before', '-')))} →
            {html.escape(str(record.get('facing_after', '-')))} ｜
            位置 ({record.get('pos_before', '-')}) → ({record.get('pos_after', '-')}) ｜
            VLM {obs.get('latency', 0)}s ／ 规划 {plan.get('latency', 0)}s
          </div>
          <details><summary>本步原始数据</summary>
            {_json_block({"observation": obs, "verification": ver,
                          "plan": plan, "result": result})}
          </details>
        </div>
      </section>""")

        return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sim-VLM-Agent 运行轨迹</title><style>{CSS}</style></head>
<body><div class="wrap">
  <header>
    <h1>Sim-VLM-Agent 运行轨迹</h1>
    <p>{html.escape(summary.get("task", ""))} ｜ 结束原因：{html.escape(str(summary.get("reason", "-")))}</p>
    <div style="margin-top:12px">{badge}</div>
  </header>
  <div class="stats">{stat_html}</div>
  {''.join(pieces)}
  <footer>感知 → 规划 → 行动 闭环，共 {summary.get("steps", 0)} 轮 ｜
    生成于 {html.escape(str(summary.get("finished_at", "")))}</footer>
</div></body></html>"""
