from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel, Field

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class OrchestratorPlan(BaseModel):
    complexity: str = Field(description="simple | medium | complex")
    selected_images: list[str] = Field(description="从目录中选择要分析的图片路径")
    worker_budgets: dict[str, int] = Field(description="各 worker 的调用预算")
    stop_conditions: list[str]


class WorkerResult(BaseModel):
    vision_findings: list[str]
    prompt_understanding: list[str]
    uncertainties: list[str]


class SynthesizedDraft(BaseModel):
    answer: str
    evidence: list[str]
    unresolved: list[str]


class EvaluationReport(BaseModel):
    quality_score: float = Field(ge=0, le=1)
    passed_checks: list[str]
    failed_checks: list[str]
    revise_instructions: list[str]


class FinalAnswer(BaseModel):
    final_answer: str
    confidence: float = Field(ge=0, le=1)
    used_images: list[str]
    evidence: list[str]
    limitations: list[str]


@dataclass
class OrderedMessageBus:
    path: Path
    soft_context_limit: int = 14000

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _read_all(self) -> list[dict[str, Any]]:
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    def append(self, sender: str, stage: str, content: str) -> dict[str, Any]:
        rows = self._read_all()
        record = {
            "seq": len(rows) + 1,
            "sender": sender,
            "stage": stage,
            "content": content,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def export_ordered_text(self, max_items: int = 14) -> str:
        rows = self._read_all()[-max_items:]
        return "\n".join(
            f"[{r['seq']}] {r['stage']} ({r['sender']}): {r['content']}" for r in rows
        )

    def maybe_compress(self) -> bool:
        raw = self.path.read_text(encoding="utf-8")
        if len(raw) <= self.soft_context_limit:
            return False

        rows = self._read_all()
        if len(rows) < 6:
            return False

        compressible = [
            r
            for r in rows
            if r["stage"]
            in {
                "orchestrator_plan",
                "worker_vision",
                "worker_prompt",
                "worker_pack",
                "synthesis_draft",
            }
        ]
        if not compressible:
            return False

        keep = []
        summary_lines = []
        compress_ids = {id(x) for x in compressible[:-2]}

        for r in rows:
            if id(r) in compress_ids:
                summary_lines.append(
                    f"[{r['seq']}] {r['stage']}: {r['content'].replace(chr(10), ' ')[:120]}"
                )
            else:
                keep.append(r)

        summary_record = {
            "seq": 1,
            "sender": "system",
            "stage": "context_summary",
            "content": " | ".join(summary_lines),
        }

        rewritten = [summary_record]
        for i, r in enumerate(keep, start=2):
            row = dict(r)
            row["seq"] = i
            rewritten.append(row)

        with self.path.open("w", encoding="utf-8") as f:
            for r in rewritten:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return True


class LocalImageCatalogTool(BaseTool):
    name: str = "local_image_catalog"
    description: str = "输入目录路径，返回其中图片文件列表(JSON)。"

    def _run(self, folder: str) -> str:
        root = Path(folder)
        if not root.exists() or not root.is_dir():
            return json.dumps({"error": f"folder not found: {folder}"}, ensure_ascii=False)

        images = []
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(str(p.resolve()))
        return json.dumps({"images": images}, ensure_ascii=False)


class VisionQATool(BaseTool):
    name: str = "vision_qa"
    description: str = (
        "输入 JSON: {'image_path': '...', 'question': '...'}。"
        "支持 GPT 和 Claude 视觉模型（通过 model/provider 参数）。"
    )

    def _run(self, input_json: str) -> str:
        payload = json.loads(input_json)
        image_path = payload["image_path"]
        question = payload["question"]
        model = payload.get("model", "gpt-4.1-mini")
        provider = payload.get("provider", "auto")

        image_file = Path(image_path)
        if not image_file.exists():
            return f"ERROR: image not found: {image_path}"

        if provider == "auto":
            provider = "claude" if "claude" in model.lower() else "gpt"

        mime = mimetypes.guess_type(image_file.name)[0] or "image/jpeg"
        image_bytes = image_file.read_bytes()
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        if provider == "claude":
            client = Anthropic()
            media_type = mime if mime.startswith("image/") else "image/jpeg"
            message = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ],
            )
            text_parts = [b.text for b in message.content if getattr(b, "type", "") == "text"]
            return "\n".join(text_parts).strip()

        client = OpenAI()
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                    ],
                }
            ],
        )
        return response.output_text


class MultiAgentFolderVisionQA:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        vision_model: str | None = None,
        provider: str = "auto",
        comm_file: str = "./runtime/agent_bus.jsonl",
    ) -> None:
        self.model = model
        self.vision_model = vision_model or model
        self.provider = provider
        self.bus = OrderedMessageBus(Path(comm_file))
        self.catalog_tool = LocalImageCatalogTool()
        self.vision_tool = VisionQATool()

    def _orchestrator(self) -> Agent:
        return Agent(
            role="Orchestrator",
            goal="拆解任务，选择图片，分配 worker 预算与停止条件",
            backstory="你负责资源分配，不做内容结论。",
            llm=self.model,
            verbose=True,
        )

    def _prompt_worker(self) -> Agent:
        return Agent(
            role="PromptWorker",
            goal="澄清用户问题与判定标准",
            backstory="你专注问题语义，不做图像识别。",
            llm=self.model,
            verbose=True,
        )

    def _synthesizer(self) -> Agent:
        return Agent(
            role="Synthesizer",
            goal="汇总 worker 输出形成一致草稿",
            backstory="你只做整合与冲突消解。",
            llm=self.model,
            verbose=True,
        )

    def _evaluator(self) -> Agent:
        return Agent(
            role="Evaluator",
            goal="审查草稿质量并给出修正意见",
            backstory="你严格检查证据覆盖和边界遵守。",
            llm=self.model,
            verbose=True,
        )

    def _finalizer(self) -> Agent:
        return Agent(
            role="Finalizer",
            goal="输出对用户友好的最终答案",
            backstory="你只基于评估通过内容给出最终答复。",
            llm=self.model,
            verbose=True,
        )

    def _catalog_images(self, folder: str) -> list[str]:
        raw = self.catalog_tool.run(folder)
        data = json.loads(raw)
        return data.get("images", [])

    def _vision_observations(self, images: list[str], prompt: str) -> list[str]:
        observations: list[str] = []
        for img in images:
            question = f"用户问题: {prompt}\n请只提取可观察事实和风险点，不要过度推断。"
            raw = self.vision_tool.run(
                json.dumps(
                    {
                        "image_path": img,
                        "question": question,
                        "model": self.vision_model,
                        "provider": self.provider,
                    },
                    ensure_ascii=False,
                )
            )
            observations.append(f"image={img}\n{raw}")
        return observations

    def run(self, input_folder: str, prompt: str) -> dict[str, Any]:
        images = self._catalog_images(input_folder)
        self.bus.append(
            "user",
            "user_input",
            json.dumps(
                {
                    "input_folder": input_folder,
                    "prompt": prompt,
                    "image_count": len(images),
                    "llm_model": self.model,
                    "vision_model": self.vision_model,
                    "provider": self.provider,
                },
                ensure_ascii=False,
            ),
        )
        self.bus.append("system", "catalog", json.dumps({"images": images}, ensure_ascii=False))
        self.bus.maybe_compress()

        orchestrator_task = Task(
            description=(
                "根据用户问题与图片列表制定执行计划，并选择要分析的图片。"
                "必须输出 OrchestratorPlan JSON。\n\n"
                f"ordered_messages:\n{self.bus.export_ordered_text()}"
            ),
            expected_output="OrchestratorPlan JSON",
            output_pydantic=OrchestratorPlan,
            agent=self._orchestrator(),
        )
        Crew(
            agents=[orchestrator_task.agent],
            tasks=[orchestrator_task],
            process=Process.sequential,
            verbose=True,
        ).kickoff()

        plan = (
            orchestrator_task.output.pydantic.model_dump()
            if orchestrator_task.output
            else {
                "complexity": "simple",
                "selected_images": images[:2],
                "worker_budgets": {},
                "stop_conditions": [],
            }
        )
        selected_images = plan.get("selected_images") or images[:2]
        self.bus.append("orchestrator", "orchestrator_plan", json.dumps(plan, ensure_ascii=False))
        self.bus.maybe_compress()

        vision_obs = self._vision_observations(selected_images, prompt)
        self.bus.append("vision_worker", "worker_vision", json.dumps({"observations": vision_obs}, ensure_ascii=False))
        self.bus.maybe_compress()

        prompt_task = Task(
            description=(
                "你是 PromptWorker。根据用户 prompt 输出判定标准与回答边界。"
                "必须输出 WorkerResult JSON，其中 vision_findings 可置空。\n\n"
                f"prompt: {prompt}\n"
                f"ordered_messages:\n{self.bus.export_ordered_text()}"
            ),
            expected_output="WorkerResult JSON",
            output_pydantic=WorkerResult,
            agent=self._prompt_worker(),
        )
        Crew(agents=[prompt_task.agent], tasks=[prompt_task], process=Process.sequential, verbose=True).kickoff()

        prompt_worker_result = (
            prompt_task.output.pydantic.model_dump()
            if prompt_task.output
            else {"vision_findings": [], "prompt_understanding": [], "uncertainties": []}
        )
        self.bus.append("prompt_worker", "worker_prompt", json.dumps(prompt_worker_result, ensure_ascii=False))
        self.bus.append(
            "system",
            "worker_pack",
            json.dumps({"vision_observations": vision_obs, "prompt_worker": prompt_worker_result}, ensure_ascii=False),
        )
        self.bus.maybe_compress()

        synth_task = Task(
            description=(
                "你是 Synthesizer。整合 worker_pack，形成 SynthesizedDraft JSON。\n"
                f"ordered_messages:\n{self.bus.export_ordered_text()}"
            ),
            expected_output="SynthesizedDraft JSON",
            output_pydantic=SynthesizedDraft,
            agent=self._synthesizer(),
        )
        Crew(agents=[synth_task.agent], tasks=[synth_task], process=Process.sequential, verbose=True).kickoff()

        synth = (
            synth_task.output.pydantic.model_dump()
            if synth_task.output
            else {"answer": "", "evidence": [], "unresolved": []}
        )
        self.bus.append("synthesizer", "synthesis_draft", json.dumps(synth, ensure_ascii=False))
        self.bus.maybe_compress()

        eval_task = Task(
            description=(
                "你是 Evaluator。评估 synthesis_draft，输出 EvaluationReport JSON。\n"
                f"ordered_messages:\n{self.bus.export_ordered_text()}"
            ),
            expected_output="EvaluationReport JSON",
            output_pydantic=EvaluationReport,
            agent=self._evaluator(),
        )
        Crew(agents=[eval_task.agent], tasks=[eval_task], process=Process.sequential, verbose=True).kickoff()

        evaluation = (
            eval_task.output.pydantic.model_dump()
            if eval_task.output
            else {
                "quality_score": 0.5,
                "passed_checks": [],
                "failed_checks": [],
                "revise_instructions": [],
            }
        )
        self.bus.append("evaluator", "evaluation_report", json.dumps(evaluation, ensure_ascii=False))
        self.bus.maybe_compress()

        final_task = Task(
            description=(
                "你是 Finalizer。基于 evaluation_report 与 synthesis_draft 输出 FinalAnswer JSON。\n"
                f"ordered_messages:\n{self.bus.export_ordered_text()}"
            ),
            expected_output="FinalAnswer JSON",
            output_pydantic=FinalAnswer,
            agent=self._finalizer(),
        )
        raw_result = Crew(
            agents=[final_task.agent],
            tasks=[final_task],
            process=Process.sequential,
            verbose=True,
        ).kickoff()

        final = (
            final_task.output.pydantic.model_dump()
            if final_task.output
            else {
                "final_answer": synth.get("answer", ""),
                "confidence": max(0.0, min(1.0, float(evaluation.get("quality_score", 0.5)))),
                "used_images": selected_images,
                "evidence": synth.get("evidence", []),
                "limitations": synth.get("unresolved", []),
            }
        )

        self.bus.append("finalizer", "final_answer", json.dumps(final, ensure_ascii=False))
        self.bus.maybe_compress()

        return {
            "raw": str(raw_result),
            "plan": plan,
            "selected_images": selected_images,
            "synthesized": synth,
            "evaluation": evaluation,
            "final": final,
            "comm_file": str(self.bus.path),
            "ordered_messages": self.bus.export_ordered_text(max_items=24),
        }
