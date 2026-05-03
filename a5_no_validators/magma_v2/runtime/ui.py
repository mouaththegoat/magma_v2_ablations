"""Terminal UI controller for MAGMA v2."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


ACTION_LABELS = {
    "call_data_worker": "run data worker",
    "call_data_worker_validator": "run data worker validator",
    "call_model_worker": "run model worker",
    "call_model_worker_validator": "run model worker validator",
    "call_judge": "run judge",
    "ask_human_question": "request human input",
    "write_final_report": "write final report",
    "decide_repair_route": "decide repair route",
    "write_run_file": "write run file",
}


@dataclass
class PipelineUI:
    mode: str = "compact"
    _active_tool_signature: str | None = field(default=None, init=False)
    _active_tool_name: str | None = field(default=None, init=False)

    def show_start(self, prompt: str, run_dir: str, reward_lambda: float) -> None:
        if self.mode == "silent":
            return
        print("MAGMA v2 clinical AI pipeline")
        print(f"Run: {os.path.abspath(run_dir)}")
        print(f"Reward: RT = {reward_lambda} * AUROC + {1 - reward_lambda:.3g} * TRIPODScore")
        if self.mode == "verbose":
            print(f"Request: {prompt}")
        print()

    def show_event(self, event: dict[str, Any]) -> None:
        if self.mode == "silent":
            return
        if event.get("type") == "tool_use_stream":
            current = event.get("current_tool_use", {})
            name = current.get("name") if isinstance(current, dict) else None
            if name and self._is_new_tool_event(current, name):
                if name == "ask_human_question":
                    return
                label = ACTION_LABELS.get(name, name)
                if isinstance(current, dict):
                    tool_input = current.get("input", {})
                    path = tool_input.get("path") if isinstance(tool_input, dict) else None
                    if name == "write_run_file" and path:
                        label = f"{label} -> {path}"
                print(f"[action] {label}")
            return
        if event.get("type") in {"tool_result", "tool_use_end"} or "result" in event:
            self._active_tool_signature = None
            self._active_tool_name = None
        if self.mode == "verbose" and "data" in event:
            print(str(event["data"]), end="", flush=True)

    def _is_new_tool_event(self, current: dict[str, Any], name: str) -> bool:
        tool_id = (
            current.get("toolUseId")
            or current.get("tool_use_id")
            or current.get("id")
        )
        signature = f"{name}:{tool_id}" if tool_id else name
        if signature == self._active_tool_signature:
            return False
        if not tool_id and name == self._active_tool_name:
            return False
        self._active_tool_signature = signature
        self._active_tool_name = name
        return True

    def show_done(self, run_dir: str) -> None:
        if self.mode == "silent":
            return
        print()
        print(f"Run artifacts: {os.path.abspath(run_dir)}")
        print(f"Final report: {os.path.abspath(os.path.join(run_dir, 'final_report.md'))}")

    def show_attempt_summary(
        self,
        attempt: int,
        reward_summary: dict[str, Any],
        reward_threshold: float,
        will_retry: bool,
        max_retries: int,
    ) -> None:
        if self.mode == "silent":
            return
        print()
        print(f"[attempt {attempt}] reward summary")
        if reward_summary:
            equation = reward_summary.get("equation")
            reward_value = reward_summary.get("reward")
            auroc = reward_summary.get("auroc")
            tripod_score = reward_summary.get("tripod_score")
            reward_lambda = reward_summary.get("lambda")
            if equation is not None:
                print(f"equation: {equation}")
            print(f"lambda: {reward_lambda}")
            print(f"auroc: {auroc}")
            print(f"tripod_score: {tripod_score}")
            print(f"reward: {reward_value}")
        else:
            print("reward.json not found or could not be parsed.")

        if will_retry:
            print(f"next: triggering a new loop because reward is below {reward_threshold} and attempt {attempt}/{max_retries} has not reached the max.")
        else:
            print(f"next: no new loop will be triggered. stop condition reached for attempt {attempt}/{max_retries}.")

    def collect_human_input(self, request: dict[str, Any]) -> dict[str, str]:
        if self.mode == "silent":
            raise RuntimeError("Human input was requested but UI mode is silent.")
        print()
        print("[action] request human input")
        if request.get("message"):
            print(request["message"])
        print()
        answers: dict[str, str] = {}
        for item in request.get("questions", []):
            if not isinstance(item, dict):
                continue
            key = str(item.get("name") or "answer")
            prompt = str(item.get("prompt") or item.get("description") or key)
            response = input(f"{prompt}\n> ").strip()
            answers[key] = response
        print()
        return answers
