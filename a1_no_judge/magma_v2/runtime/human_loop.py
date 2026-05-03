"""Human-in-the-loop request/response artifacts for MAGMA v2."""

from __future__ import annotations

from datetime import UTC, datetime
import os
from typing import Any
from uuid import uuid4

from magma_v2.runtime.artifacts import read_json, write_json
from magma_v2.runtime.errors.schema import StageError


def human_input_request_path(run_dir: str) -> str:
    return os.path.join(run_dir, "HUMAN_INPUT_REQUEST.json")


def human_input_response_path(run_dir: str) -> str:
    return os.path.join(run_dir, "HUMAN_INPUT_RESPONSE.json")


def load_human_input_request(run_dir: str) -> dict[str, Any] | None:
    path = human_input_request_path(run_dir)
    return read_json(path) if os.path.exists(path) else None


def load_human_input_response(run_dir: str) -> dict[str, Any] | None:
    path = human_input_response_path(run_dir)
    return read_json(path) if os.path.exists(path) else None


def pending_human_input(run_dir: str) -> dict[str, Any] | None:
    request = load_human_input_request(run_dir)
    if not request:
        return None
    response = load_human_input_response(run_dir)
    if not response:
        return request
    if response.get("request_id") != request.get("request_id"):
        return request
    if not _has_nonempty_answer(response.get("answers")):
        return request
    return None


def write_human_input_request(run_dir: str, error: StageError) -> dict[str, Any]:
    existing = load_human_input_request(run_dir)
    if existing and pending_human_input(run_dir) and _request_matches_error(existing, error):
        return existing

    details = error.structured_details or {}
    required = details.get("required_user_input")
    questions: list[dict[str, Any]] = []
    if isinstance(required, list) and required:
        for index, item in enumerate(required, start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or f"question_{index}")
            description = str(item.get("description") or item.get("prompt") or f"Provide {name}.")
            questions.append({
                "name": name,
                "description": description,
                "prompt": str(item.get("prompt") or description),
            })
    if not questions:
        questions = [{
            "name": "human_decision",
            "description": error.repair_instruction or error.message,
            "prompt": error.repair_instruction or error.message,
        }]

    request = {
        "status": "awaiting_human_input",
        "request_id": str(uuid4()),
        "stage": error.stage,
        "owner_stage": "human",
        "error_class": error.error_class,
        "error_subclass": error.error_subclass,
        "message": error.message,
        "root_cause_hypothesis": error.root_cause_hypothesis,
        "repair_instruction": error.repair_instruction,
        "suggested_next_action": error.suggested_next_action or "ask_human",
        "questions": questions,
        "related_artifacts": error.related_artifacts,
        "blocking_artifacts": error.blocking_artifacts,
        "observed_inputs": details.get("observed_inputs"),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    write_json(human_input_request_path(run_dir), request)
    return request


def write_human_input_response(run_dir: str, request: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    response = {
        "status": "provided",
        "request_id": request.get("request_id"),
        "stage": request.get("stage"),
        "answers": answers,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    write_json(human_input_response_path(run_dir), response)
    return response


def human_response_as_prompt_text(run_dir: str) -> str:
    response = load_human_input_response(run_dir)
    request = load_human_input_request(run_dir)
    if not response or not isinstance(response.get("answers"), dict):
        return ""
    lines = ["Human answers collected for this run:"]
    answers = {
        key: value
        for key, value in response["answers"].items()
        if str(value).strip()
    }
    if not answers:
        return ""
    prompts = {item.get("name"): item.get("prompt") for item in (request or {}).get("questions", []) if isinstance(item, dict)}
    for key, value in answers.items():
        label = prompts.get(key) or key
        lines.append(f"- {label}: {value}")
    lines.append("These answers override prior ambiguity for this run. Use them explicitly.")
    return "\n".join(lines)


def _has_nonempty_answer(answers: Any) -> bool:
    if not isinstance(answers, dict):
        return False
    return any(str(value).strip() for value in answers.values())


def _request_matches_error(request: dict[str, Any], error: StageError) -> bool:
    return (
        request.get("stage") == error.stage
        and request.get("error_class") == error.error_class
        and request.get("error_subclass") == error.error_subclass
        and request.get("message") == error.message
    )
