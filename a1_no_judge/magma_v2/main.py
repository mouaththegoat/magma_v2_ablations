"""CLI entrypoint for MAGMA v2."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from datetime import datetime, UTC
from pathlib import Path

from dotenv import load_dotenv

from magma_v2.agents.orchestrator import ask_human_question, orchestrator, plan_initial_human_clarification
from magma_v2.runtime.actions import log_action
from magma_v2.runtime.artifacts import create_run_dir, write_json
from magma_v2.runtime.human_loop import human_response_as_prompt_text, pending_human_input, write_human_input_response
from magma_v2.runtime.provider_errors import classify_provider_error, install_provider_retry_noise_filter
from magma_v2.runtime.errors.history import load_recovery_state
from magma_v2.runtime.strands_policy import install_run_scoped_strands_file_write_policy, restore_active_run_dir, set_active_run_dir
from magma_v2.runtime.ui import PipelineUI
from scripts.setup_gpu import install_target as setup_gpu_install_target
from scripts.setup_gpu import print_status as setup_gpu_print_status
from scripts.setup_gpu import choose_target as setup_gpu_choose_target


DEFAULT_DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "sepsis_combined.csv")
)

DEFAULT_PROMPT = (
    f"Build a clinical prediction model using {DEFAULT_DATASET_PATH}. "
    "Target is SepsisLabel (1 = sepsis onset, 0 = no sepsis). Infer the patient/group identifiers, modality layout, and split strategy from the request and files. "
    "Use a clinically aware workflow with leakage checks, train/val/test evaluation, "
    "calibration, TRIPOD assessment, and a final report."
)

REWARD_THRESHOLD = 0.9
MAX_RETRIES = 10
RETRY_PATIENCE = 3
RETRY_MIN_DELTA = 0.01
RETRY_MIN_ATTEMPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAGMA v2.")
    parser.add_argument("prompt", nargs="*", help="Natural-language clinical ML request.")
    parser.add_argument("--artifact-root", default="artifacts", help="Base artifact directory.")
    parser.add_argument("--reward-lambda", type=float, default=0.5, help="Reward lambda in RT = lambda AUROC + (1-lambda) TRIPODScore.")
    parser.add_argument("--max-attempts", "--max-iterations", dest="max_attempts", type=int, default=None, help=f"Maximum reward-improvement attempts before stopping. Defaults to MAGMA_MAX_ATTEMPTS or {MAX_RETRIES}.")
    parser.add_argument("--ui", choices=("compact", "verbose", "silent"), default="compact", help="Terminal output mode.")
    parser.add_argument("--agent-delay-seconds", type=float, default=0.0, help="Sleep this many seconds before each nested agent call to reduce provider rate-limit pressure.")
    parser.add_argument("--setup-gpu", choices=("none", "check", "auto", "cu121", "cu118", "cpu"), default="none", help="Optionally verify or install GPU-enabled PyTorch before starting MAGMA.")
    return parser.parse_args()


def _resolve_max_attempts(cli_value: int | None) -> int:
    raw_value = cli_value if cli_value is not None else os.getenv("MAGMA_MAX_ATTEMPTS")
    if raw_value is None:
        return MAX_RETRIES
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid max attempts value {raw_value!r}; expected a positive integer.") from exc
    if value < 1:
        raise SystemExit("Max attempts must be at least 1.")
    return value


def _load_reward_summary(run_dir: str) -> dict[str, object]:
    reward_path = os.path.join(run_dir, "reward.json")
    if not os.path.exists(reward_path):
        return {}

    try:
        with open(reward_path, encoding="utf-8") as reward_file:
            reward_data = json.load(reward_file)
    except (OSError, json.JSONDecodeError):
        return {}

    return reward_data if isinstance(reward_data, dict) else {}


def _load_judge_verdict(run_dir: str) -> dict[str, object]:
    verdict_path = os.path.join(run_dir, "judge_verdict.json")
    if not os.path.exists(verdict_path):
        return {}

    try:
        with open(verdict_path, encoding="utf-8") as verdict_file:
            verdict_data = json.load(verdict_file)
    except (OSError, json.JSONDecodeError):
        return {}

    return verdict_data if isinstance(verdict_data, dict) else {}


def _load_final_report(run_dir: str) -> str | None:
    report_path = os.path.join(run_dir, "final_report.md")
    if not os.path.exists(report_path):
        return None

    try:
        with open(report_path, encoding="utf-8") as report_file:
            report_text = report_file.read().strip()
    except OSError:
        return None

    return report_text or None


def _build_retry_context(run_dir: str, attempt: int, reward_threshold: float) -> str:
    if attempt <= 1:
        return ""

    reward_summary = _load_reward_summary(run_dir)
    verdict = _load_judge_verdict(run_dir)
    improvement_plan = verdict.get("improvement_plan") if isinstance(verdict.get("improvement_plan"), dict) else {}
    judge_retry_signals = verdict.get("judge_retry_signals") if isinstance(verdict.get("judge_retry_signals"), dict) else {}
    reward_value = reward_summary.get("reward")
    auroc = reward_summary.get("auroc")
    tripod_score = reward_summary.get("tripod_score")

    if isinstance(reward_value, (int, float)) and reward_value < reward_threshold:
        retry_payload = {
            "previous_reward_summary": reward_summary,
            "improvement_plan": improvement_plan,
            "judge_retry_signals": judge_retry_signals,
        }
        return (
            "This is a reward-improvement retry, not a fresh data rebuild.\n"
            f"Previous reward={reward_value}, threshold={reward_threshold}, AUROC={auroc}, TRIPODScore={tripod_score}.\n"
            "Use judge_verdict improvement_plan as the next-trial critique and exploration plan.\n"
            f"Structured retry context:\n{json.dumps(retry_payload, indent=2, default=str)}\n\n"
            "Required retry behavior:\n"
            "- Keep DATA_HANDOFF fixed when improvement_plan.preserve_data_handoff=true unless a concrete data-owned failure is found.\n"
            "- Do not finalize from existing artifacts just because they are valid or because the judge marked the run passed; reward is still below threshold.\n"
            "- The judge does not provide a retry/no-retry decision; the reward threshold controls continuation.\n"
            "- If owner_stage=model, call model_worker with the required_next_actions and required_new_artifacts from improvement_plan.\n"
            "- For model-owned retries, preserve leakage controls, run bounded hyperparameter optimization, test candidate families, select by validation AUROC/AUPRC, and write explicit search/selection/convergence artifacts.\n"
            "- After model improvement, call model_worker_validator and judge again so reward.json is refreshed.\n"
            "- If the judge improvement plan has no high-confidence action, perform bounded exploration: try untried repo-declared model families, feature strategies, convergence instrumentation, calibration, or TRIPOD/evaluation artifact improvements.\n\n"
        )

    final_report = _load_final_report(run_dir)
    return (
        "This is a retry attempt. Re-inspect artifacts and repair only real blocking issues.\n"
        "Previous final report for context:\n"
        f"{final_report or 'Final report not available.'}\n\n"
    )


def _retry_reward_plateaued(
    history: list[float | None],
    *,
    patience: int = RETRY_PATIENCE,
    min_delta: float = RETRY_MIN_DELTA,
    min_attempts: int = RETRY_MIN_ATTEMPTS,
) -> bool:
    valid = [value for value in history if isinstance(value, (int, float))]
    if len(valid) < min_attempts or len(valid) <= patience:
        return False
    previous_best = max(valid[:-patience])
    recent_best = max(valid[-patience:])
    return recent_best < previous_best + min_delta


def _snapshot_attempt_artifacts(run_dir: str, attempt: int, reward_summary: dict[str, object]) -> str:
    run_path = Path(run_dir).resolve()
    snapshot_dir = run_path / "trials" / f"attempt_{attempt:02d}"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied_root_files: list[str] = []
    for path in sorted(run_path.iterdir()):
        if not path.is_file():
            continue
        target = snapshot_dir / path.name
        shutil.copy2(path, target)
        copied_root_files.append(path.name)

    copied_dirs: list[str] = []
    for dirname in ("data", "model", "scratch"):
        source = run_path / dirname
        if source.exists() and source.is_dir():
            shutil.copytree(
                source,
                snapshot_dir / dirname,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            copied_dirs.append(dirname)

    manifest = {
        "attempt": attempt,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "snapshot_dir": str(snapshot_dir),
        "reward_summary": reward_summary,
        "copied_root_files": copied_root_files,
        "copied_stage_dirs": copied_dirs,
    }
    write_json(str(snapshot_dir / "snapshot_manifest.json"), manifest)

    index_path = run_path / "trials" / "index.json"
    existing = []
    if index_path.exists():
        try:
            loaded = json.loads(index_path.read_text())
            existing = loaded if isinstance(loaded, list) else []
        except (OSError, json.JSONDecodeError):
            existing = []
    existing = [item for item in existing if not (isinstance(item, dict) and item.get("attempt") == attempt)]
    existing.append(manifest)
    write_json(str(index_path), sorted(existing, key=lambda item: item.get("attempt", 0)))
    return str(snapshot_dir)


def _run_gpu_setup(mode: str) -> None:
    if mode == "none":
        return
    if mode == "check":
        if not setup_gpu_print_status():
            raise SystemExit("GPU setup check failed: CUDA is not available to PyTorch.")
        return

    target = setup_gpu_choose_target(None if mode == "auto" else mode)
    print(f"[setup-gpu] selected target: {target}")
    rc = setup_gpu_install_target(target)
    if rc != 0:
        raise SystemExit(f"GPU setup failed for target {target} with exit code {rc}.")
    if not setup_gpu_print_status():
        raise SystemExit("GPU setup completed but CUDA is still not available to PyTorch.")


async def run(
    prompt: str,
    artifact_root: str,
    reward_lambda: float,
    ui_mode: str = "compact",
    agent_delay_seconds: float = 0.0,
    setup_gpu: str = "none",
    max_attempts: int = MAX_RETRIES,
) -> str:
    run_dir = create_run_dir(prompt, artifact_root)
    previous_active_run_dir = set_active_run_dir(run_dir)
    ui = PipelineUI(ui_mode)
    try:
        max_attempts = max(1, int(max_attempts))
        task_input = {
            "raw_prompt": prompt,
            "reward_lambda": reward_lambda,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "run_dir": os.path.abspath(run_dir),
            "setup_gpu": setup_gpu,
            "max_attempts": max_attempts,
        }
        write_json(os.path.join(run_dir, "task_input.json"), task_input)
        log_action(run_dir, "main", "start_run", state_summary=prompt, output_refs=[os.path.join(run_dir, "task_input.json")])

        invocation_state = {
            "run_dir": os.path.abspath(run_dir),
            "reward_lambda": reward_lambda,
            "agent_delay_seconds": max(0.0, float(agent_delay_seconds or 0.0)),
        }
        ui.show_start(prompt, run_dir, reward_lambda)
        attempt = 0
        reward: float | None = None
        reward_history: list[float | None] = []
        should_retry = True
        while should_retry:
            attempt += 1
            log_action(
                os.path.abspath(run_dir),
                "main",
                "start_attempt",
                metadata={"attempt": attempt, "max_retries": max_attempts, "reward_threshold": REWARD_THRESHOLD},
            )

            while True:
                human_answers = human_response_as_prompt_text(os.path.abspath(run_dir))
                invocation_state.update(
                    {
                        "user_request": prompt,
                        "human_answers_text": human_answers,
                        "attempt": attempt,
                    }
                )
                run_root = os.path.abspath(run_dir)
                assessment = await plan_initial_human_clarification(
                    run_dir=run_root,
                    user_request=prompt,
                    prior_answers_text=human_answers,
                )
                if isinstance(assessment, dict) and assessment.get("needs_human_input"):
                    ask_human_question(assessment, tool_context={"invocation_state": invocation_state})
                    request = pending_human_input(run_root)
                    if request:
                        answers = ui.collect_human_input(request)
                        response = write_human_input_response(run_root, request, answers)
                        log_action(
                            run_root,
                            "main",
                            "capture_human_input",
                            output_refs=[],
                            metadata={"request_id": response.get("request_id"), "answer_keys": sorted((response.get("answers") or {}).keys())},
                        )
                        continue
                previous_attempt_context = _build_retry_context(os.path.abspath(run_dir), attempt, REWARD_THRESHOLD)
                orchestrator_prompt = (
                    f"User request:\n{prompt}\n\n"
                    + previous_attempt_context
                    + (f"{human_answers}\n\n" if human_answers else "")
                    + f"Run directory: {os.path.abspath(run_dir)}\n"
                    + f"Reward lambda: {reward_lambda}\n"
                    + f"Attempt: {attempt} of {max_attempts}\n"
                )

                try:
                    stream = orchestrator.stream_async(orchestrator_prompt, invocation_state=invocation_state)
                    async for event in stream:
                        ui.show_event(event)
                except Exception as exc:
                    classified = classify_provider_error(exc)
                    log_action(
                        run_root,
                        "main",
                        "orchestrator_stream_failed",
                        success=False,
                        error=classified["message"],
                        metadata={"provider_error": classified},
                    )
                    if ui_mode != "silent":
                        print(f"\n[provider error] {classified['kind']}: {classified['message']}")
                        if classified["detail"]:
                            print(classified["detail"])
                    break

                request = pending_human_input(run_root)
                if not request:
                    break
                answers = ui.collect_human_input(request)
                response = write_human_input_response(run_root, request, answers)
                log_action(
                    run_root,
                    "main",
                    "capture_human_input",
                    output_refs=[],
                    metadata={"request_id": response.get("request_id"), "answer_keys": sorted((response.get("answers") or {}).keys())},
                )

            reward_summary = _load_reward_summary(os.path.abspath(run_dir))
            reward_value = reward_summary.get("reward")
            reward = reward_value if isinstance(reward_value, (int, float)) else None
            reward_history.append(reward)
            below_threshold = isinstance(reward, float) and reward < REWARD_THRESHOLD
            reward_plateaued = _retry_reward_plateaued(reward_history)
            recovery_state = load_recovery_state(os.path.abspath(run_dir))
            terminal_failure = recovery_state.run_status == "failed" and bool(recovery_state.final_escalation_reason)
            should_retry = attempt < max_attempts and below_threshold and not reward_plateaued and not terminal_failure
            ui.show_attempt_summary(
                attempt=attempt,
                reward_summary=reward_summary,
                reward_threshold=REWARD_THRESHOLD,
                will_retry=should_retry,
                max_retries=max_attempts,
            )
            log_action(
                os.path.abspath(run_dir),
                "main",
                "finish_attempt",
                metadata={
                    "attempt": attempt,
                    "reward": reward,
                    "reward_threshold": REWARD_THRESHOLD,
                    "max_attempts": max_attempts,
                    "below_threshold": below_threshold,
                    "reward_history": reward_history,
                    "reward_plateaued": reward_plateaued,
                    "terminal_failure": terminal_failure,
                    "terminal_reason": recovery_state.final_escalation_reason,
                    "retry_patience": RETRY_PATIENCE,
                    "retry_min_delta": RETRY_MIN_DELTA,
                    "retry_min_attempts": RETRY_MIN_ATTEMPTS,
                    "retry_control": "reward_threshold",
                },
            )
            snapshot_path = _snapshot_attempt_artifacts(os.path.abspath(run_dir), attempt, reward_summary)
            log_action(
                os.path.abspath(run_dir),
                "main",
                "snapshot_attempt_artifacts",
                output_refs=[snapshot_path],
                metadata={"attempt": attempt, "snapshot_path": snapshot_path},
            )

        ui.show_done(run_dir)
        return os.path.abspath(run_dir)
    finally:
        restore_active_run_dir(previous_active_run_dir)


def main() -> None:
    load_dotenv()
    install_provider_retry_noise_filter()
    install_run_scoped_strands_file_write_policy()
    args = parse_args()
    max_attempts = _resolve_max_attempts(args.max_attempts)
    _run_gpu_setup(args.setup_gpu)
    prompt = " ".join(args.prompt).strip() or DEFAULT_PROMPT
    asyncio.run(run(prompt, args.artifact_root, args.reward_lambda, args.ui, args.agent_delay_seconds, args.setup_gpu, max_attempts))


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
