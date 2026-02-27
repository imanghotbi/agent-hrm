import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Command
from pymongo import MongoClient

from app.config.config import config
from app.config.logger import logger
from app.workflow.builder import build_graph

RUNTIME_DIR = Path("runtime")
RUN_STATE_FILE = RUNTIME_DIR / "run_state.json"
parser = StrOutputParser()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def load_run_state() -> dict | None:
    if not RUN_STATE_FILE.exists():
        return None
    try:
        return json.loads(RUN_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not parse runtime/run_state.json; starting fresh.")
        return None


def save_run_state(payload: dict) -> None:
    ensure_runtime_dir()
    RUN_STATE_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_interrupt(snapshot) -> str | dict | None:
    if not snapshot.next:
        return None
    tasks = getattr(snapshot, "tasks", None) or []
    if not tasks:
        return None
    interrupts = getattr(tasks[0], "interrupts", None) or []
    if not interrupts:
        return None
    return interrupts[0].value


def _to_dict_if_possible(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def write_final_artifact(values: dict) -> Path:
    ensure_runtime_dir()
    session_id = values.get("session_id", "unknown")
    artifact_path = RUNTIME_DIR / f"review_result_{session_id}.json"
    payload = {
        "session_id": session_id,
        "hiring_reqs": _to_dict_if_possible(values.get("hiring_reqs")),
        "review_started_at": values.get("review_started_at"),
        "review_completed_at": values.get("review_completed_at"),
        "review_duration_seconds": values.get("review_duration_seconds"),
        "review_summary": values.get("review_summary"),
    }
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifact_path


def persist_runtime_snapshot(
    snapshot,
    session_id: str,
    resume_dir: str,
    status: str,
    error: str | None = None,
) -> None:
    values = snapshot.values if snapshot else {}
    interrupt_value = extract_interrupt(snapshot) if snapshot else None
    run_state = {
        "session_id": session_id,
        "resume_dir": resume_dir,
        "status": status,
        "updated_at": utc_now_iso(),
        "next_nodes": list(snapshot.next) if snapshot and snapshot.next else [],
        "interrupt": interrupt_value,
        "error": error,
        "files_count": len(values.get("all_files", [])) if values else 0,
        "evaluated_count": len(values.get("evaluated_results", [])) if values else 0,
        "review_started_at": values.get("review_started_at") if values else None,
        "review_completed_at": values.get("review_completed_at") if values else None,
        "review_duration_seconds": values.get("review_duration_seconds") if values else None,
    }
    save_run_state(run_state)


def print_event_updates(event: dict) -> None:
    for _, updates in event.items():
        if not updates:
            continue
        if "hiring_messages" in updates:
            messages = updates["hiring_messages"]
            if isinstance(messages, list) and messages:
                last_message = messages[-1]
                if getattr(last_message, "type", None) == "ai" and getattr(last_message, "content", None):
                    print(f"\nAgent: {parser.invoke(last_message)}\n")


async def prompt_user(prompt_text: str) -> str:
    return (await asyncio.to_thread(input, prompt_text)).strip()


async def run_cycle(graph, thread_config: dict, session_id: str, resume_dir: str, initial_input):
    next_input = initial_input

    while True:
        async for event in graph.astream(next_input, config=thread_config, stream_mode="updates"):
            print_event_updates(event)

        snapshot = await graph.aget_state(thread_config)
        interrupt_value = extract_interrupt(snapshot)

        if not snapshot.next:
            values = snapshot.values or {}
            artifact_path = write_final_artifact(values)
            persist_runtime_snapshot(snapshot, session_id, resume_dir, status="completed")

            print("Resume review process completed.")
            print(f"Session ID: {session_id}")
            print(f"Duration (seconds): {values.get('review_duration_seconds')}")
            print(f"Result artifact: {artifact_path}")
            return

        if interrupt_value == "hiring_input":
            persist_runtime_snapshot(snapshot, session_id, resume_dir, status="interrupted")
            user_text = await prompt_user("You: ")
            next_input = Command(resume=user_text)
            continue

        persist_runtime_snapshot(snapshot, session_id, resume_dir, status="running")
        next_input = None


async def main() -> None:
    cli_parser = argparse.ArgumentParser(description="Hiring-only resume review runner.")
    cli_parser.add_argument("--resume-dir", default=None, help="Local folder containing PDF resumes.")
    cli_parser.add_argument("--session-id", default=None, help="Resume a specific session id.")
    cli_parser.add_argument("--new-session", action="store_true", help="Ignore saved run_state and start fresh.")
    args = cli_parser.parse_args()

    mongo_client = MongoClient(config.mongo_uri)
    checkpointer = MongoDBSaver(mongo_client)
    graph = build_graph().compile(checkpointer=checkpointer)

    saved_state = load_run_state()
    session_id = args.session_id
    resume_dir = args.resume_dir or config.resume_source_dir
    resuming_existing = False

    if not session_id and not args.new_session and saved_state:
        previous_status = saved_state.get("status")
        if previous_status in {"running", "interrupted", "error"}:
            session_id = saved_state.get("session_id")
            resume_dir = saved_state.get("resume_dir") or resume_dir
            resuming_existing = True

    if not session_id:
        session_id = f"session_{uuid4().hex}"
        resuming_existing = False

    thread_config = {"configurable": {"thread_id": session_id}}
    logger.info(f"Using session_id={session_id} resume_dir={resume_dir}")

    if resuming_existing:
        snapshot = await graph.aget_state(thread_config)
        interrupt_value = extract_interrupt(snapshot)
        persist_runtime_snapshot(snapshot, session_id, resume_dir, status="running")
        if interrupt_value == "hiring_input":
            user_text = await prompt_user("You: ")
            initial_input = Command(resume=user_text)
        else:
            initial_input = None
    else:
        initial_input = {
            "session_id": session_id,
            "resume_dir": str(Path(resume_dir).expanduser()),
            "all_files": [],
            "evaluated_results": [],
            "ocr_results": [],
            "hiring_messages": [HumanMessage(content="شروع کن و نیازمندی‌های استخدام را جمع‌آوری کن.")],
        }
        save_run_state({
            "session_id": session_id,
            "resume_dir": str(Path(resume_dir).expanduser()),
            "status": "running",
            "updated_at": utc_now_iso(),
            "next_nodes": [],
            "interrupt": None,
            "error": None,
        })

    try:
        await run_cycle(graph, thread_config, session_id, str(Path(resume_dir).expanduser()), initial_input)
    except Exception as exc:
        logger.exception("Unhandled runtime error")
        snapshot = await graph.aget_state(thread_config)
        persist_runtime_snapshot(snapshot, session_id, str(Path(resume_dir).expanduser()), status="error", error=str(exc))
        raise


if __name__ == "__main__":
    asyncio.run(main())
