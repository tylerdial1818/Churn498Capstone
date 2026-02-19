"""
Agent service — trigger and monitor agent pipeline runs.
"""

import logging
import uuid

logger = logging.getLogger(__name__)

# In-memory store for agent run status
_agent_runs: dict[str, dict] = {}


def trigger_agent_run(
    pipeline_name: str,
    params: dict | None = None,
    demo_mode: bool = False,
) -> dict:
    """Trigger an agent pipeline run."""
    run_id = str(uuid.uuid4())

    if demo_mode:
        _agent_runs[run_id] = {
            "run_id": run_id,
            "pipeline": pipeline_name,
            "status": "complete",
            "result": {"message": f"Demo {pipeline_name} run completed"},
        }
        return {"run_id": run_id, "status": "complete"}

    _agent_runs[run_id] = {
        "run_id": run_id,
        "pipeline": pipeline_name,
        "status": "running",
        "result": None,
    }

    # In production, would kick off async pipeline execution here
    _agent_runs[run_id]["status"] = "complete"
    _agent_runs[run_id]["result"] = {
        "message": f"{pipeline_name} pipeline completed"
    }

    return {"run_id": run_id, "status": _agent_runs[run_id]["status"]}


def get_agent_status(run_id: str) -> dict | None:
    """Check status of an agent run."""
    run = _agent_runs.get(run_id)
    if run is None:
        return None
    return {
        "run_id": run["run_id"],
        "status": run["status"],
        "result": run["result"],
    }
