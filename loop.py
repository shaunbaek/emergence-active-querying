# Episode runners for:
# - simple rule-based agent
# - Gemini-powered single-agent
# - Gemini-powered agent with repair
# - Multi-agent (leader + handicapped) with communication

import os
import time
import json
from typing import List, Dict, Any
import argparse

from env import ThorEnv
from policy import (
    llm_policy_gemini,
    llm_policy_gemini_handicapped,
    get_target_info,
    llm_multiagent_policy_gemini,
)

from tasks_utils import (
    generate_dev_task_list_20,
    generate_task_list_100,
    sample_random_start_position,
    is_success,
)

def ensure_dir_for(path: str):
    """Create parent directory for a file path if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def default_microwave_task_meta() -> Dict[str, Any]:
    """
    Minimal task metadata for current hard-coded experiments.
    Later this will come from tasks_utils (tasks_dev_20).
    """
    return {
        "task_id": 1,
        "scene": "FloorPlan1",
        "room_type": "kitchen",
        "target_type": "Microwave",
        "goal_text": "Navigate to the microwave.",
    }


# Saves attempts at approaching the goal
class TrajectoryLogger:
    """
    Very simple in-memory logger for an episode.
    You can append dicts each step and save them as JSON later.
    """
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def log_step(self, data: Dict[str, Any]):
        self.steps.append(data)

    def to_list(self) -> List[Dict[str, Any]]:
        return self.steps

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.steps, f, indent=2)


def run_gemini_agent_baseline(
    max_steps: int = 15,
    use_memory: bool = True,
    save_path: str | None = None,
    task_meta: Dict[str, Any] | None = None,
):
    """
    Single-agent baseline with no repair logic.
    Logs per-step data in a format compatible with analyze_baseline_vs_two_agent.py.
    """
    if task_meta is None:
        task_meta = default_microwave_task_meta()

    scene = task_meta["scene"]
    goal_text = task_meta["goal_text"]
    target_type = task_meta["target_type"]

    # Default save path if not provided
    if save_path is None:
        save_path = f"logs/baseline/task_{task_meta['task_id']}.json"

    env = ThorEnv(scene=scene)
    # randomize agent starting point
    sample_random_start_position(env)
    event = env.get_event()

    memory_lines: List[str] = []
    logger = TrajectoryLogger()

    # Keeping Track of Episode Run Time
    start_time = time.perf_counter()

    # For just in case we hit a redundant choice stream
    last_action = None
    last_target_distance = None
    stuck_counter = 0
    prev_dist = None

    for step_i in range(max_steps):
        # For now still using microwave-specific helper; later, get_target_info(target_type)
        target = get_target_info(event,target_type)
        agent_pos = event.metadata["agent"]["position"]

        # Update short-term memory to remember where we've seen the microwave
        if use_memory and target["visible"] and target["distance"] is not None:
            memory_str = (
                f"Previously saw {target_type} at distance {target['distance']:.2f} m "
                f"near position x={agent_pos['x']:.2f}, z={agent_pos['z']:.2f}."
            )
            memory_lines.append(memory_str)

        # query Gemini for the next reasonable action, passing memory
        action = llm_policy_gemini(
            event,
            goal_text=goal_text,
            target_type=target_type,
            proximity_done=1.0,
            memory_lines=memory_lines if use_memory else None,
            last_action=last_action,
            last_target_distance=last_target_distance,
        )

        last_action = action
        last_target_distance = target["distance"]

        print(
            f"[Baseline][Step {step_i}] action={action}, "
            f"{target_type}_visible={target['visible']}, "
            f"{target_type}_distance={target['distance']}"
        )

        cur_dist = target["distance"]

        # detect non-progress
        if prev_dist is not None and cur_dist is not None and cur_dist >= prev_dist:
            stuck_counter += 1
        else:
            stuck_counter = 0
        prev_dist = cur_dist

        # If the model keeps saying MoveAhead and we see no progress, override once
        if action == "MoveAhead" and stuck_counter >= 2:
            action = "RotateRight"

        #log in target_* format
        logger.log_step(
            {
                "step": step_i,
                "task_id": task_meta["task_id"],
                "scene": scene,
                "room_type": task_meta["room_type"],
                "target_type": target_type,
                "goal_text": goal_text,
                "policy": "baseline",
                "role": "solo",

                "is_repair": False,
                "repair_helpful": None,

                "action": action,
                "agent_position": agent_pos,
                "target_visible": target["visible"],
                "target_distance": target["distance"],

                "memory_snapshot": list(memory_lines) if use_memory else None,
                "message": "",
                "conversation_log": None,
            }
        )


        if action == "Done":
            print("Gemini believes we reached the target. Stopping.")
            break

        event = env.step(action)
        time.sleep(0.2)

    env.close()

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    ensure_dir_for(save_path)
    result_data = logger.to_list()
    output = {
        "task_meta": task_meta,
        "episode_time_sec": elapsed,
        "steps": result_data,
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved baseline trajectory to {save_path} (elapsed {elapsed:.2f}s)")
    return result_data

# --------- High-level token -> concrete action mapping for multi-agent --------- #

def make_env_action_from_token(token: str, event, agent_id: int = 0):
    """
    Map a high-level action token to a concrete AI2-THOR action or action dict.
    - For navigation tokens, just return the string.
    - For manipulation tokens, construct a dict with 'action' and extra args.
    """
    # Navigation actions pass through
    if token in ["MoveAhead", "MoveLeft", "MoveRight", "MoveBack", "RotateRight", "RotateLeft", "Done"]:
        return token

    objs = event.metadata["objects"]
    held_obj = event.metadata.get("heldObject", None)

    # Helper: nearest visible pickupable object
    def nearest_visible_pickupable():
        best = None
        best_dist = None
        for o in objs:
            if not o.get("visible", False):
                continue
            if not o.get("pickupable", False):
                continue
            d = o.get("distance", None)
            if d is None:
                continue
            if best is None or d < best_dist:
                best = o
                best_dist = d
        return best

    # Helper: nearest visible openable (receptacle) object
    def nearest_visible_openable():
        best = None
        best_dist = None
        for o in objs:
            if not o.get("visible", False):
                continue
            if not o.get("openable", False):
                continue
            d = o.get("distance", None)
            if d is None:
                continue
            if best is None or d < best_dist:
                best = o
                best_dist = d
        return best

    if token == "PickupNearestVisibleObject":
        target = nearest_visible_pickupable()
        if target is None:
            # fallback: rotate to search more
            return "RotateRight"
        return {
            "action": "PickupObject",
            "objectId": target["objectId"],
            "agentId": agent_id,
        }

    if token == "DropHeldObject":
        if held_obj is not None:
            return {
                "action": "DropHandObject",
                "agentId": agent_id,
            }
        return "RotateRight"

    if token == "OpenNearestVisibleReceptacle":
        target = nearest_visible_openable()
        if target is None:
            return "RotateRight"
        return {
            "action": "OpenObject",
            "objectId": target["objectId"],
            "agentId": agent_id,
        }

    if token == "CloseNearestVisibleReceptacle":
        target = nearest_visible_openable()
        if target is None:
            return "RotateRight"
        return {
            "action": "CloseObject",
            "objectId": target["objectId"],
            "agentId": agent_id,
        }

    # Fallback for unknown tokens
    return "RotateRight"

# -------- Handicapped LLM ------------- #

def run_gemini_agent_handicapped(
    max_steps: int = 15,
    use_memory: bool = True,
    save_path: str | None = None,
    task_meta: Dict[str, Any] | None = None,
):
    """
    Run a single 'handicapped' Gemini agent with reduced perception.
    Logs per-step data in a format consistent with baseline and two-agent episodes.
    """
    if task_meta is None:
        task_meta = default_microwave_task_meta()

    scene = task_meta["scene"]
    goal_text = task_meta["goal_text"]
    target_type = task_meta["target_type"]

    # Default save path if not provided
    if save_path is None:
        save_path = f"logs/handicapped/task_{task_meta['task_id']}.json"

    # --- Initialize environment ---
    env = ThorEnv(scene=scene)
    # randomize start
    # sample_random_start_position(env)
    event = env.get_event()

    memory_lines: List[str] = []
    logger = TrajectoryLogger()

    # Keeping Track of Episode Run Time
    start_time = time.perf_counter()

    for step_i in range(max_steps):
        # Retrieve target info (e.g., microwave, apple, etc.)
        target = get_target_info(event, target_type)
        agent_pos = event.metadata["agent"]["position"]

        # Remember last time the target was seen
        if use_memory and target["visible"] and target["distance"] is not None:
            memory_str = (
                f"Previously saw {target_type} at coarse distance near position "
                f"x={agent_pos['x']:.2f}, z={agent_pos['z']:.2f}."
            )
            memory_lines.append(memory_str)

        # --- Query LLM policy (handicapped) ---
        action = llm_policy_gemini_handicapped(
            event,
            goal_text=goal_text,
            target_type=target_type,
            proximity_done=1.0,
            memory_lines=memory_lines if use_memory else None,
        )

        print(
            f"[Handicapped][Step {step_i}] action={action}, "
            f"{target_type}_visible={target['visible']}, "
            f"{target_type}_distance={target['distance']}"
        )

        # --- Log step data ---
        logger.log_step(
            {
                "step": step_i,
                "task_id": task_meta["task_id"],
                "scene": scene,
                "room_type": task_meta["room_type"],
                "target_type": target_type,
                "goal_text": goal_text,
                "policy": "handicapped",
                "role": "solo",

                "is_repair": False,
                "repair_helpful": None,

                "action": action,
                "agent_position": agent_pos,
                "target_visible": target["visible"],
                "target_distance": target["distance"],

                "memory_snapshot": list(memory_lines) if use_memory else None,
                "message": "",
                "conversation_log": None,
            }
        )

        # --- Check stopping condition ---
        if action == "Done":
            print("Handicapped Gemini believes we reached the target. Stopping.")
            break

        event = env.step(action)
        time.sleep(0.2)

    # --- Finalize ---
    env.close()

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    ensure_dir_for(save_path)
    result_data = logger.to_list()
    output = {
        "task_meta": task_meta,
        "episode_time_sec": elapsed,
        "steps": result_data,
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved handicapped trajectory to {save_path} (elapsed {elapsed:.2f}s)")
    return result_data


# --------- Two-agent LLM runner (leader + handicapped, with communication) --------- #

def run_two_agent_llm(
    max_steps: int = 15,
    use_memory: bool = True,
    save_path: str | None = None,
    task_meta: Dict[str, Any] | None = None,
):
    """
    Run two ManipulaTHOR agents in the same scene, controlled by a single LLM 'mind':
      - Agent 0: LEADER (full state text)
      - Agent 1: HANDICAPPED (compact state text)

    Logs per-step data in a format compatible with analyze_baseline_vs_two_agent.py.
    """
    if task_meta is None:
        task_meta = default_microwave_task_meta()

    scene = task_meta["scene"]
    goal_text = f"Cooperatively navigate to the {task_meta['target_type']}."
    target_type = task_meta["target_type"]

    if save_path is None:
        save_path = f"logs/two_agent/task_{task_meta['task_id']}.json"

    env = ThorEnv(scene=scene, agent_count=2)

    event_leader = env.get_event(agent_id=0)
    event_handicapped = env.get_event(agent_id=1)

    leader_memory: List[str] = []
    handicapped_memory: List[str] = []

    conversation_log: List[str] = []
    logger = TrajectoryLogger()

    # Keeping Track of Episode Run Time
    start_time = time.perf_counter()

    for step_i in range(max_steps):
        target_leader = get_target_info(event_leader,target_type)
        target_handicapped = get_target_info(event_handicapped,target_type)

        pos_leader = event_leader.metadata["agent"]["position"]
        pos_handicapped = event_handicapped.metadata["agent"]["position"]

        # Update memories
        if use_memory and target_leader["visible"] and target_leader["distance"] is not None:
            leader_memory.append(
                f"Leader saw {target_type} at distance {target_leader['distance']:.2f} m "
                f"near position x={pos_leader['x']:.2f}, z={pos_leader['z']:.2f}."
            )

        if use_memory and target_handicapped["visible"] and target_handicapped["distance"] is not None:
            handicapped_memory.append(
                f"Handicapped saw {target_type} at distance {target_handicapped['distance']:.2f} m "
                f"near position x={pos_handicapped['x']:.2f}, z={pos_handicapped['z']:.2f}."
            )

        # --- Multi-agent LLM decision w/ communication ---
        leader_token, handicapped_token, leader_msg, handicapped_msg = llm_multiagent_policy_gemini(
            leader_event=event_leader,
            handicapped_event=event_handicapped,
            goal_text=goal_text,
            proximity_done=1.0,
            leader_memory=leader_memory if use_memory else None,
            handicapped_memory=handicapped_memory if use_memory else None,
            conversation_lines=conversation_log,
        )

        # Append messages to conversation log
        if leader_msg:
            conversation_log.append(f"LEADER: {leader_msg}")
        if handicapped_msg:
            conversation_log.append(f"HANDICAPPED: {handicapped_msg}")

        print(
            f"[Step {step_i}] LeaderAction={leader_token}, "
            f"HandicappedAction={handicapped_token}"
        )
        if leader_msg or handicapped_msg:
            print(f"  LeaderMsg: {leader_msg}")
            print(f"  HandicappedMsg: {handicapped_msg}")

        # Log both agents (note: target_* keys!)
        logger.log_step(
            {
                "step": step_i,
                "task_id": task_meta["task_id"],
                "scene": scene,
                "room_type": task_meta["room_type"],
                "target_type": target_type,
                "goal_text": goal_text,
                "policy": "two_agent",

                "agent_id": 0,
                "role": "leader",
                "action": leader_token,
                "message": leader_msg,
                "agent_position": pos_leader,
                "target_visible": target_leader["visible"],
                "target_distance": target_leader["distance"],

                "leader_memory_snapshot": list(leader_memory) if use_memory else None,
                "handicapped_memory_snapshot": list(handicapped_memory) if use_memory else None,
                "conversation_log": list(conversation_log),
            }
        )
        logger.log_step(
            {
                "step": step_i,
                "task_id": task_meta["task_id"],
                "scene": scene,
                "room_type": task_meta["room_type"],
                "target_type": target_type,
                "goal_text": goal_text,
                "policy": "two_agent",

                "agent_id": 1,
                "role": "handicapped",
                "action": handicapped_token,
                "message": handicapped_msg,
                "agent_position": pos_handicapped,
                "target_visible": target_handicapped["visible"],
                "target_distance": target_handicapped["distance"],

                "leader_memory_snapshot": list(leader_memory) if use_memory else None,
                "handicapped_memory_snapshot": list(handicapped_memory) if use_memory else None,
                "conversation_log": list(conversation_log),
            }
        )

        # Both Done => stop
        if leader_token == "Done" and handicapped_token == "Done":
            print("Both agents signaled Done. Stopping.")
            break

        # Convert tokens to environment actions
        if leader_token != "Done":
            env_action_leader = make_env_action_from_token(leader_token, event_leader, agent_id=0)
            event_leader = env.step(env_action_leader, agent_id=0)

        if handicapped_token != "Done":
            env_action_handicapped = make_env_action_from_token(handicapped_token, event_handicapped, agent_id=1)
            event_handicapped = env.step(env_action_handicapped, agent_id=1)

    env.close()

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    ensure_dir_for(save_path)
    result_data = logger.to_list()
    output = {
        "task_meta": task_meta,
        "episode_time_sec": elapsed,
        "steps": result_data,
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved two agent trajectory to {save_path} (elapsed {elapsed:.2f}s)")
    return result_data

import os

def run_tasks_for_policy(
    policy: str,
    max_steps: int = 30,
    use_memory: bool = True,
    tasks: List[Dict[str, Any]] | None = None,
    resume: bool = False,
):
    """
    Run a set of tasks for a single policy type.
    policy ∈ {"baseline", "handicapped", "two_agent"}.
    Each task produces its own JSON log file under logs/<policy>/.
    If resume=True, skip tasks that already have a JSON log file.
    """
    if tasks is None:
        tasks = generate_dev_task_list_20()

    for task in tasks:
        task_id = task["task_id"]

        # Determine output folder and file path
        if policy == "baseline":
            save_path = f"logs/baseline/task_{task_id}.json"
        elif policy == "handicapped":
            save_path = f"logs/handicapped/task_{task_id}.json"
        elif policy == "two_agent":
            save_path = f"logs/two_agent/task_{task_id}.json"
        else:
            raise ValueError(f"Unknown policy type: {policy}")

        # --- Resume logic: skip if log already exists ---
        if resume and os.path.exists(save_path):
            print(f"[SKIP] Task {task_id} ({task['scene']}, {task['target_type']}) already completed.")
            continue

        print(f"\n[RUN] {policy.upper()} → Task {task_id}: {task['scene']} → {task['target_type']}")

        # --- Run the appropriate policy ---
        if policy == "baseline":
            run_gemini_agent_baseline(
                max_steps=max_steps,
                use_memory=use_memory,
                save_path=save_path,
                task_meta=task,
            )

        elif policy == "handicapped":
            run_gemini_agent_handicapped(
                max_steps=max_steps,
                use_memory=use_memory,
                save_path=save_path,
                task_meta=task,
            )

        elif policy == "two_agent":
            run_two_agent_llm(
                max_steps=max_steps,
                use_memory=use_memory,
                save_path=save_path,
                task_meta=task,
            )
        else:
            raise ValueError(f"Unknown policy type: {policy}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AI2-THOR navigation experiments.")
    parser.add_argument(
        "--policy",
        type=str,
        default="baseline",
        choices=["baseline", "handicapped", "two_agent"],
        help="Which policy to run over the task set.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=30,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--taskset",
        type=str,
        default="full100",
        choices=["dev20", "full100"],
        help="Which task set to use: full100 or dev20.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks that already have JSON log files.",
    )
    args = parser.parse_args()

    if args.taskset == "dev20":
        tasks = generate_dev_task_list_20()
    elif args.taskset == "full100":
        tasks = generate_task_list_100()
    else:
        raise ValueError(f"Unknown taskset: {args.taskset}")

    print(f"Running policy={args.policy} on {len(tasks)} tasks (taskset={args.taskset}) | resume={args.resume}")
    run_tasks_for_policy(
        policy=args.policy,
        max_steps=args.max_steps,
        use_memory=True,
        tasks=tasks,
        resume=args.resume,
    )


