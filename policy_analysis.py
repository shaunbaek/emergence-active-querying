import json
import glob
from pathlib import Path

TARGET_SUCCESS_DIST = 1.0  # meters


def load_episode(path: str):
    """
    Load a logged episode.

    Supports both:
      - old style: a plain list of step dicts
      - new wrapped style: { "task_meta": ..., "episode_time_sec": ..., "steps": [...] }

    Returns: (steps, episode_time_sec or None, task_meta or None)
    """
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "steps" in data:
        steps = data["steps"]
        runtime = data.get("episode_time_sec", None)
        task_meta = data.get("task_meta", None)
    else:
        # backward compatibility
        steps = data
        runtime = None
        task_meta = None

    return steps, runtime, task_meta



# ------------------ SUCCESS METRICS ------------------ #

def episode_success_generic(steps):
    """
    Generic success: success if at any step target_distance <= threshold.
    Used for both baseline and handicapped solo episodes.
    """
    for s in steps:
        dist = s.get("target_distance", None)
        if dist is not None and dist <= TARGET_SUCCESS_DIST:
            return True
    return False


def episode_success_two_agent(steps):
    """
    For a two-agent episode:
    Compute per-role success (leader vs handicapped).
    Success if that role ever gets target_distance <= threshold.
    """
    leader_success = False
    handicapped_success = False

    for s in steps:
        role = s.get("role", None)
        dist = s.get("target_distance", None)
        if dist is None:
            continue
        if dist <= TARGET_SUCCESS_DIST:
            if role == "leader":
                leader_success = True
            elif role == "handicapped":
                handicapped_success = True

    return leader_success, handicapped_success


def episode_steps_to_success_solo(steps):
    """
    For a solo episode (baseline or handicapped):
    Return the first step index (1-based) at which target_distance <= threshold.
    If never succeeds, return None.
    """
    for s in steps:
        dist = s.get("target_distance", None)
        if dist is not None and dist <= TARGET_SUCCESS_DIST:
            # steps are usually logged with a 'step' field; prefer that
            if "step" in s:
                return s["step"] + 1  # convert 0-based to 1-based count
            # fallback: use list index if no 'step' key
            # (only safe if caller passes enumerate index instead, but kept for robustness)
    return None


def episode_steps_to_success_two_agent(steps, role: str):
    """
    For a two-agent episode:
    Return the first step index (1-based) for the given role
    (either 'leader' or 'handicapped') at which target_distance <= threshold.
    If that role never achieves success, return None.
    """
    best_step = None
    for s in steps:
        if s.get("role") != role:
            continue
        dist = s.get("target_distance", None)
        if dist is None or dist > TARGET_SUCCESS_DIST:
            continue
        if "step" in s:
            step_idx = s["step"] + 1  # 1-based
        else:
            continue
        if best_step is None or step_idx < best_step:
            best_step = step_idx
    return best_step


# ------------------ MESSAGE ANALYSIS (TWO-AGENT ONLY) ------------------ #

def classify_message_speaker(msg: str) -> str:
    """
    Classify message speaker by prefix in conversation log:
    'LEADER: ...' or 'HANDICAPPED: ...' (from conversation_log entries).
    Returns 'leader', 'handicapped', or 'unknown'.
    """
    u = msg.upper()
    if u.startswith("LEADER:"):
        return "leader"
    if u.startswith("HANDICAPPED:"):
        return "handicapped"
    return "unknown"


def is_instruction_message(text: str) -> bool:
    """
    Heuristic: leader 'instruction' messages.
    Looks for imperative-like tokens and directive phrases.
    """
    t = text.lower()
    keywords = [
        "go", "move", "walk", "turn", "rotate",
        "follow", "you should", "please", "i will", "i'll",
        "stay", "wait", "check", "look", "search",
        "head towards", "keep going", "stop there",
    ]
    return any(k in t for k in keywords)


def is_help_request_message(text: str) -> bool:
    """
    Heuristic: handicapped 'help / uncertainty' messages.
    """
    t = text.lower()
    keywords = [
        "i can't", "i cannot", "i don't see", "don't see",
        "where is", "where's", "what should i do",
        "i'm not sure", "i am not sure",
        "help", "confused", "lost", "no idea",
        "?",  # questions are often help/clarification
    ]
    return any(k in t for k in keywords)


def is_clarifying_question_message(text: str) -> bool:
    """
    Heuristic: helper's clarifying questions directed at the impaired agent.
    Examples: "What do you see?", "Do you see the microwave?", "Can you see anything?"
    """
    t = text.lower()
    keywords = [
        "what do you see",
        "what can you see",
        "do you see",
        "can you see",
        "what is near you",
        "what objects are near you",
        "where are you",
        "are you close",
        "?",
    ]
    return any(k in t for k in keywords)


def episode_message_stats_two_agent(steps):
    """
    For a two-agent episode:
    Compute message statistics:
      - total leader/helper messages
      - total handicapped messages
      - leader instructions (counts)
      - leader clarifying questions (counts)
      - handicapped help/uncertainty messages (counts)
    We use both per-step 'message' fields and conversation_log snapshots.
    """
    convo_seen = set()

    total_leader_msgs = 0
    total_handicapped_msgs = 0

    leader_instructions = 0
    leader_clarifying = 0
    handicapped_help_reqs = 0

    for s in steps:
        role = s.get("role", None)
        msg = s.get("message", "")
        if msg:
            t = msg.strip()
            if role == "leader":
                total_leader_msgs += 1
                if is_instruction_message(t):
                    leader_instructions += 1
                if is_clarifying_question_message(t):
                    leader_clarifying += 1
            elif role == "handicapped":
                total_handicapped_msgs += 1
                if is_help_request_message(t):
                    handicapped_help_reqs += 1

        convo = s.get("conversation_log", None)
        if convo:
            for line in convo:
                if line in convo_seen:
                    continue
                convo_seen.add(line)
                speaker = classify_message_speaker(line)
                if ":" in line:
                    _, content = line.split(":", 1)
                    content = content.strip()
                else:
                    content = line.strip()

                if speaker == "leader":
                    total_leader_msgs += 1
                    if is_instruction_message(content):
                        leader_instructions += 1
                    if is_clarifying_question_message(content):
                        leader_clarifying += 1
                elif speaker == "handicapped":
                    total_handicapped_msgs += 1
                    if is_help_request_message(content):
                        handicapped_help_reqs += 1

    return {
        "leader_msgs": total_leader_msgs,
        "handicapped_msgs": total_handicapped_msgs,
        "leader_instructions": leader_instructions,
        "leader_clarifying": leader_clarifying,
        "handicapped_help_requests": handicapped_help_reqs,
    }


# ------------------ SUMMARY FUNCTIONS ------------------ #

def summarize_condition_baseline(pattern: str):
    summaries = []
    for path in sorted(glob.glob(pattern)):
        steps, runtime, task_meta = load_episode(path)
        succ = episode_success_generic(steps)
        steps_to_succ = episode_steps_to_success_solo(steps)

        fname = Path(path).name
        # expects filename like task_1.json
        task_id = int(fname.split("_")[1].split(".")[0])

        summaries.append(
            {
                "task_id": task_id,
                "condition": "baseline",
                "file": fname,
                "success": succ,
                "steps_to_success": steps_to_succ,
                "runtime_sec": runtime,
            }
        )
    return summaries




def summarize_condition_handicapped(pattern: str):
    summaries = []
    for path in sorted(glob.glob(pattern)):
        steps, runtime, task_meta = load_episode(path)
        succ = episode_success_generic(steps)
        steps_to_succ = episode_steps_to_success_solo(steps)

        fname = Path(path).name
        task_id = int(fname.split("_")[1].split(".")[0])

        summaries.append(
            {
                "task_id": task_id,
                "condition": "handicapped",
                "file": fname,
                "success": succ,
                "steps_to_success": steps_to_succ,
                "runtime_sec": runtime,
            }
        )
    return summaries


def summarize_condition_two_agent(pattern: str):
    summaries = []
    for path in sorted(glob.glob(pattern)):
        steps, runtime, task_meta = load_episode(path)
        leader_succ, handicapped_succ = episode_success_two_agent(steps)

        leader_steps_to_succ = episode_steps_to_success_two_agent(steps, role="leader")
        handicapped_steps_to_succ = episode_steps_to_success_two_agent(steps, role="handicapped")

        msg_stats = episode_message_stats_two_agent(steps)

        fname = Path(path).name
        task_id = int(fname.split("_")[1].split(".")[0])

        summaries.append(
            {
                "task_id": task_id,
                "condition": "two_agent",
                "file": fname,

                "leader_success": leader_succ,
                "handicapped_success": handicapped_succ,
                "leader_steps_to_success": leader_steps_to_succ,
                "handicapped_steps_to_success": handicapped_steps_to_succ,

                "leader_msgs": msg_stats["leader_msgs"],
                "handicapped_msgs": msg_stats["handicapped_msgs"],
                "leader_instructions": msg_stats["leader_instructions"],
                "leader_clarifying": msg_stats["leader_clarifying"],
                "handicapped_help_requests": msg_stats["handicapped_help_requests"],

                "runtime_sec": runtime,
            }
        )
    return summaries



def print_taskwise_comparison(baseline_summaries, handicapped_summaries, two_agent_summaries):
    """
    Join baseline, handicapped, and two-agent results on task_id
    and print a compact comparison table.
    """
    base_by_task = {s["task_id"]: s for s in baseline_summaries}
    handi_by_task = {s["task_id"]: s for s in handicapped_summaries}
    two_by_task = {s["task_id"]: s for s in two_agent_summaries}

    header = (
        f"{'Task':<6} "
        f"{'BaseSucc':<9} "
        f"{'HandiSolo':<10} "
        f"{'2A_Leader':<10} "
        f"{'2A_Handi':<10} "
        f"{'LeadInstr':<9} "
        f"{'HandiHelp':<9}"
    )
    print(header)
    print("-" * len(header))

    task_ids = sorted(
        set(base_by_task.keys()) &
        set(handi_by_task.keys()) &
        set(two_by_task.keys())
    )

    for tid in task_ids:
        b = base_by_task[tid]
        h = handi_by_task[tid]
        t = two_by_task[tid]
        print(
            f"{tid:<6} "
            f"{str(b['success']):<9} "
            f"{str(h['success']):<10} "
            f"{str(t['leader_success']):<10} "
            f"{str(t['handicapped_success']):<10} "
            f"{t['leader_instructions']:<9} "
            f"{t['handicapped_help_requests']:<9}"
        )

def print_taskwise_comparison_all(baseline_summaries, handicapped_summaries, two_agent_summaries):
    """
    Print a readable task-by-task comparison across all three policy types:
      - Baseline (solo)
      - Handicapped (solo)
      - Two-Agent (leader & handicapped)

    Each row shows success, steps to success, and runtime for each policy.
    """
    # Convert summaries to dicts keyed by task_id for easy joining
    base_by_task = {s["task_id"]: s for s in baseline_summaries}
    handi_by_task = {s["task_id"]: s for s in handicapped_summaries}
    two_by_task = {s["task_id"]: s for s in two_agent_summaries}

    # Header
    header = (
        f"{'Task':<6} "
        f"{'Base_Success':<13} {'Base_Steps':<12} {'Base_Runtime':<14} "
        f"{'Handi_Success':<14} {'Handi_Steps':<13} {'Handi_Runtime':<14} "
        f"{'2A_Leader_Success':<18} {'2A_Leader_Steps':<16} {'2A_Handi_Success':<18} {'2A_Handi_Steps':<16} {'2A_Runtime':<12}"
    )
    print(header)
    print("-" * len(header))

    # Collect all task IDs that appear in any set
    all_task_ids = sorted(
        set(base_by_task.keys()) | set(handi_by_task.keys()) | set(two_by_task.keys())
    )

    for tid in all_task_ids:
        b = base_by_task.get(tid, {})
        h = handi_by_task.get(tid, {})
        t = two_by_task.get(tid, {})

        print(
            f"{tid:<6} "
            f"{str(b.get('success', 'NA')):<13} {str(b.get('steps_to_success', 'NA')):<12} {str(round(b.get('runtime_sec', 0), 2))+'s':<14} "
            f"{str(h.get('success', 'NA')):<14} {str(h.get('steps_to_success', 'NA')):<13} {str(round(h.get('runtime_sec', 0), 2))+'s':<14} "
            f"{str(t.get('leader_success', 'NA')):<18} {str(t.get('leader_steps_to_success', 'NA')):<16} "
            f"{str(t.get('handicapped_success', 'NA')):<18} {str(t.get('handicapped_steps_to_success', 'NA')):<16} "
            f"{str(round(t.get('runtime_sec', 0), 2))+'s':<12}"
        )

def compute_aggregates(baseline_summaries, handicapped_summaries, two_agent_summaries):
    """
    Compute overall success rates:
      - baseline solo success rate
      - handicapped solo success rate
      - two-agent leader success rate
      - two-agent handicapped success rate
      - fraction of tasks where leader succeeded but handicapped did not

    Plus:
      - average leader messages & instructions (overall, and conditioned on handicapped success)
      - average handicapped messages & help-requests (overall, and conditioned on handicapped success)
    """
    n_base = len(baseline_summaries)
    n_hand = len(handicapped_summaries)
    n_two = len(two_agent_summaries)

    base_successes = sum(1 for s in baseline_summaries if s["success"])
    hand_successes = sum(1 for s in handicapped_summaries if s["success"])

    base_success_rate = base_successes / n_base if n_base > 0 else 0.0
    hand_success_rate = hand_successes / n_hand if n_hand > 0 else 0.0

    leader_successes = sum(1 for s in two_agent_summaries if s["leader_success"])
    handicapped_successes = sum(1 for s in two_agent_summaries if s["handicapped_success"])

    leader_success_rate = leader_successes / n_two if n_two > 0 else 0.0
    two_hand_success_rate = handicapped_successes / n_two if n_two > 0 else 0.0

    # tasks where leader success = True and handicapped_success = False
    leader_only = sum(
        1
        for s in two_agent_summaries
        if s["leader_success"] and not s["handicapped_success"]
    )
    leader_only_fraction = leader_only / n_two if n_two > 0 else 0.0

    # ---- Message aggregates ---- #
    total_leader_msgs = sum(s["leader_msgs"] for s in two_agent_summaries)
    total_handicapped_msgs = sum(s["handicapped_msgs"] for s in two_agent_summaries)
    total_leader_instr = sum(s["leader_instructions"] for s in two_agent_summaries)
    total_hand_help = sum(s["handicapped_help_requests"] for s in two_agent_summaries)
    total_leader_clarifying = sum(s["leader_clarifying"] for s in two_agent_summaries)

    avg_leader_msgs = total_leader_msgs / n_two if n_two > 0 else 0.0
    avg_hand_msgs = total_handicapped_msgs / n_two if n_two > 0 else 0.0
    avg_leader_instr = total_leader_instr / n_two if n_two > 0 else 0.0
    avg_hand_help = total_hand_help / n_two if n_two > 0 else 0.0
    avg_leader_clarifying = total_leader_clarifying / n_two if n_two > 0 else 0.0

    # conditioned on handicapped success / failure (two-agent setting)
    succ_eps = [s for s in two_agent_summaries if s["handicapped_success"]]
    fail_eps = [s for s in two_agent_summaries if not s["handicapped_success"]]

    def safe_avg(seq, key):
        if not seq:
            return 0.0
        return sum(s[key] for s in seq) / len(seq)

    avg_leader_msgs_when_hand_succ = safe_avg(succ_eps, "leader_msgs")
    avg_leader_msgs_when_hand_fail = safe_avg(fail_eps, "leader_msgs")

    avg_leader_instr_when_hand_succ = safe_avg(succ_eps, "leader_instructions")
    avg_leader_instr_when_hand_fail = safe_avg(fail_eps, "leader_instructions")

    avg_hand_msgs_when_succ = safe_avg(succ_eps, "handicapped_msgs")
    avg_hand_msgs_when_fail = safe_avg(fail_eps, "handicapped_msgs")

    avg_hand_help_when_succ = safe_avg(succ_eps, "handicapped_help_requests")
    avg_hand_help_when_fail = safe_avg(fail_eps, "handicapped_help_requests")

    avg_leader_clar_when_hand_succ = safe_avg(succ_eps, "leader_clarifying")
    avg_leader_clar_when_hand_fail = safe_avg(fail_eps, "leader_clarifying")

    print("\n=== Aggregate success rates ===")
    print(f"Baseline solo success rate:        {base_success_rate:.3f} ({base_successes}/{n_base})")
    print(f"Handicapped solo success rate:     {hand_success_rate:.3f} ({hand_successes}/{n_hand})")
    print(f"Two-agent leader success rate:     {leader_success_rate:.3f} ({leader_successes}/{n_two})")
    print(f"Two-agent handicapped success:     {two_hand_success_rate:.3f} ({handicapped_successes}/{n_two})")
    print(f"Leader-only success fraction:      {leader_only_fraction:.3f} ({leader_only}/{n_two})")

    print("\n=== Message statistics (two-agent, overall) ===")
    print(f"Avg leader messages per episode:          {avg_leader_msgs:.2f}")
    print(f"Avg handicapped messages per episode:     {avg_hand_msgs:.2f}")
    print(f"Avg leader instructions per episode:      {avg_leader_instr:.2f}")
    print(f"Avg leader clarifying questions per ep.:  {avg_leader_clarifying:.2f}")
    print(f"Avg handicapped help-requests per episode:{avg_hand_help:.2f}")

    print("\n=== Message statistics conditioned on two-agent handicapped SUCCESS ===")
    print(f"Avg leader messages (handicapped success):          {avg_leader_msgs_when_hand_succ:.2f}")
    print(f"Avg leader instructions (handicapped success):      {avg_leader_instr_when_hand_succ:.2f}")
    print(f"Avg leader clarifying qs (handicapped success):     {avg_leader_clar_when_hand_succ:.2f}")
    print(f"Avg handicapped messages (handicapped success):     {avg_hand_msgs_when_succ:.2f}")
    print(f"Avg handicapped help-requests (handicapped success):{avg_hand_help_when_succ:.2f}")

    print("\n=== Message statistics conditioned on two-agent handicapped FAILURE ===")
    print(f"Avg leader messages (handicapped failure):          {avg_leader_msgs_when_hand_fail:.2f}")
    print(f"Avg leader instructions (handicapped failure):      {avg_leader_instr_when_hand_fail:.2f}")
    print(f"Avg leader clarifying qs (handicapped failure):     {avg_leader_clar_when_hand_fail:.2f}")
    print(f"Avg handicapped messages (handicapped failure):     {avg_hand_msgs_when_fail:.2f}")
    print(f"Avg handicapped help-requests (handicapped failure):{avg_hand_help_when_fail:.2f}")

        # ---- Runtime aggregates ---- #
    def avg_runtime(summaries):
        valid = [s["runtime_sec"] for s in summaries if s.get("runtime_sec") is not None]
        return sum(valid) / len(valid) if valid else 0.0

    def min_runtime(summaries):
        valid = [s["runtime_sec"] for s in summaries if s.get("runtime_sec") is not None]
        return min(valid) if valid else 0.0

    def max_runtime(summaries):
        valid = [s["runtime_sec"] for s in summaries if s.get("runtime_sec") is not None]
        return max(valid) if valid else 0.0

    avg_rt_base = avg_runtime(baseline_summaries)
    avg_rt_hand = avg_runtime(handicapped_summaries)
    avg_rt_two = avg_runtime(two_agent_summaries)

    print("\n=== Runtime statistics (seconds) ===")
    print(f"Baseline average runtime:    {avg_rt_base:.2f} s "
          f"(min={min_runtime(baseline_summaries):.2f}, max={max_runtime(baseline_summaries):.2f})")
    print(f"Handicapped average runtime: {avg_rt_hand:.2f} s "
          f"(min={min_runtime(handicapped_summaries):.2f}, max={max_runtime(handicapped_summaries):.2f})")
    print(f"Two-agent average runtime:   {avg_rt_two:.2f} s "
          f"(min={min_runtime(two_agent_summaries):.2f}, max={max_runtime(two_agent_summaries):.2f})")

def compute_aggregates_all(baseline_summaries, handicapped_summaries, two_agent_summaries):
    """
    Compute and print high-level summary statistics for all 3 policy conditions:
      - Baseline (single agent, full visibility)
      - Handicapped (single agent, limited visibility)
      - Two-Agent (leader + handicapped cooperative pair)
    """

    # --- Helper for averages ---
    def safe_avg(seq, key):
        vals = [s[key] for s in seq if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    # --- Baseline ---
    n_base = len(baseline_summaries)
    base_success_rate = sum(1 for s in baseline_summaries if s["success"]) / n_base if n_base else 0.0
    base_avg_steps = safe_avg(baseline_summaries, "steps_to_success")
    base_avg_runtime = safe_avg(baseline_summaries, "runtime_sec")

    # --- Handicapped ---
    n_hand = len(handicapped_summaries)
    hand_success_rate = sum(1 for s in handicapped_summaries if s["success"]) / n_hand if n_hand else 0.0
    hand_avg_steps = safe_avg(handicapped_summaries, "steps_to_success")
    hand_avg_runtime = safe_avg(handicapped_summaries, "runtime_sec")

    # --- Two-Agent ---
    n_two = len(two_agent_summaries)
    leader_success_rate = sum(1 for s in two_agent_summaries if s["leader_success"]) / n_two if n_two else 0.0
    handicapped_success_rate = sum(1 for s in two_agent_summaries if s["handicapped_success"]) / n_two if n_two else 0.0

    leader_avg_steps = safe_avg(two_agent_summaries, "leader_steps_to_success")
    handicapped_avg_steps = safe_avg(two_agent_summaries, "handicapped_steps_to_success")
    two_avg_runtime = safe_avg(two_agent_summaries, "runtime_sec")

    # --- Print Summary ---
    print("\n=== Aggregate Policy Performance Summary ===")
    print(f"{'Condition':<18} {'Success_Rate':<15} {'Avg_Steps_To_Success':<25} {'Avg_Runtime(s)':<15}")
    print("-" * 73)

    print(f"{'Baseline (solo)':<18} {base_success_rate:.3f}{'':<6} {base_avg_steps:<25.2f} {base_avg_runtime:<15.2f}")
    print(f"{'Handicapped (solo)':<18} {hand_success_rate:.3f}{'':<6} {hand_avg_steps:<25.2f} {hand_avg_runtime:<15.2f}")
    print(f"{'Two-Agent (leader)':<18} {leader_success_rate:.3f}{'':<6} {leader_avg_steps:<25.2f} {two_avg_runtime:<15.2f}")
    print(f"{'Two-Agent (handicap)':<18} {handicapped_success_rate:.3f}{'':<6} {handicapped_avg_steps:<25.2f} {two_avg_runtime:<15.2f}")

# ------------------ MAIN ------------------ #

def main():
    baseline_summaries = summarize_condition_baseline("logs/baseline/task_*.json")
    handicapped_summaries = summarize_condition_handicapped("logs/handicapped/task_*.json")
    two_agent_summaries = summarize_condition_two_agent("logs/two_agent/task_*.json")

    if not (baseline_summaries or handicapped_summaries or two_agent_summaries):
        print("No episodes found. Make sure you've run loop.py with run_experiment_on_tasks.")
        return

    print("\n=== Task-wise comparison (Baseline vs Handicapped vs Two-Agent) ===")
    print_taskwise_comparison_all(baseline_summaries, handicapped_summaries, two_agent_summaries)

    compute_aggregates_all(baseline_summaries, handicapped_summaries, two_agent_summaries)


if __name__ == "__main__":
    main()
