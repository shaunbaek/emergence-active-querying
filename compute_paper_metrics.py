#!/usr/bin/env python3
"""
Compute 14 specific metrics for paper tables from logs/ directory.

This script analyzes 100 tasks from each policy condition (baseline, handicapped, two_agent)
and outputs metrics for:
  - Table 1: Aggregate Performance (Success Rates and Steps to Success)
  - Table 2: Collaboration Metrics (Message counts conditioned on success/failure)

Success Definition: target_distance <= 1.0 meters at any step.
Steps to Success (STS): Averaged only over successful episodes.
"""

import json
import glob
from pathlib import Path

# Success threshold (meters)
TARGET_SUCCESS_DIST = 1.0


# ============================================================================
# HELPER FUNCTIONS FROM policy_analysis.py
# ============================================================================

def load_episode(path: str):
    """Load episode JSON, handling both old and new formats."""
    with open(path, "r") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "steps" in data:
        steps = data["steps"]
        runtime = data.get("episode_time_sec", None)
        task_meta = data.get("task_meta", None)
    else:
        steps = data
        runtime = None
        task_meta = None
    
    return steps, runtime, task_meta


def episode_success_solo(steps):
    """Check if solo episode (baseline or handicapped) succeeded."""
    for s in steps:
        dist = s.get("target_distance", None)
        if dist is not None and dist <= TARGET_SUCCESS_DIST:
            return True
    return False


def episode_steps_to_success_solo(steps):
    """Return first step index (1-based) where success occurs for solo episode."""
    for s in steps:
        dist = s.get("target_distance", None)
        if dist is not None and dist <= TARGET_SUCCESS_DIST:
            if "step" in s:
                return s["step"] + 1  # 0-based to 1-based
    return None


def episode_success_two_agent_by_role(steps, role: str):
    """Check if specific role (leader or handicapped) succeeded in two-agent episode."""
    for s in steps:
        if s.get("role") != role:
            continue
        dist = s.get("target_distance", None)
        if dist is not None and dist <= TARGET_SUCCESS_DIST:
            return True
    return False


def episode_steps_to_success_two_agent(steps, role: str):
    """Return first step index (1-based) where role achieves success."""
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


def classify_message_speaker(msg: str) -> str:
    """Classify message speaker from conversation log prefix."""
    u = msg.upper()
    if u.startswith("LEADER:"):
        return "leader"
    if u.startswith("HANDICAPPED:"):
        return "handicapped"
    return "unknown"


def is_instruction_message(text: str) -> bool:
    """Heuristic: detect leader instruction messages."""
    t = text.lower()
    keywords = [
        "go", "move", "walk", "turn", "rotate",
        "follow", "you should", "please", "i will", "i'll",
        "stay", "wait", "check", "look", "search",
        "head towards", "keep going", "stop there",
    ]
    return any(k in t for k in keywords)


def is_help_request_message(text: str) -> bool:
    """Heuristic: detect handicapped help/uncertainty messages."""
    t = text.lower()
    keywords = [
        "i can't", "i cannot", "i don't see", "don't see",
        "where is", "where's", "what should i do",
        "i'm not sure", "i am not sure",
        "help", "confused", "lost", "no idea",
        "?",
    ]
    return any(k in t for k in keywords)


def count_messages_two_agent(steps):
    """
    Count leader instructions and handicapped help requests.
    Uses both per-step 'message' fields and conversation_log.
    """
    convo_seen = set()
    leader_instructions = 0
    handicapped_help_reqs = 0
    
    for s in steps:
        role = s.get("role", None)
        msg = s.get("message", "")
        if msg:
            t = msg.strip()
            if role == "leader" and is_instruction_message(t):
                leader_instructions += 1
            elif role == "handicapped" and is_help_request_message(t):
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
                
                if speaker == "leader" and is_instruction_message(content):
                    leader_instructions += 1
                elif speaker == "handicapped" and is_help_request_message(content):
                    handicapped_help_reqs += 1
    
    return leader_instructions, handicapped_help_reqs


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_table1_metrics():
    """
    Compute Table 1 metrics: Success Rates and Avg Steps to Success.
    
    Returns dict with 8 values:
      - baseline_sr, baseline_avg_sts
      - handicapped_sr, handicapped_avg_sts
      - two_agent_leader_sr, two_agent_leader_avg_sts
      - two_agent_handicapped_sr, two_agent_handicapped_avg_sts
    """
    metrics = {}
    
    # --- BASELINE ---
    baseline_files = sorted(glob.glob("logs/baseline/task_*.json"))
    baseline_successes = []
    baseline_sts = []
    
    for path in baseline_files:
        steps, _, _ = load_episode(path)
        success = episode_success_solo(steps)
        baseline_successes.append(success)
        if success:
            sts = episode_steps_to_success_solo(steps)
            if sts is not None:
                baseline_sts.append(sts)
    
    n_baseline = len(baseline_successes)
    metrics["baseline_sr"] = sum(baseline_successes) / n_baseline if n_baseline > 0 else 0.0
    metrics["baseline_avg_sts"] = sum(baseline_sts) / len(baseline_sts) if baseline_sts else 0.0
    
    # --- HANDICAPPED ---
    handicapped_files = sorted(glob.glob("logs/handicapped/task_*.json"))
    handicapped_successes = []
    handicapped_sts = []
    
    for path in handicapped_files:
        steps, _, _ = load_episode(path)
        success = episode_success_solo(steps)
        handicapped_successes.append(success)
        if success:
            sts = episode_steps_to_success_solo(steps)
            if sts is not None:
                handicapped_sts.append(sts)
    
    n_handicapped = len(handicapped_successes)
    metrics["handicapped_sr"] = sum(handicapped_successes) / n_handicapped if n_handicapped > 0 else 0.0
    metrics["handicapped_avg_sts"] = sum(handicapped_sts) / len(handicapped_sts) if handicapped_sts else 0.0
    
    # --- TWO-AGENT (LEADER) ---
    two_agent_files = sorted(glob.glob("logs/two_agent/task_*.json"))
    leader_successes = []
    leader_sts = []
    
    for path in two_agent_files:
        steps, _, _ = load_episode(path)
        success = episode_success_two_agent_by_role(steps, "leader")
        leader_successes.append(success)
        if success:
            sts = episode_steps_to_success_two_agent(steps, "leader")
            if sts is not None:
                leader_sts.append(sts)
    
    n_two_agent = len(two_agent_files)
    metrics["two_agent_leader_sr"] = sum(leader_successes) / n_two_agent if n_two_agent > 0 else 0.0
    metrics["two_agent_leader_avg_sts"] = sum(leader_sts) / len(leader_sts) if leader_sts else 0.0
    
    # --- TWO-AGENT (HANDICAPPED) ---
    handicapped_two_successes = []
    handicapped_two_sts = []
    
    for path in two_agent_files:
        steps, _, _ = load_episode(path)
        success = episode_success_two_agent_by_role(steps, "handicapped")
        handicapped_two_successes.append(success)
        if success:
            sts = episode_steps_to_success_two_agent(steps, "handicapped")
            if sts is not None:
                handicapped_two_sts.append(sts)
    
    metrics["two_agent_handicapped_sr"] = sum(handicapped_two_successes) / n_two_agent if n_two_agent > 0 else 0.0
    metrics["two_agent_handicapped_avg_sts"] = sum(handicapped_two_sts) / len(handicapped_two_sts) if handicapped_two_sts else 0.0
    
    return metrics


def compute_table2_metrics():
    """
    Compute Table 2 metrics: Collaboration/Communication metrics from two_agent logs.
    
    Returns dict with 6 values:
      - avg_leader_instr_all
      - avg_handicapped_help_all
      - avg_leader_instr_success
      - avg_handicapped_help_success
      - avg_leader_instr_failure
      - avg_handicapped_help_failure
    """
    metrics = {}
    
    two_agent_files = sorted(glob.glob("logs/two_agent/task_*.json"))
    
    all_leader_instr = []
    all_handicapped_help = []
    
    success_leader_instr = []
    success_handicapped_help = []
    
    failure_leader_instr = []
    failure_handicapped_help = []
    
    for path in two_agent_files:
        steps, _, _ = load_episode(path)
        
        # Check handicapped success
        handicapped_success = episode_success_two_agent_by_role(steps, "handicapped")
        
        # Count messages
        leader_instr, handicapped_help = count_messages_two_agent(steps)
        
        # Overall counts
        all_leader_instr.append(leader_instr)
        all_handicapped_help.append(handicapped_help)
        
        # Conditional counts
        if handicapped_success:
            success_leader_instr.append(leader_instr)
            success_handicapped_help.append(handicapped_help)
        else:
            failure_leader_instr.append(leader_instr)
            failure_handicapped_help.append(handicapped_help)
    
    # Compute averages
    n_all = len(all_leader_instr)
    metrics["avg_leader_instr_all"] = sum(all_leader_instr) / n_all if n_all > 0 else 0.0
    metrics["avg_handicapped_help_all"] = sum(all_handicapped_help) / n_all if n_all > 0 else 0.0
    
    n_success = len(success_leader_instr)
    metrics["avg_leader_instr_success"] = sum(success_leader_instr) / n_success if n_success > 0 else 0.0
    metrics["avg_handicapped_help_success"] = sum(success_handicapped_help) / n_success if n_success > 0 else 0.0
    
    n_failure = len(failure_leader_instr)
    metrics["avg_leader_instr_failure"] = sum(failure_leader_instr) / n_failure if n_failure > 0 else 0.0
    metrics["avg_handicapped_help_failure"] = sum(failure_handicapped_help) / n_failure if n_failure > 0 else 0.0
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("COMPUTING 14 PAPER METRICS FROM LOGS")
    print("=" * 80)
    print()
    
    # Check if log directories exist
    import os
    required_dirs = ["logs/baseline", "logs/handicapped", "logs/two_agent"]
    for d in required_dirs:
        if not os.path.exists(d):
            print(f"WARNING: Directory '{d}' not found!")
    print()
    
    # Compute Table 1 metrics
    print("Computing Table 1 (Aggregate Performance) metrics...")
    table1 = compute_table1_metrics()
    print("✓ Table 1 metrics computed")
    print()
    
    # Compute Table 2 metrics
    print("Computing Table 2 (Collaboration Metrics) metrics...")
    table2 = compute_table2_metrics()
    print("✓ Table 2 metrics computed")
    print()
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    print("=" * 80)
    print("TABLE 1: AGGREGATE PERFORMANCE")
    print("=" * 80)
    print()
    print(f"{'Policy':<30} {'Success Rate':<15} {'Avg. STS':<15}")
    print("-" * 60)
    print(f"{'Baseline (Solo)':<30} {table1['baseline_sr']:.3f} ({table1['baseline_sr']*100:.1f}%){'':<3} {table1['baseline_avg_sts']:.2f}")
    print(f"{'Handicapped (Solo)':<30} {table1['handicapped_sr']:.3f} ({table1['handicapped_sr']*100:.1f}%){'':<3} {table1['handicapped_avg_sts']:.2f}")
    print(f"{'Two-Agent (Leader)':<30} {table1['two_agent_leader_sr']:.3f} ({table1['two_agent_leader_sr']*100:.1f}%){'':<3} {table1['two_agent_leader_avg_sts']:.2f}")
    print(f"{'Two-Agent (Handicapped)':<30} {table1['two_agent_handicapped_sr']:.3f} ({table1['two_agent_handicapped_sr']*100:.1f}%){'':<3} {table1['two_agent_handicapped_avg_sts']:.2f}")
    print()
    
    print("=" * 80)
    print("TABLE 2: COLLABORATION METRICS (Two-Agent Only)")
    print("=" * 80)
    print()
    print(f"{'Metric':<50} {'Average':<15}")
    print("-" * 65)
    print(f"{'Avg. Leader Instructions (All Episodes)':<50} {table2['avg_leader_instr_all']:.2f}")
    print(f"{'Avg. Handicapped Help Requests (All Episodes)':<50} {table2['avg_handicapped_help_all']:.2f}")
    print()
    print(f"{'Avg. Leader Instructions (On Success)':<50} {table2['avg_leader_instr_success']:.2f}")
    print(f"{'Avg. Handicapped Help Requests (On Success)':<50} {table2['avg_handicapped_help_success']:.2f}")
    print()
    print(f"{'Avg. Leader Instructions (On Failure)':<50} {table2['avg_leader_instr_failure']:.2f}")
    print(f"{'Avg. Handicapped Help Requests (On Failure)':<50} {table2['avg_handicapped_help_failure']:.2f}")
    print()
    
    # ========================================================================
    # SUMMARY FOR COPY-PASTE
    # ========================================================================
    
    print("=" * 80)
    print("SUMMARY: 14 VALUES FOR PAPER")
    print("=" * 80)
    print()
    print("TABLE 1 (8 values):")
    print(f"  1. Baseline SR:                    {table1['baseline_sr']:.3f}")
    print(f"  2. Baseline Avg. STS:              {table1['baseline_avg_sts']:.2f}")
    print(f"  3. Handicapped SR:                 {table1['handicapped_sr']:.3f}")
    print(f"  4. Handicapped Avg. STS:           {table1['handicapped_avg_sts']:.2f}")
    print(f"  5. Two-Agent Leader SR:            {table1['two_agent_leader_sr']:.3f}")
    print(f"  6. Two-Agent Leader Avg. STS:      {table1['two_agent_leader_avg_sts']:.2f}")
    print(f"  7. Two-Agent Handicapped SR:       {table1['two_agent_handicapped_sr']:.3f}")
    print(f"  8. Two-Agent Handicapped Avg. STS: {table1['two_agent_handicapped_avg_sts']:.2f}")
    print()
    print("TABLE 2 (6 values):")
    print(f"  9.  Avg. Leader Instructions (All):     {table2['avg_leader_instr_all']:.2f}")
    print(f"  10. Avg. Handicapped Help Reqs (All):   {table2['avg_handicapped_help_all']:.2f}")
    print(f"  11. Avg. Leader Instructions (Success): {table2['avg_leader_instr_success']:.2f}")
    print(f"  12. Avg. Handicapped Help Reqs (Success):{table2['avg_handicapped_help_success']:.2f}")
    print(f"  13. Avg. Leader Instructions (Failure): {table2['avg_leader_instr_failure']:.2f}")
    print(f"  14. Avg. Handicapped Help Reqs (Failure):{table2['avg_handicapped_help_failure']:.2f}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
