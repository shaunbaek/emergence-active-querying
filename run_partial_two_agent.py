#!/usr/bin/env python3
"""
Run a partial subset of tasks for the two_agent policy.

Usage:
    python3 run_partial_two_agent.py --start 1 --end 20 --max_steps 30
    python3 run_partial_two_agent.py --tasks 1,5,10,15 --max_steps 30
    python3 run_partial_two_agent.py --start 50 --end 100 --max_steps 30 --resume
"""

import argparse
import json
import os
from tasks_utils import generate_task_list_100
from loop import run_two_agent_llm


def parse_task_ids(task_string):
    """Parse comma-separated task IDs into a list of integers."""
    return [int(tid.strip()) for tid in task_string.split(',')]


def get_task_range(start, end, tasks_list):
    """Get tasks within a specific range."""
    return [task for task in tasks_list if start <= task['task_id'] <= end]


def get_specific_tasks(task_ids, tasks_list):
    """Get specific tasks by their IDs."""
    task_dict = {task['task_id']: task for task in tasks_list}
    return [task_dict[tid] for tid in task_ids if tid in task_dict]


def should_skip_task(task_id, resume):
    """Check if task should be skipped (already completed)."""
    if not resume:
        return False
    log_path = f"logs/two_agent/task_{task_id}.json"
    return os.path.exists(log_path)


def main():
    parser = argparse.ArgumentParser(
        description='Run partial tasks for two_agent policy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run tasks 1-20
  python3 run_partial_two_agent.py --start 1 --end 20 --max_steps 30
  
  # Run specific tasks
  python3 run_partial_two_agent.py --tasks 1,5,10,15,20 --max_steps 30
  
  # Resume interrupted run (skip completed tasks)
  python3 run_partial_two_agent.py --start 1 --end 50 --max_steps 30 --resume
  
  # Run last 10 tasks
  python3 run_partial_two_agent.py --start 91 --end 100 --max_steps 30
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--start',
        type=int,
        help='Start task ID (use with --end for range)'
    )
    group.add_argument(
        '--tasks',
        type=str,
        help='Comma-separated list of task IDs (e.g., "1,5,10,15")'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        help='End task ID (use with --start for range)'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=50,
        help='Maximum steps per episode (default: 50)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip tasks that already have log files'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start is not None and args.end is None:
        parser.error('--start requires --end')
    if args.end is not None and args.start is None:
        parser.error('--end requires --start')
    
    # Generate full task list
    print("Loading tasks from generate_task_list_100()...")
    all_tasks = generate_task_list_100()
    
    # Select tasks based on arguments
    if args.tasks:
        task_ids = parse_task_ids(args.tasks)
        selected_tasks = get_specific_tasks(task_ids, all_tasks)
        print(f"Selected {len(selected_tasks)} specific tasks: {task_ids}")
    else:
        selected_tasks = get_task_range(args.start, args.end, all_tasks)
        print(f"Selected tasks {args.start}-{args.end}: {len(selected_tasks)} tasks")
    
    if not selected_tasks:
        print("No tasks selected. Exiting.")
        return
    
    # Show task distribution
    room_counts = {}
    for task in selected_tasks:
        room = task['room_type']
        room_counts[room] = room_counts.get(room, 0) + 1
    
    print(f"\nTask distribution:")
    for room, count in sorted(room_counts.items()):
        print(f"  {room}: {count}")
    
    print(f"\nRunning two_agent policy with max_steps={args.max_steps}")
    if args.resume:
        print("Resume mode: skipping already completed tasks")
    print("-" * 60)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs/two_agent", exist_ok=True)
    
    # Run tasks
    completed = 0
    skipped = 0
    
    for task in selected_tasks:
        task_id = task['task_id']
        
        if should_skip_task(task_id, args.resume):
            print(f"[SKIP] Task {task_id} already completed")
            skipped += 1
            continue
        
        print(f"\n[RUN] TWO_AGENT → Task {task_id}: {task['scene']} → {task['target_object']}")
        
        try:
            run_two_agent_llm(
                max_steps=args.max_steps,
                scene=task['scene'],
                target_object=task['target_object'],
                task_meta=task
            )
            completed += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Progress saved.")
            break
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}")
            # Continue with next task
            continue
    
    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Tasks completed: {completed}")
    print(f"  Tasks skipped: {skipped}")
    print(f"  Total selected: {len(selected_tasks)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
