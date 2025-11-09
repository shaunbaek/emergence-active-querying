import json
import random
from typing import Dict, List, Any
from env import ThorEnv
from policy import get_target_info 

def sample_random_start_position(env: ThorEnv) -> Dict[str, float]:
    """
    Use AI2-THOR's GetReachablePositions to sample a random valid start position.
    Teleport the *first* agent there and update env.events so that
    subsequent env.get_event() calls see the new position.
    Returns the chosen position dict.
    """
    event = env.controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata.get("actionReturn", [])

    if not reachable_positions:
        raise RuntimeError("No reachable positions returned. Check scene or controller setup.")

    pos = random.choice(reachable_positions)

    # Teleport agent 0 to this position, keeping default rotation/horizon
    env.controller.step(
        action="TeleportFull",
        x=pos["x"],
        y=pos["y"],
        z=pos["z"],
        rotation={"x": 0.0, "y": 0.0, "z": 0.0},
        horizon=0.0,
        standing=True,
        forceAction=True,
        agentId=0,   # explicit: which agent to teleport
    )

    # Refresh env.events so ThorEnv stays in sync
    ev = env.controller.step(action="Pass")
    if hasattr(ev, "events"):
        env.events = ev.events       # multi-agent
    else:
        env.events = [ev]            # single-agent

    return pos

def is_success(event: Any, target_type: str, success_radius: float = 1.0) -> bool:
    """
    Check if the current event satisfies the navigation success condition
    for the given target_type.
    """
    info = get_target_info(event, target_type)
    if not info["visible"]:
        return False
    dist = info["distance"]
    if dist is None:
        return False
    return dist <= success_radius


# ---- Room → scene ranges + target objects ---- #

KITCHEN_SCENES = list(range(1, 31))        # FloorPlan1–30
LIVING_SCENES = list(range(201, 231))      # FloorPlan201–230
BEDROOM_SCENES = list(range(301, 331))     # FloorPlan301–330
BATHROOM_SCENES = list(range(401, 431))    # FloorPlan401–430

KITCHEN_OBJECTS = [
    "Refrigerator",
    "Apple",
    "CoffeeMachine",
    "CounterTop",
    "Cup",
    "Egg",
    "Knife",
    "Microwave",
    "LightSwitch",
    "PepperShaker",
    "Pot",
    "StoveBurner",
    "Toaster",
    "Tomato",
]

LIVING_OBJECTS = [
    "Box",
    "GarbageCan",
    "HousePlant",
    "Laptop",
    "LightSwitch",
    "Painting",
    "Pencil",
    "RemoteControl",
    "Sofa",
    "Television",
]

BEDROOM_OBJECTS = [
    "AlarmClock",
    "Bed",
    "Book",
    "CellPhone",
    "CreditCard",
    "DeskLamp",
    "Mirror",
    "Pen",
    "Pillow",
    "Window",
]

BATHROOM_OBJECTS = [
    "Candle",
    "Cloth",
    "HandTowel",
    "Plunger",
    "SoapBar",
    "SprayBottle",
    "Toilet",
    "ToiletPaper",
    "Towel",
    "TowelHolder",
]


def _make_tasks_for_room(
    room_type: str,
    scene_ids: List[int],
    object_types: List[str],
) -> List[Dict]:
    tasks = []
    for scene_id in scene_ids:
        scene_name = f"FloorPlan{scene_id}"
        for obj in object_types:
            tasks.append(
                {
                    # task_id will be filled later when we combine
                    "task_id": None,
                    "scene": scene_name,
                    "room_type": room_type,
                    "target_type": obj,
                    "goal_text": f"Navigate to the {obj}.",
                }
            )
    return tasks

# ----- GENERATING THE TASK LISTS ----- #

def generate_full_task_list() -> List[Dict]:
    kitchen_tasks = _make_tasks_for_room("kitchen", KITCHEN_SCENES, KITCHEN_OBJECTS)
    living_tasks = _make_tasks_for_room("living_room", LIVING_SCENES, LIVING_OBJECTS)
    bedroom_tasks = _make_tasks_for_room("bedroom", BEDROOM_SCENES, BEDROOM_OBJECTS)
    bathroom_tasks = _make_tasks_for_room("bathroom", BATHROOM_SCENES, BATHROOM_OBJECTS)

    all_tasks = kitchen_tasks + living_tasks + bedroom_tasks + bathroom_tasks

    # assign task_ids 1..N
    for i, t in enumerate(all_tasks, start=1):
        t["task_id"] = i

    return all_tasks

def generate_task_list_100(seed: int = 42) -> List[Dict]:
    """
    Generate a reproducible subset of 100 navigation tasks
    sampled from the full 1320-task list.

    - Uses generate_full_task_list()
    - Shuffles with a fixed seed so you can re-generate the same 100 tasks
    - Reassigns task_id from 1..100
    """
    all_tasks = generate_full_task_list()
    rng = random.Random(seed)
    rng.shuffle(all_tasks)

    subset = all_tasks[:100]

    # Reassign task_ids 1..100 for this subset
    for i, t in enumerate(subset, start=1):
        t["task_id"] = i

    return subset


def generate_dev_task_list_20() -> List[Dict]:
    tasks: List[Dict] = []

    # 2 kitchen scenes × 3 objects
    for scene_id in KITCHEN_SCENES[:2]:
        scene = f"FloorPlan{scene_id}"
        for obj in KITCHEN_OBJECTS[:3]:
            tasks.append(
                {
                    "task_id": None,
                    "scene": scene,
                    "room_type": "kitchen",
                    "target_type": obj,
                    "goal_text": f"Navigate to the {obj}.",
                }
            )

    # 2 living scenes × 3 objects
    for scene_id in LIVING_SCENES[:2]:
        scene = f"FloorPlan{scene_id}"
        for obj in LIVING_OBJECTS[:3]:
            tasks.append(
                {
                    "task_id": None,
                    "scene": scene,
                    "room_type": "living_room",
                    "target_type": obj,
                    "goal_text": f"Navigate to the {obj}.",
                }
            )

    # 2 bedroom scenes × 2 objects
    for scene_id in BEDROOM_SCENES[:2]:
        scene = f"FloorPlan{scene_id}"
        for obj in BEDROOM_OBJECTS[:2]:
            tasks.append(
                {
                    "task_id": None,
                    "scene": scene,
                    "room_type": "bedroom",
                    "target_type": obj,
                    "goal_text": f"Navigate to the {obj}.",
                }
            )

    # 2 bathroom scenes × 2 objects
    for scene_id in BATHROOM_SCENES[:2]:
        scene = f"FloorPlan{scene_id}"
        for obj in BATHROOM_OBJECTS[:2]:
            tasks.append(
                {
                    "task_id": None,
                    "scene": scene,
                    "room_type": "bathroom",
                    "target_type": obj,
                    "goal_text": f"Navigate to the {obj}.",
                }
            )

    # assign task_ids 1..20
    for i, t in enumerate(tasks, start=1):
        t["task_id"] = i

    return tasks

# ----- SAVING THE TASK LISTS ------ # 

def save_full_task_list(path: str = "tasks_full_1320.json"):
    tasks = generate_full_task_list()
    print(f"Generated {len(tasks)} tasks")  # should be 1320
    with open(path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Saved full task list to {path}")

def save_task_list_100(path: str = "tasks_100.json", seed: int = 42):
    tasks = generate_task_list_100(seed=seed)
    print(f"Generated {len(tasks)} tasks for 100-task set")  # should be 100
    with open(path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Saved 100-task list to {path}")

def save_dev_task_list(path: str = "tasks_dev_20.json"):
    tasks = generate_dev_task_list_20()
    print(f"Generated {len(tasks)} dev tasks")  # should be 20
    with open(path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Saved dev task list to {path}")


if __name__ == "__main__":
    save_full_task_list("tasks_full_1320.json")
    save_task_list_100("tasks_100.json")
    save_dev_task_list("tasks_dev_20.json")
