# Policy-related utilities:
# - extracting info from events
# - turning state (+ memory) into text
# - Gemini LLM policies (single agent baseline, single agent handicapped, multi-agent with communication)

import random
from dotenv import load_dotenv
import os
from typing import Any, Dict, List, Mapping, Tuple

import google.generativeai as genai

# Load variables from .env into the environment
load_dotenv()

# Configure Gemini once (expects GEMINI_API_KEY in env)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

ALLOWED_ACTIONS = [
    # navigation
    "MoveAhead",
    "MoveRight",
    "MoveLeft",
    "LookUp",
    "LookDown",
    "RotateRight",
    "RotateLeft",
    "Done",
    # high-level manipulation tokens (mapped in loop.py)
    "PickupNearestVisibleObject",
    "DropHeldObject",
    "OpenNearestVisibleReceptacle",
    "CloseNearestVisibleReceptacle",
]

# Max distance at which the impaired agent can "see" objects, in meters.
HANDICAP_MAX_VIEW_DIST = 2.0

# Lightweight in-memory cache: (state+memory hash) -> action
_llm_action_cache: Dict[str, str] = {}


# --------- Helpers: parse event metadata --------- #

def get_target_info(event, target_type: str) -> Dict[str, Any]:
    """
    Return basic info about the FIRST visible object of the given target_type.

    target_type is compared to obj["objectType"] case-insensitively.
    """
    target_type = target_type.lower()
    for obj in event.metadata["objects"]:
        if obj.get("objectType", "").lower() == target_type:
            return {
                "visible": obj.get("visible", False),
                "distance": obj.get("distance", None),
                "object_id": obj.get("objectId", None),
            }
    return {"visible": False, "distance": None, "object_id": None}



def find_visible_object(
    event,
    object_type: str | None = None,
    pickupable_only: bool = False,
) -> Dict[str, Any] | None:
    """
    Find the first visible object that matches the criteria.
    Returns the full object metadata dict or None.

    - If object_type is given, it must match obj["objectType"] (case-insensitive).
    - If pickupable_only is True, only return objects with pickupable == True.
    """
    for obj in event.metadata["objects"]:
        if not obj.get("visible", False):
            continue
        if pickupable_only and not obj.get("pickupable", False):
            continue
        if object_type is not None and obj.get("objectType", "").lower() != object_type.lower():
            continue
        return obj
    return None


def build_state_text(event, goal_text: str, memory_lines: List[str] | None = None) -> str:
    agent = event.metadata["agent"]
    objs = event.metadata["objects"]

    pos = agent["position"]
    rot_deg = agent["rotation"]["y"]

    visible = [o for o in objs if o.get("visible", False)]

    lines: List[str] = []
    lines.append(f"GOAL: {goal_text}")
    lines.append(
        f"AGENT POSITION: x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f}"
    )
    lines.append(f"AGENT FACING (degrees): {rot_deg:.1f}")

    if visible:
        lines.append("VISIBLE OBJECTS:")
        for o in visible:
            obj_type = o.get("objectType", "UnknownObject")
            obj_name = o.get("name", "UnknownName")
            dist = o.get("distance", None)
            if dist is not None:
                lines.append(f"- {obj_type} ({obj_name}), distance {dist:.2f} m")
            else:
                lines.append(f"- {obj_type} ({obj_name}), distance UNKNOWN")
    else:
        lines.append("VISIBLE OBJECTS: none")

    if memory_lines:
        lines.append("MEMORY:")
        for m in memory_lines:
            lines.append(f"- {m}")

    lines.append(
        "ALLOWED ACTIONS: MoveAhead, MoveRight, MoveLeft, RotateRight, " \
        "RotateLeft, LookUp, LookDown, Done, "
    )

    return "\n".join(lines)


def _bucket_distance(d: float) -> str:
    if d <= 0.7:
        return "very close (within an arm's reach)"
    elif d <= 1.5:
        return "nearby"
    else:
        return "a bit farther away"

def build_state_text_compact(event, goal_text: str, memory_lines: List[str] | None = None) -> str:
    objs = event.metadata["objects"]

    visible_nearby = []
    for o in objs:
        if not o.get("visible", False):
            continue
        dist = o.get("distance", None)
        if dist is None or dist > HANDICAP_MAX_VIEW_DIST:
            continue
        visible_nearby.append(o)

    lines: List[str] = []
    lines.append(f"GOAL: {goal_text}")
    lines.append("AGENT POSITION: (coarse, unknown to you)")
    lines.append(
        "You have limited perception: you only notice nearby objects and only get coarse distance categories."
    )

    if visible_nearby:
        lines.append("YOU SEE NEARBY OBJECTS (with coarse distance):")
        for o in visible_nearby:
            obj_type = o.get("objectType", "UnknownObject")
            dist = o.get("distance", None)
            bucket = _bucket_distance(dist) if dist is not None else "distance unknown"
            lines.append(f"- {obj_type}: {bucket}")
    else:
        lines.append("YOU SEE NEARBY OBJECTS: none")

    if memory_lines:
        lines.append("MEMORY (brief):")
        for m in memory_lines:
            lines.append(f"- {m}")

    lines.append(
        "ALLOWED ACTIONS: MoveAhead, MoveRight, MoveLeft, RotateRight, "
        "RotateLeft, LookUp, LookDown, Done"
    )

    return "\n".join(lines)


# --------- Gemini-based single-agent LLM policy --------- #
# Full Visibility Distance
def llm_policy_gemini(
    event,
    goal_text: str = "Navigate to the microwave.",
    target_type: str = "microwave",
    proximity_done: float = 1.0,
    memory_lines: List[str] | None = None,
    last_action: str | None = None,
    last_target_distance: float | None = None,
    model_name: str = "gemini-2.5-flash",
) -> str:

    """
    Ask Gemini which action to take next.
    target_type: objectType string to navigate to (e.g., 'Microwave', 'Apple')
    """

    # Local stopping check: already close enough to target
    target = get_target_info(event, target_type)
    if target["visible"] and target["distance"] is not None and target["distance"] <= proximity_done:
        return "Done"

    state_txt = build_state_text(event, goal_text, memory_lines=memory_lines)
    extra_lines = []
    if last_action is not None:
        extra_lines.append(f"LAST ACTION: {last_action}")
    if last_target_distance is not None:
        extra_lines.append(f"LAST TARGET DISTANCE: {last_target_distance:.2f} m")

    if extra_lines:
        state_txt = state_txt + "\n" + "\n".join(extra_lines)

    cache_key = str(hash(state_txt))
    if cache_key in _llm_action_cache:
        return _llm_action_cache[cache_key]

    prompt = f"""
You control a mobile robot in an AI2-THOR 3D kitchen.
Your task is to navigate to the {target_type.upper()}.

ONLY RESPOND WITH A PHRASE FROM THIS BRACKETED LIST:
[ MoveAhead, LookUp, LookDown, MoveRight, MoveLeft, RotateRight, RotateLeft, Done ]

Guidelines:
- If the target object ("{target_type}") is visible and its distance is clearly decreasing as you move, prefer MoveAhead.
- If the target object is visible and already very close (within about {proximity_done} meters), choose Done.
- If the target object is NOT visible, DO NOT just keep moving ahead blindly:
  - Use RotateRight or RotateLeft to scan the environment.
  - Use MoveAhead to explore the environment and move toward the target when it is visible.
- Avoid repeating the same action more than twice in a row (for example, do not choose MoveAhead three times in a row if the target is still not visible).
- If the LAST ACTION was MoveAhead and the current target distance did not decrease compared to LAST TARGET DISTANCE, avoid choosing MoveAhead again. Prefer RotateRight or RotateLeft to change direction.
- If MEMORY is provided, use it to reason about where the target was last seen and which direction might be promising.
- DO NOT explain.
- DO NOT output anything except ONE action choice from the bracketed list.

Here is the current world state and memory:
---
{state_txt}
---
Respond with ONLY the action token.
""".strip()

    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)

    raw = (resp.text or "").strip()
    if not raw:
        # Pick a random exploratory move instead of always RotateRight
        action = random.choice(["MoveAhead", "MoveRight", "MoveLeft", "MoveBack"])
    else:
        token = raw.split()[0]
        if token.endswith("."):
            token = token[:-1]
        if token in ALLOWED_ACTIONS:
            action = token
        else:
            # If LLM output is invalid, still explore instead of spinning
            action = random.choice(["MoveAhead", "MoveRight", "MoveLeft", "MoveBack"])


    # ---- STORE IN CACHE ----
    _llm_action_cache[cache_key] = action
    return action


############# Reduced Visibility Distance #############
def llm_policy_gemini_handicapped(
    event,
    goal_text: str,
    target_type: str,
    proximity_done: float = 1.0,
    memory_lines: List[str] | None = None,
    model_name: str = "gemini-2.5-flash",
) -> str:
    # same stopping check
    target = get_target_info(event, target_type)
    if target["visible"] and target["distance"] is not None and target["distance"] <= proximity_done:
        return "Done"

    #handicapped view
    state_txt = build_state_text_compact(event, goal_text, memory_lines=memory_lines)

    cache_key = str(hash(state_txt))
    if cache_key in _llm_action_cache:
        return _llm_action_cache[cache_key]

    prompt = f"""
You control a mobile robot in an AI2-THOR 3D kitchen.
Your perception is limited: you only notice nearby objects and do not know exact distances.
Your task is to navigate to the {target_type.upper()}.

ONLY RESPOND WITH A PHRASE FROM THIS BRACKETED LIST:
[ MoveAhead, LookUp, LookDown, MoveRight, MoveLeft, RotateRight, RotateLeft, Done ]

Guidelines:
- If the {target_type} is visible and you seem to be getting closer, prefer MoveAhead.
- If the {target_type} is visible and already very close (within about {proximity_done} meters), choose Done.
- If the {target_type} is NOT visible:
  - Use RotateRight or RotateLeft to scan the environment and locate the target.
  - Use MoveAhead to explore the environment and move toward the target when it is visible.
- Avoid repeating the same action too many times in a row if your situation does not seem to change.
- If MEMORY is provided, use it to reason about where the {target_type} was last seen.
- If MEMORY is provided, use it to reason about where the target was last seen.
- DO NOT EXPLAIN AND DO NOT OUT ANYTHING EXCEPT FOR [ MoveAhead, LookUp, LookDown, MoveRight, MoveLeft, RotateRight, RotateLeft, Done ].

Here is the current world state and memory:
---
{state_txt}
---
Respond with ONLY the action token.
""".strip()

    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)

    raw = (resp.text or "").strip()
    if not raw:
        # Default to exploratory movement for robustness
        action = random.choice(["MoveAhead", "MoveRight", "MoveLeft", "MoveBack"])
    else:
        token = raw.split()[0]
        if token.endswith("."):
            token = token[:-1]
        if token in ALLOWED_ACTIONS:
            action = token
        else:
            action = random.choice(["MoveAhead", "MoveRight", "MoveLeft", "MoveBack"])

    # ---- STORE IN CACHE ----
    _llm_action_cache[cache_key] = action
    return action

# --------- Multi-agent LLM policy with explicit communication --------- #

def llm_multiagent_policy_gemini(
    leader_event,
    handicapped_event,
    goal_text: str = "Cooperatively navigate to the target.",
    proximity_done: float = 1.0,
    leader_memory: List[str] | None = None,
    handicapped_memory: List[str] | None = None,
    conversation_lines: List[str] | None = None,
    model_name: str = "gemini-2.5-flash",
) -> Tuple[str, str, str, str]:
    """
    Single LLM 'mind' that controls two agents:
      - Leader: full state text
      - Handicapped: compact / degraded state text

    Supports explicit communication:
      - LEADER_MESSAGE
      - HANDICAPPED_MESSAGE

    Returns: (leader_action_token, handicapped_action_token, leader_message, handicapped_message)
    """

    leader_state_txt = build_state_text(
        leader_event,
        goal_text=f"[LEADER VIEW] {goal_text}",
        memory_lines=leader_memory,
    )

    handicapped_state_txt = build_state_text_compact(
        handicapped_event,
        goal_text=f"[HANDICAPPED VIEW] {goal_text}",
        memory_lines=handicapped_memory,
    )

    if conversation_lines:
        convo_block = "DIALOGUE SO FAR:\n" + "\n".join(conversation_lines) + "\n"
    else:
        convo_block = "DIALOGUE SO FAR:\n(None yet)\n"

    prompt = f"""
You control TWO cooperating robots in the same 3D kitchen.

- The LEADER has a rich, detailed perception.
- The HANDICAPPED robot has limited perception and only sees a coarse view.

Your overall task: {goal_text}

You have the following high-level actions available for each robot:
[ MoveAhead, LookUp, LookDown, MoveRight, MoveLeft, RotateRight, RotateLeft, Done ].

The robots can also communicate with short natural-language messages each step to better reason how to navigate.

Guidelines:
- The LEADER should use its richer view and MEMORY to guide the whole team and give clear instructions.
- The HANDICAPPED robot should still act sensibly, but can follow the LEADER's instructions.
- "Done" should be used only when that agent is close enough to the target or when it should stop moving.
- Keep messages short and task-focused.
- BOTH AGENTS SHOULD CONSIDER THEIR DISTANCE AWAY FROM THE TARGET WHEN MAKING CHOICES.
- EVEN IF THE TARGET IS OUTSIDE OF THE LEADER'S VIEW, THE LEADER SHOULD STILL TAKE THE ACTION THAT MINIMIZES THE DISTANCE TO THE TARGET.
- WHEN THE DISTANCE TO THE TARGET IS EQUAL TO OR LESS THAN {proximity_done} BUT THE TARGET IS STILL NOT IN VIEW,
IN ORDER TO FIND IT LookUp AND ROTATE 270 DEGREES, LookDown AND ROTATE 270 DEGREES AND THEN LookDown AGAIN AND ROTATE 270 DEGREES.

{convo_block}

Here is the LEADER's view:
---
{leader_state_txt}
---

Here is the HANDICAPPED robot's view:
---
{handicapped_state_txt}
---

You must respond in the following EXACT format:

LEADER_ACTION: <one of the allowed action tokens> [e.g. MoveAhead, LookUp, LookDown, MoveRight, MoveLeft, RotateRight, RotateLeft, Done ].

LEADER_MESSAGE: <a short natural-language message to the handicapped robot>

HANDICAPPED_ACTION: <one of the allowed action tokens>  [e.g. MoveAhead, LookUp, LookDown, MoveRight, MoveLeft, RotateRight, RotateLeft, Done ].

HANDICAPPED_MESSAGE: <a short natural-language message to the leader>

DO NOT INCLUDE ANYTHING ELSE.
""".strip()

    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)

    raw = (resp.text or "").strip()

    # Defaults
    leader_action = random.choice(["MoveAhead", "MoveRight", "MoveLeft", "MoveBack"]) 
    handicapped_action = random.choice(["MoveAhead", "MoveRight", "MoveLeft", "MoveBack"])
    leader_message = ""
    handicapped_message = ""

    if raw:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        for line in lines:
            u = line.upper()
            if u.startswith("LEADER_ACTION"):
                parts = line.split(":", 1)
                if len(parts) >= 2:
                    token = parts[1].strip().split()[0]
                    if token.endswith("."):
                        token = token[:-1]
                    if token in ALLOWED_ACTIONS:
                        leader_action = token
            elif u.startswith("LEADER_MESSAGE"):
                parts = line.split(":", 1)
                if len(parts) >= 2:
                    leader_message = parts[1].strip()
            elif u.startswith("HANDICAPPED_ACTION"):
                parts = line.split(":", 1)
                if len(parts) >= 2:
                    token = parts[1].strip().split()[0]
                    if token.endswith("."):
                        token = token[:-1]
                    if token in ALLOWED_ACTIONS:
                        handicapped_action = token
            elif u.startswith("HANDICAPPED_MESSAGE"):
                parts = line.split(":", 1)
                if len(parts) >= 2:
                    handicapped_message = parts[1].strip()

    return leader_action, handicapped_action, leader_message, handicapped_message
