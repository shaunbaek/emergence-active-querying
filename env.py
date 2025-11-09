# Thin wrapper around AI2-THOR's Controller.
# Responsible ONLY for simulator setup, stepping, and teardown.

from typing import Any, Mapping, Union

from ai2thor.controller import Controller


class ThorEnv:
    def __init__(
        self,
        scene: str = "FloorPlan1",
        visibility_distance: float = 5.0,
        grid_size: float = 0.25,
        rotate_step_degrees: float = 90.0,
        width: int = 400,
        height: int = 300,
        agent_count: int = 1,
    ):
        """
        Initialize AI2-THOR and spawn one or more agents.
        For ManipulaTHOR, we use agentMode="arm".
        """
        self.agent_count = agent_count

        self.controller = Controller(
            scene=scene,
            agentMode="arm",  # ManipulaTHOR arm agent
            agentCount=agent_count,
            visibilityDistance=visibility_distance,
            gridSize=grid_size,
            rotateStepDegrees=rotate_step_degrees,
            width=width,
            height=height,
        )

        # initial no-op step; in multi-agent mode this returns an Event with .events list
        ev = self.controller.step(action="Pass")
        if hasattr(ev, "events"):
            # multi-agent: list of per-agent events
            self.events = ev.events
        else:
            # single-agent: wrap in a list
            self.events = [ev]

    def step(self, action: Union[str, Mapping[str, Any]], agent_id: int = 0):
        """
        Take an action for a specific agent and update the current events.
        Supports:
          - simple string actions: "MoveAhead"
          - dict actions: {"action": "PickupObject", "objectId": "...", "agentId": ...}
        """
        if isinstance(action, dict):
            action_dict = dict(action)
            if "agentId" not in action_dict:
                action_dict["agentId"] = agent_id
            ev = self.controller.step(action_dict)
        else:
            ev = self.controller.step(action=action, agentId=agent_id)

        if hasattr(ev, "events"):
            self.events = ev.events
        else:
            self.events = [ev]

        return self.events[agent_id]

    def get_event(self, agent_id: int = 0):
        """
        Return the most recent event (observation + metadata) for a given agent.
        """
        return self.events[agent_id]

    def close(self):
        """
        Cleanly shut down the simulator.
        """
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
