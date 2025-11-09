"""
AI2-THOR environment wrapper for single and multi-agent navigation experiments.
"""

from typing import Any, Dict, List
import ai2thor.controller


class ThorEnv:
    """
    Wrapper for AI2-THOR Controller that supports both single-agent and multi-agent scenarios.
    Provides convenient methods for stepping through actions and retrieving events.
    """

    def __init__(
        self,
        scene: str = "FloorPlan1",
        agent_count: int = 1,
        grid_size: float = 0.25,
        width: int = 600,
        height: int = 600,
        render_depth_image: bool = False,
        render_instance_segmentation: bool = False,
    ):
        """
        Initialize the AI2-THOR environment.

        Args:
            scene: The AI2-THOR scene name (e.g., "FloorPlan1", "FloorPlan201")
            agent_count: Number of agents (1 for single-agent, 2+ for multi-agent)
            grid_size: Navigation grid size (default: 0.25 meters)
            width: Frame width in pixels
            height: Frame height in pixels
            render_depth_image: Whether to render depth images
            render_instance_segmentation: Whether to render instance segmentation
        """
        self.scene = scene
        self.agent_count = agent_count

        # Initialize the controller
        if agent_count == 1:
            # Single-agent mode
            self.controller = ai2thor.controller.Controller(
                scene=scene,
                gridSize=grid_size,
                width=width,
                height=height,
                renderDepthImage=render_depth_image,
                renderInstanceSegmentation=render_instance_segmentation,
                agentMode="default",
                visibilityDistance=1.5,
            )
            # Store the initial event
            self.events = [self.controller.last_event]
        else:
            # Multi-agent mode
            self.controller = ai2thor.controller.Controller(
                scene=scene,
                gridSize=grid_size,
                width=width,
                height=height,
                renderDepthImage=render_depth_image,
                renderInstanceSegmentation=render_instance_segmentation,
                agentMode="default",
                visibilityDistance=1.5,
                agentCount=agent_count,
            )
            # Store events for all agents
            self.events = self.controller.last_event.events

    def step(self, action: str | Dict[str, Any], agent_id: int = 0) -> Any:
        """
        Execute an action in the environment.

        Args:
            action: Either a string action name (e.g., "MoveAhead") or a dict with action parameters
            agent_id: Which agent to control (0-indexed, only used in multi-agent mode)

        Returns:
            The event object after executing the action
        """
        if isinstance(action, str):
            # Simple string action
            if self.agent_count == 1:
                event = self.controller.step(action=action)
                self.events = [event]
            else:
                event = self.controller.step(action=action, agentId=agent_id)
                self.events = event.events
        else:
            # Dict with action parameters (e.g., for PickupObject)
            if self.agent_count == 1:
                event = self.controller.step(**action)
                self.events = [event]
            else:
                # Ensure agentId is set for multi-agent
                if "agentId" not in action:
                    action["agentId"] = agent_id
                event = self.controller.step(**action)
                self.events = event.events

        return self.get_event(agent_id)

    def get_event(self, agent_id: int = 0) -> Any:
        """
        Get the current event for a specific agent.

        Args:
            agent_id: Which agent's event to retrieve (0-indexed)

        Returns:
            The event object for the specified agent
        """
        if self.agent_count == 1:
            return self.events[0]
        else:
            return self.events[agent_id]

    def close(self):
        """
        Close the environment and clean up resources.
        """
        self.controller.stop()

    def reset(self, scene: str | None = None):
        """
        Reset the environment to initial state.

        Args:
            scene: Optional scene name to switch to. If None, resets current scene.
        """
        if scene is not None:
            self.scene = scene
            self.controller.reset(scene)
        else:
            self.controller.reset(self.scene)

        if self.agent_count == 1:
            self.events = [self.controller.last_event]
        else:
            self.events = self.controller.last_event.events
