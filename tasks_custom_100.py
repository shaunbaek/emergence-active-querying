"""
Custom 100-task list based on specific test case requirements.
Each task includes the scene (FloorPlan), room type, target object, and goal text.
"""

import json
import random
from typing import List, Dict

def generate_custom_100_tasks() -> List[Dict]:
    """
    Generate 100 custom navigation tasks across 4 room types:
    - 25 Kitchen tasks (FloorPlan1-30)
    - 25 Living room tasks (FloorPlan201-230)
    - 25 Bedroom tasks (FloorPlan301-330)
    - 25 Bathroom tasks (FloorPlan401-430)
    """
    tasks = []
    task_id = 1
    
    # ==================== KITCHEN TASKS (25) ==================== #
    kitchen_scenes = list(range(1, 31))  # FloorPlan1-30
    
    kitchen_tasks_data = [
        ("Egg", "Find the Egg in the kitchen."),
        ("Refrigerator", "Locate the Refrigerator in the kitchen."),
        ("CoffeeMachine", "Navigate to the kitchen and find the CoffeeMachine."),
        ("CounterTop", "Go to the kitchen and find the CounterTop."),
        ("Cup", "Search the kitchen for the Cup."),
        ("Knife", "Head to the kitchen and look for the Knife."),
        ("Microwave", "Find the Microwave in the kitchen."),
        ("LightSwitch", "Locate the LightSwitch in the kitchen."),
        ("PepperShaker", "Search the kitchen for the Pepper Shaker."),
        ("Pot", "Find the Pot in the kitchen."),
        ("StoveBurner", "Navigate to the kitchen and find the Stove Burner."),
        ("Toaster", "Find the Toaster in the kitchen."),
        ("Tomato", "Find the Tomato in the kitchen."),
        ("Apple", "Find the Apple in the kitchen."),
        ("Cup", "Locate the Cup on the kitchen CounterTop."),
        ("Knife", "Go to the kitchen and find the Knife near the Stove Burner."),
        ("CoffeeMachine", "Navigate to the kitchen and look for the CoffeeMachine next to the Refrigerator."),
        ("Pot", "Head to the kitchen and find the Pot on the Stove Burner."),
        ("Microwave", "Search the kitchen for the Microwave above the CounterTop."),
        ("LightSwitch", "In the kitchen, find the LightSwitch by the door."),
        ("PepperShaker", "Find the Pepper Shaker on the CounterTop in the kitchen."),
        ("Toaster", "Locate the Toaster on the CounterTop in the kitchen."),
        ("Apple", "Go to the kitchen and look for the Apple on the CounterTop."),
        ("Tomato", "Navigate to the kitchen and find the Tomato near the Refrigerator."),
        ("Cup", "Head to the kitchen and find the Cup next to the CoffeeMachine."),
    ]
    
    for i, (target, goal) in enumerate(kitchen_tasks_data):
        # Distribute tasks across kitchen scenes
        scene_id = kitchen_scenes[i % len(kitchen_scenes)]
        tasks.append({
            "task_id": task_id,
            "scene": f"FloorPlan{scene_id}",
            "room_type": "kitchen",
            "target_type": target,
            "goal_text": goal,
        })
        task_id += 1
    
    # ==================== LIVING ROOM TASKS (25) ==================== #
    living_scenes = list(range(201, 231))  # FloorPlan201-230
    
    living_tasks_data = [
        ("Laptop", "Find the Laptop in the living room."),
        ("Television", "Locate the Television in the living room."),
        ("RemoteControl", "Navigate to the living room and find the RemoteControl."),
        ("Sofa", "Search the living room for the Sofa."),
        ("Painting", "Find the Painting in the living room."),
        ("HousePlant", "Go to the living room and look for the HousePlant."),
        ("GarbageCan", "Locate the GarbageCan in the living room."),
        ("Box", "Find the Box in the living room."),
        ("Pencil", "Search the living room for the Pencil."),
        ("LightSwitch", "Find the LightSwitch in the living room."),
        ("RemoteControl", "Navigate to the living room and find the RemoteControl on the Sofa."),
        ("Laptop", "Locate the Laptop on the Sofa in the living room."),
        ("GarbageCan", "Head to the living room and find the GarbageCan next to the Sofa."),
        ("HousePlant", "Find the HousePlant near the Television in the living room."),
        ("Box", "Search the living room for the Box under the Painting."),
        ("Pencil", "Go to the living room and find the Pencil near the RemoteControl."),
        ("LightSwitch", "Navigate to the living room and look for the LightSwitch by the entrance."),
        ("Television", "Head to the living room and find the Television above the Sofa."),
        ("Painting", "Find the Painting above the Sofa in the living room."),
        ("Box", "Locate the Box beside the GarbageCan in the living room."),
        ("HousePlant", "Search the living room for the HousePlant next to the Painting."),
        ("RemoteControl", "Find the RemoteControl near the Television in the living room."),
        ("GarbageCan", "Go to the living room and locate the GarbageCan by the door."),
        ("Laptop", "Navigate to the living room and find the Laptop on the Sofa."),
        ("LightSwitch", "Head to the living room and find the LightSwitch near the Television."),
    ]
    
    for i, (target, goal) in enumerate(living_tasks_data):
        scene_id = living_scenes[i % len(living_scenes)]
        tasks.append({
            "task_id": task_id,
            "scene": f"FloorPlan{scene_id}",
            "room_type": "living_room",
            "target_type": target,
            "goal_text": goal,
        })
        task_id += 1
    
    # ==================== BEDROOM TASKS (25) ==================== #
    bedroom_scenes = list(range(301, 331))  # FloorPlan301-330
    
    bedroom_tasks_data = [
        ("Bed", "Find the Bed in the bedroom."),
        ("Pillow", "Locate the Pillow in the bedroom."),
        ("AlarmClock", "Navigate to the bedroom and find the AlarmClock."),
        ("DeskLamp", "Search the bedroom for the DeskLamp."),
        ("Window", "Find the Window in the bedroom."),
        ("Mirror", "Go to the bedroom and look for the Mirror."),
        ("CreditCard", "Locate the CreditCard in the bedroom."),
        ("Book", "Find the Book in the bedroom."),
        ("Pen", "Search the bedroom for the Pen."),
        ("CellPhone", "Find the CellPhone in the bedroom."),
        ("AlarmClock", "Navigate to the bedroom and find the AlarmClock near the Bed."),
        ("DeskLamp", "Head to the bedroom and find the DeskLamp near the Bed."),
        ("Mirror", "Go to the bedroom and find the Mirror on the wall."),
        ("Book", "Search the bedroom for the Book on the Bed."),
        ("Pen", "Locate the Pen near the DeskLamp in the bedroom."),
        ("CreditCard", "Find the CreditCard near the Mirror in the bedroom."),
        ("CellPhone", "Navigate to the bedroom and find the CellPhone on the Bed."),
        ("Pillow", "Head to the bedroom and find the Pillow on the Bed."),
        ("Window", "Search the bedroom for the Window beside the Bed."),
        ("Book", "Go to the bedroom and find the Book near the Window."),
        ("AlarmClock", "Locate the AlarmClock near the Pillow in the bedroom."),
        ("Pen", "Find the Pen on the Bed in the bedroom."),
        ("CreditCard", "Navigate to the bedroom and look for the CreditCard near the Pillow."),
        ("CellPhone", "Head to the bedroom and find the CellPhone next to the Pillow."),
        ("DeskLamp", "Go to the bedroom and find the DeskLamp next to the Window."),
    ]
    
    for i, (target, goal) in enumerate(bedroom_tasks_data):
        scene_id = bedroom_scenes[i % len(bedroom_scenes)]
        tasks.append({
            "task_id": task_id,
            "scene": f"FloorPlan{scene_id}",
            "room_type": "bedroom",
            "target_type": target,
            "goal_text": goal,
        })
        task_id += 1
    
    # ==================== BATHROOM TASKS (25) ==================== #
    bathroom_scenes = list(range(401, 431))  # FloorPlan401-430
    
    bathroom_tasks_data = [
        ("Toilet", "Find the Toilet in the bathroom."),
        ("TowelHolder", "Locate the TowelHolder in the bathroom."),
        ("SoapBar", "Navigate to the bathroom and find the SoapBar."),
        ("Plunger", "Search the bathroom for the Plunger."),
        ("HandTowel", "Find the HandTowel in the bathroom."),
        ("Towel", "Go to the bathroom and look for the Towel."),
        ("SprayBottle", "Locate the SprayBottle in the bathroom."),
        ("Candle", "Find the Candle in the bathroom."),
        ("ToiletPaper", "Search the bathroom for the Toilet Paper."),
        ("Cloth", "Find the Cloth in the bathroom."),
        ("ToiletPaper", "Navigate to the bathroom and find the Toilet Paper near the Toilet."),
        ("SoapBar", "Head to the bathroom and find the SoapBar on the shelf."),
        ("SprayBottle", "Go to the bathroom and find the SprayBottle near the TowelHolder."),
        ("Candle", "Search the bathroom for the Candle near the Towel."),
        ("Plunger", "Find the Plunger beside the Toilet in the bathroom."),
        ("HandTowel", "Navigate to the bathroom and look for the HandTowel on the TowelHolder."),
        ("Cloth", "Head to the bathroom and find the Cloth near the Towel."),
        ("Towel", "Go to the bathroom and find the Towel hanging on the TowelHolder."),
        ("ToiletPaper", "Locate the Toilet Paper on the holder in the bathroom."),
        ("SprayBottle", "Locate the SprayBottle by the door in the bathroom."),
        ("Candle", "Search the bathroom for the Candle on the shelf."),
        ("Towel", "Navigate to the bathroom and find the Towel folded near the Toilet."),
        ("HandTowel", "Head to the bathroom and find the HandTowel near the SoapBar."),
        ("Cloth", "Go to the bathroom and find the Cloth next to the TowelHolder."),
        ("Toilet", "Locate the Toilet in the bathroom."),
    ]
    
    for i, (target, goal) in enumerate(bathroom_tasks_data):
        scene_id = bathroom_scenes[i % len(bathroom_scenes)]
        tasks.append({
            "task_id": task_id,
            "scene": f"FloorPlan{scene_id}",
            "room_type": "bathroom",
            "target_type": target,
            "goal_text": goal,
        })
        task_id += 1
    
    return tasks


def save_custom_100_tasks(path: str = "tasks_custom_100.json"):
    """Save the custom 100-task list to a JSON file."""
    tasks = generate_custom_100_tasks()
    print(f"Generated {len(tasks)} custom tasks")
    
    with open(path, "w") as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Saved custom 100-task list to {path}")
    
    # Print summary
    room_counts = {}
    for task in tasks:
        room = task["room_type"]
        room_counts[room] = room_counts.get(room, 0) + 1
    
    print("\nTask distribution:")
    for room, count in sorted(room_counts.items()):
        print(f"  {room}: {count} tasks")


if __name__ == "__main__":
    save_custom_100_tasks("tasks_custom_100.json")
