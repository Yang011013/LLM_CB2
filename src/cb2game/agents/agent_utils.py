import functools
import concurrent.futures
from cb2game.pyclient.game_endpoint import Action, GameState
import logging
import time
import json
from cb2game.server.db_tools import follower_view_description
from cb2game.pyclient.follower_data_masking import (
    CensorFollowerMap,
    CensorFollowerProps,
)
from cb2game.server.db_tools.follower_view_description import get_instruction_to_location
from cb2game.server.messages.prop import PropUpdate
import re
from cb2game.server.hex import HecsCoord
from cb2game.pyclient.client_utils import DescribeMap
import ast
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
def crop_non_white_square(image_path):

    image = Image.open(image_path)
    image_array = np.array(image)
    non_white_indices = np.where(np.any(image_array != 255, axis=-1))

    min_x, max_x = np.min(non_white_indices[1]), np.max(non_white_indices[1])
    min_y, max_y = np.min(non_white_indices[0]), np.max(non_white_indices[0])
    side_length = max(max_x - min_x, max_y - min_y)

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    left = max(0, center_x - side_length // 2)
    upper = max(0, center_y - side_length // 2)

    right = min(image.width, center_x + side_length // 2)
    lower = min(image.height, center_y + side_length // 2)

    cropped_image = image.crop((left, upper, right, lower))
    cropped_image.save(image_path)

def extract_card_interaction(task_dict):
    immediate_task_str = task_dict.get("Immediate Task", "")
    if "Card Interaction: " in immediate_task_str:
        card_interaction_str = immediate_task_str.split("Card Interaction: ")[1].strip()
        try:
            return ast.literal_eval(card_interaction_str)
        except (SyntaxError, ValueError):
            pass
    return None


def format_checker(response_text, map_description):
    formate_error = ["The response should be in json format with the following keys:Immediate Task, Deferred Task.",
                     "The Immediate Task should be one of the following types: Change Direction, Move, Next Location, Card Interaction.",
                     "When the Immediate Task is type of Change Direction, the Change Direction should be one of the following types: Turn Right, Turn Left, Turn Around.",
                     "When the Immediate Task is type of Move, the Move should be one of the following types: Forward, Backward.",
                     "When the Immediate Task is type of Next Location, The location should be extracted from the NEARBY TILES or FURTHER TILES part of the structured string of your first-view map provided.",
                     "When you deselect a card, the location of the card should be extracted from the SELECTED CARDS part of the structured string of your first-view map provided.",
                     "When you select a card, the location of the card should be extracted from the UNSELECTED CARDS part of the structured string of your first-view map provided.",
                     "The Location of a card or tile shouldn't be appeared in Deferred Task",
                     "The Card Interaction must be the format of 'Card Interaction: ['<interaction> at Tile at heading <angle> and distance <distance>: CARD', '<interaction> at Tile at heading <angle> and distance <distance>: CARD', ...]."
                     ]

    if "json" in response_text:
        start_index = response_text.find("{")
        end_index = response_text.find("}")
        response_text = response_text[start_index:end_index + 1]
    # check if the response is in json format
    try:
        response_dict = json.loads(response_text)
    except:
        return response_text, formate_error[0]
    # check if the response has the correct keys
    print(response_dict)
    if "Immediate Task" not in response_dict.keys() or "Deferred Task" not in response_dict.keys():
        return "", formate_error[0]

    immediate_task = response_dict["Immediate Task"]
    task_type, task_detail = map(str.strip, immediate_task.split(":", 1))

    if task_type not in ["Change Direction", "Move", "Next Location", "Card Interaction"]:
        return response_dict, formate_error[1]


    if task_type == "Change Direction" and task_detail not in ["Turn Right", "Turn Left", "Turn Around"]:
        return response_dict, formate_error[2]
    elif task_type == "Move" and task_detail not in ["Forward", "Backward"]:
        return response_dict, formate_error[3]
    elif task_type == "Card Interaction":
        card_interactions = extract_card_interaction(response_dict)
        if card_interactions is None:
            return response_dict, formate_error[8]

        selected_cards = re.search(r'SELECTED CARDS:\n(.*?)\n\s*UNSELECTED CARDS:', map_description, re.DOTALL)
        unselected_cards = re.search(r'UNSELECTED CARDS:\n(.*?)\n\s*MAP DESCRIPTION', map_description, re.DOTALL)

        for card_interaction in card_interactions:
            card_location = card_interaction.split("Card at", 1)[1].strip()
            if ("Deselect" in card_interaction and card_location not in selected_cards.group(1)) or \
                    ("Select" in card_interaction and card_location not in unselected_cards.group(1)):
                return response_dict, formate_error[5] if "Deselect" in card_interaction else formate_error[6]
            elif "Deselect" not in card_interaction and "Select" not in card_interaction:
                return response_dict, "The card interaction should be either Select or Deselect."
    elif task_type == "Next Location" and task_detail not in map_description.split("NEARBY TILES")[1]:
        return response_dict, formate_error[4]

    if any(keyword in response_dict["Deferred Task"] for keyword in ["Tile at", "Next Location"]):
        return response_dict, formate_error[7]

    return response_dict, None

def get_prompt(instruction, map_update, props, follower, config):
    follower_map_update = CensorFollowerMap(map_update, follower, config)
    follower_props_update = PropUpdate(CensorFollowerProps(props, follower, config))
    description = DescribeMap(follower_map_update, follower_props_update, follower)

    p1 = f"""Here in the instruction you received from the leader: {instruction}
    """
    p2 = f"""Here is the structured string representing your first-view map: \n{description}"""
    p3 = "\nPlease provide your response:"
    prompt = [p1, p2, p3]
    return follower_props_update, prompt, description


def actions_from_code(action_code, i_uuid: str = None):
    if i_uuid is None:
        return None
    # Split action code by comma.
    characters_in_prompt = action_code.split(",")
    if len(characters_in_prompt) == 0:
        # logger.warning("Empty action string.")
        return None
    actions = []
    for c in characters_in_prompt:
        # Convert to lower and strip whitespace.
        c = c.lower().strip()
        if "forward" in c or "f".startswith(c):
            actions.append(Action.Forwards())
        elif "backward" in c or "b".startswith(c):
            actions.append(Action.Backwards())
        elif "left" in c or "l".startswith(c):
            actions.append(Action.Left())
        elif "right" in c or "r".startswith(c):
            actions.append(Action.Right())
        elif "around" in c:
            actions.append(Action.Right())
            actions.append(Action.Right())
            actions.append(Action.Right())
        elif "done" in c or "d".startswith(c):
            actions.append(Action.InstructionDone(i_uuid))
        elif "noop" in c or len(c) == 0:
            actions.append(Action.NoopAction())
        else:
            logger.warning(f"Invalid action code: {c}")
    return actions


def find_matching_tiles(data, keyword):
    if "distance 0" in keyword:
        return "done"
    lines = data.split('\n')
    matching_line = ""
    found = False
    for i, line in enumerate(lines):
        if keyword.lower() in line.lower():
            if "You are standing here." in lines[i + 2]:
                return "done"
            for j in range(i + 2, i + 9):
                if j < len(lines) and "cannot reach" not in lines[j]:
                    matching_line = lines[j].split(":")[1]
                    break
            break
    return matching_line


def deselect_card(mapu, description_atomic, follower, card_location):
    if "distance 0" in card_location:
        follower_orientation = follower.heading_degrees() - 60
        atomic_instructions = {0: "forward, backward", 180: "backward, forward", 60: "right, forward, backward",
                               120: "left, backward, forward", -120: "right, backward, forward",
                               -60: "left, forward, backward"}
        degrees_aways = []
        # available tiles' name: "GROUND_TILE":3, "GROUND_TILE_PATH":28,"MOUNTAIN_TILE":30,"RAMP_TO_MOUNTAIN":31,"SNOWY_MOUNTAIN_TILE":32,"SNOWY_RAMP_TO_MOUNTAIN":36
        actionable_tiles_id = [3, 28, 30, 31, 32, 36]
        for neighbors_location in follower.location().neighbors():
            # 检索该tile的名字，并判断是否在可行的tile中
            neighbors_tile_id = mapu.get_tile_id(neighbors_location)
            if neighbors_tile_id not in actionable_tiles_id:
                continue
            # if the follower doesn't stand on the mountain tiles, GROUND_TILE和GROUND_TILE_PATH、RAMP_TO_MOUNTAIN、SNOWY_RAMP_TO_MOUNTAIN are available.
            if mapu.get_tile_id(follower.location()) in [3, 28]:
                if neighbors_tile_id not in [3, 28, 31, 36]:
                    continue
            # if the follower stand on the mountain tiles, MOUNTAIN_TILE、SNOWY_MOUNTAIN_TILE、RAMP_TO_MOUNTAIN、SNOWY_RAMP_TO_MOUNTAIN are available.
            if mapu.get_tile_id(follower.location()) in [30, 32]:
                if neighbors_tile_id not in [30, 32, 31, 36]:
                    continue
            # if the follower stand on the RAMP, return ""forward, backward, forward" or "backward, forward, backward" to deselect the card
            if mapu.get_tile_id(follower.location()) in [31, 36]:
                return "forward, backward"
            degrees_away = follower.location().degrees_to(neighbors_location) - follower_orientation
            if degrees_away < 0:
                degrees_away += 360
            if degrees_away > 180:
                degrees_away -= 360
            degrees_aways.append(degrees_away)
        for key in atomic_instructions.keys():
            if key in degrees_aways:
                return atomic_instructions[key]
    else:
        atomic_instruction = find_matching_tiles(description_atomic, card_location)
        return atomic_instruction


def parse_hecs_coord(lines, location):
    global coord_str
    for i, line in enumerate(lines):
        if location in line:
            coord_str = lines[i + 1].strip()
            break
    pattern = r"HecsCoord\(a=(\d+), r=(\d+), c=(\d+)\)"
    match = re.match(pattern, coord_str)
    if match:
        a, r, c = map(int, match.groups())
        return HecsCoord(a=a, r=r, c=c)
    else:
        raise ValueError("Invalid HecsCoord string format")


def get_new_orientation(old_orientation, action_string):
    for action in action_string.split(","):
        if action == "right":
            old_orientation -= 60
        elif action == "left":
            old_orientation += 60
        else:
            continue
    return old_orientation % 360


def get_card_interaction_actions(description_atomic, response_dict, follower, map, cards):
    cards_interaction = extract_card_interaction(response_dict)
    card_locations = [card_interaction.split("Card at", 1)[1].strip() for card_interaction in cards_interaction]
    action_string = ""
    if "Deselect" in cards_interaction[0]:
        action_string = deselect_card(map, description_atomic, follower, card_locations[0])
    elif "Select" in cards_interaction[0]:
        if "distance 0" in card_locations[0]:
            action_string = deselect_card(map, description_atomic, follower, card_locations[0])
        else:
            action_string = find_matching_tiles(description_atomic, card_locations[0])
    lines = description_atomic.split('\n')
    last_card_coord = parse_hecs_coord(lines, card_locations[0])
    for location in card_locations[1:]:
        print(location)
        coord = parse_hecs_coord(lines, location)
        follower_new_orientation = get_new_orientation(follower.heading_degrees(), action_string)
        to_next_card = get_instruction_to_location(coord, last_card_coord, follower_new_orientation, map, cards)
        last_card_coord = coord
        action_string += ", " + to_next_card
    return action_string


def get_action_string(response_dict, mapu, prop_update, follower):
    """
    Args:
        image_view_path:
        response_dict: response from LLM
        mapu: the map_update of the whole game
        prop_update: the prop_update of follower's view
        follower:

    Returns:
        action_string: the atomic instruction string to be executed

    """
    deferred_task = response_dict["Deferred Task"]
    immediate_task = response_dict["Immediate Task"]
    description_atomic = ""
    action_string = ""
    if "Card Interaction" in immediate_task or "Next Location" in immediate_task:  # Type1: Change Direction
        start_time = time.time()
        description_atomic = follower_view_description.DescribeMap(mapu, prop_update.props, follower.location(),
                                                                   follower.heading_degrees())
        end_time = time.time()
        print("Time taken to search map: ", end_time - start_time)

    if "Change Direction" in immediate_task:  # Type1: Change Direction
        action_string = immediate_task.split(":")[1].strip()
    elif "Move" in immediate_task:  # Type2: Move
        action_string = immediate_task.split(":")[1].strip()
    elif "Card Interaction" in immediate_task:  # Type4: Card Interaction
        action_string = get_card_interaction_actions(description_atomic, response_dict, follower, mapu,
                                                     prop_update.props)
    elif "Next Location" in immediate_task:  # Type3: Next Location
        action_string = find_matching_tiles(description_atomic, immediate_task.split("Next Location:")[1].strip())
    else:
        action_string = "done"
    if deferred_task == "NULL":
        if action_string.split(",")[-1] != "done":
            action_string += ", done"
    return action_string


def delay_execution(delay_time):
    def decorator(func):
        def wrapper(*args, **kwargs):
            time.sleep(delay_time)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def timeout_decorator(timeout):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout)
                except concurrent.futures.TimeoutError:
                    print("Function call timed out")

        return wrapper

    return decorator
