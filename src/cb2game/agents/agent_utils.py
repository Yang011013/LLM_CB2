import functools
import concurrent.futures
from cb2game.pyclient.game_endpoint import Action, GameState
import logging
import time
import json
from cb2game.server.db_tools import follower_view_description
logger = logging.getLogger(__name__)
def actions_from_code(action_code, i_uuid: str = None):
    # Split action code by comma.
    characters_in_prompt = action_code.split(",")
    if len(characters_in_prompt) == 0:
        # logger.warning("Empty action string.")
        return None
    actions = []
    for c in characters_in_prompt:
        # Convert to lower and strip whitespace.
        c = c.lower().strip()
        if "forward".startswith(c) or "f".startswith(c):
            actions.append(Action.Forwards())
        elif "forward" in c:
            actions.append(Action.Forwards())
        elif "backward".startswith(c) or "b".startswith(c):
            actions.append(Action.Backwards())
        elif "backward" in c:
            actions.append(Action.Backwards())
        elif "left".startswith(c) or "l".startswith(c):
            actions.append(Action.Left())
        elif "left" in c:
            actions.append(Action.Left())
        elif "right".startswith(c) or "r".startswith(c):
            actions.append(Action.Right())
        elif "right" in c:
            actions.append(Action.Right())
        elif "around" in c:
            actions.append(Action.Right())
            actions.append(Action.Right())
            actions.append(Action.Right())
        elif "done".startswith(c) or "d".startswith(c):
            actions.append(Action.InstructionDone(i_uuid))
        elif "noop" in c:
            actions.append(Action.NoopAction())
        else:
            logger.warning(f"Invalid action code: {c}")
    return actions

def find_matching_tiles(data, keyword):# keyword-->Next Location: Tile at heading  <angle> and distance <distance>: <TILE_TYPE>
    if "distance 0" in keyword:
        return ""
    lines = data.split('\n')
    matching_line = ""
    found = False
    for i, line in enumerate(lines):
        if keyword.lower() in line.lower():
            if "You are standing here." in lines[i+2]:
                return ""
            for j in range(i+2, i+9):
                if j < len(lines) and "cannot reach" not in lines[j]:
                    matching_line = lines[j].split(":")[1]
                    break
            break
    return matching_line

def deselect_card(mapu, description_atomic, follower, card_location):
    # 1.卡片在当前位置：forward+backward+forward或者backward+forward+backward
    if "distance 0" in card_location:
        # 还要考虑follower的朝向
        follower_orientation = follower.heading_degrees() - 60
        atomic_instructions = []
        # 检查周围的tile是否可行
        # 可行的tile.name: "GROUND_TILE":3, "GROUND_TILE_PATH":28,"MOUNTAIN_TILE":30,"RAMP_TO_MOUNTAIN":31,"SNOWY_MOUNTAIN_TILE":32,"SNOWY_RAMP_TO_MOUNTAIN":36
        actionable_tiles_id = [3, 28, 30, 31, 32, 36]
        for neighbors_location in follower.location().neighbors():
            # 检索该tile的名字，并判断是否在可行的tile中
            neighbors_tile_id = mapu.get_tile_id(neighbors_location)
            if neighbors_tile_id not in actionable_tiles_id:
                continue
            # 如果当前follower不在mountain类的tile上，那么只能选择GROUND_TILE和GROUND_TILE_PATH、RAMP_TO_MOUNTAIN、SNOWY_RAMP_TO_MOUNTAIN
            if mapu.get_tile_id(follower.location()) in [3, 28]:
                if neighbors_tile_id not in [3, 28, 31, 36]:
                    continue
            # 如果当前follower在mountain类的tile上，那么只能选择MOUNTAIN_TILE、SNOWY_MOUNTAIN_TILE、RAMP_TO_MOUNTAIN、SNOWY_RAMP_TO_MOUNTAIN
            if mapu.get_tile_id(follower.location()) in [30, 32]:
                if neighbors_tile_id not in [30, 32, 31, 36]:
                    continue
            # 如果当前follower在RAMP上，志强选择""forward, backward, forward"或者"backward, forward, backward"
            if mapu.get_tile_id(follower.location()) in [31, 36]:
                # 先不考虑ramp的朝向
                atomic_instructions.append("forward, backward, forward")
                continue
            degrees_away = follower.location().degrees_to(neighbors_location) - follower_orientation
            if degrees_away < 0:
                degrees_away += 360
            if degrees_away > 180:
                degrees_away -= 360
            if degrees_away == 0:
                atomic_instructions.append("forward, backward")
            elif degrees_away == 60:
                atomic_instructions.append("right, forward, backward")
            elif degrees_away == 120:
                atomic_instructions.append("left, backward, forward")
                #atomic_instructions.append("right, right, forward, backward")
            elif degrees_away == 180:
                atomic_instructions.append("backward, forward, backward")
            elif degrees_away == -60:
                atomic_instructions.append("left, forward, backward")
            elif degrees_away == -120:
                atomic_instructions.append("right, backward, forward")
                #atomic_instructions.append("left, left, forward, backward")
        return atomic_instructions[0] # 返回第一个可行的tile，的方式
    # 2.卡片不在当前位置：直接去卡片的位置
    else:
        atomic_instruction = find_matching_tiles(description_atomic, card_location)
        return atomic_instruction

def get_action_string(response_dict, mapu, prop_update, follower):
    """
    Args:
        response_dict: response from LLM
        mapu: the map_update of the whole game
        prop_update: the prop_update in follower's view
        follower:

    Returns:
        action_string: the atomic instruction string to be executed

    """
    deferred_task = response_dict["Deferred Task"]
    immediate_task = response_dict["Immediate Task"]
    # try: # 可能输出不规范
    #     response_dict = json.loads(response_text)
    #     response_dict = json.loads(response_text)
    #     self.deferred_task = response_dict["Deferred Task"]
    #     immediate_task = response_dict["Immediate Task"]
    # except KeyError as e:

    # 只要去具体位置才广搜
    description_atomic = ""
    action_string = ""
    if "Card Interaction" in immediate_task or "Next Location" in immediate_task:  # Type1: Change Direction
        start_time = time.time()
        description_atomic = follower_view_description.DescribeMap(mapu, prop_update.props, follower.location(),
                                                                   follower.heading_degrees())
        end_time = time.time()
        print("Time taken to search map: ", end_time - start_time)
        with open("description_atomic.txt", "w") as f:
            f.write(description_atomic)

    if "Change Direction" in immediate_task:  # Type1: Change Direction
        action_string = immediate_task.split(":")[1].strip()
    elif "Card Interaction" in immediate_task:  # Type3: Next Location
        card_location = immediate_task.split("Card at")[1].strip()
        if "Deselect" in immediate_task:
            action_string = deselect_card(mapu, description_atomic, follower, card_location)
        elif "Select" in immediate_task:
            if "distance 0" in card_location:
                # 如果follower在卡片上，那么只能选择卡片相当于取消选择卡片
                action_string = deselect_card(mapu, description_atomic, follower, card_location)
            else:
                action_string = find_matching_tiles(description_atomic, card_location)

    elif "Next Location" in immediate_task:  # Type3: Next Location
        action_string = find_matching_tiles(description_atomic, immediate_task.split("Next Location:")[1].strip())
    else:
        action_string = "done"
    if deferred_task == "NULL":
        action_string += ",done"
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