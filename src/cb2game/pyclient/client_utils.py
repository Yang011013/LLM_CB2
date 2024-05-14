from typing import List

from cb2game.pyclient.follower_data_masking import CoordinateIsVisible
from cb2game.server.actor import Actor
from cb2game.server.assets import AssetId
from cb2game.server.config.config import Config
from cb2game.server.hex import HecsCoord
from cb2game.server.map_utils import AssetId, NatureAssetIds, TreeAssetIds
from cb2game.server.messages.map_update import MapUpdate
from cb2game.server.messages.objective import ObjectiveMessage
from cb2game.server.messages.prop import PropType, PropUpdate
from cb2game.server.messages.turn_state import TurnState
from cb2game.server.routing_utils import get_instruction_to_location


import numpy as np
from PIL import Image

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
    return image_path


def DescribeBearingFromActor(location: HecsCoord, actor: Actor) -> str:
    """Returns a string describing the given location from the perspective of the given actor."""
    distance = round(actor.location().distance_to(location), 1)
    direction = (
        -(
            round(
                actor.heading_degrees()
                - actor.location().degrees_to_precise(location)
                - 60,
                1,
            )
        )
        % 360
    )
    # If we're standing on the tile, give a heading of zero.
    if distance == 0:
        direction = 0
    return f"distance: {distance} units and heading: {direction} degrees"


def DescribeLocationFromActor(location: HecsCoord, actor: Actor, map, cards) -> str:
    """Returns a string describing the given location from the perspective of the given actor."""
    if actor.location() == location:
        return "You are standing here."
    default_instruction = "<Cannot reach>"
    instruction = get_instruction_to_location(
        location, actor, map, cards, default_instruction=default_instruction
    )
    if instruction == default_instruction:
        return f"Path not in sight. {DescribeBearingFromActor(location, actor)}"
    return f"Path to reach: {instruction}"


def FollowerSystemPrompt() -> str:
    # system_prompt = (
    #     "GAME EXPLANATION: \n"
    #     "You are playing a text-based videogame. In this game, you are the FOLLOWER. "
    #     "Your role is to follow the ACTIVE instruction. "
    #     "First you must type in your thoughts. You can do this by starting a line with 'THOUGHTS:' and then typing your thoughts on the same line (important!). "
    #     "Then type in your intended action. You can do this by starting a line with 'ACTIONS:' and then typing a comma-separate list of actions, ending with newline."
    #     "E.G. a List of: 'R', 'L', 'F', 'B', or 'D' (Right, Left, Forward, Back, Done). Actions MUST be one of these single-letters. "
    #     "Use 'D' to mark the instruction as completed. "
    #     "You can take up to 10 actions in your turn.Take advantage of these 10 actions to complete the instruction. "
    #     "You cannot see things behind you or to the sides. You also can't see things that are too far away. Turn around or move to explore. "
    #     "After a few observations, if you're lost, use a 'D' action to get a new instruction. "
    #     "Headings are described in degrees, with positive meaning to the right and "
    #     "negative meaning to the left. You are on a discrete hex grid, each turn is 60 degrees. "
    #     "You get a new observation each time you move. Do not hit 'done' until "
    #     "you have completed the instruction. The leader can see things you can't, "
    #     "so trust instructions. After each ACTION line, you will get a new observation, so you can explore a bit. "
    #     "There are no traps, so explore freely. Outposts, and cities are just waypoints, and have no clues."
    # )
    system_prompt = "You are playing a text-based videogame.\nThe rules of the game are as follows. The game is played by two players (leader and follower). Here I am the leader and you are the follower. This is an environment made up of hexagonal tiles, which means each player has only 6 orientations. The objective of cerealbar is to earn points by selecting valid sets of cards. A valid set has three cards with distinct color, shape, and count. When the only cards selected in the world form a valid set, the players receive a point, the selected cards disappear, three new cards are added randomly, and the number of remaining turns increases. The increase in turns decays for each set completed. An agent stepping on a card flips its selection status. The players form sets together. You only see a first-person view of the environment, and requiree instructions from the follower. You cannot see things behind you or to the sides. You also can't see things that are too far away. Turn around or move to explore. And you don't know the specific content of the card until you select it. Standing on the card means you select/get the card, stepping away from it and then standing on the card again means you deselect the card. I chooses the next target set, plans which of us should get which card, and instructs you. First you must type in your thoughts. You can do this by starting a line with 'THOUGHTS:' and then typing your thoughts on the same line (important!). Then type in your intended actions. You can do this by starting a line with 'ACTIONS:' and then typing a comma-separate list of actions, ending with newline. E.G. a List of: 'R', 'L', 'F', 'B', or 'D' which mean (Right, Left, Forward, Back, Done). Action MUST be one of these single-letters. Use 'D' to mark the instruction as completed. You can perform a turn back(turn around) by typing 'R' or 'L' three times, as this is an environment made up of hexagonal tiles. If the house/lake/tree/lamp one tile in front of you or one tile behind you, then you cannot choose 'F' or 'B'. After each ACTION line, you will get a new observation of your first-person view. Do not choose 'D' until you have completed the instruction. You can take up to 10 actions in your turn.Take advantage of these 10 actions to complete the instruction."
    return system_prompt


def SingleActionSystemPrompt() -> str:
    # system_prompt = (
    #     "GAME EXPLANATION: \n"
    #     "You are playing a text-based videogame. In this game, you are the FOLLOWER. "
    #     "Your role is to follow the ACTIVE instruction. "
    #     "First type in your thoughts. You can do this by starting a line with 'THOUGHTS:' and then typing your thoughts. "
    #     "Then type in your intended action. You can do this by starting a line with 'ACTION:' and then typing only a single action, ending with newline."
    #     "E.G. one of: 'R', 'L', 'F', 'B', or 'D' (Right, Left, Forward, Back, Done). Action MUST be one of these single-letters."
    #     "Use 'D', to mark an instruction as completed. "
    #     "You can take up to 10 actions in your turn.Take advantage of these 10 actions to complete the instruction. "
    #     "You cannot see things behind you or to the sides. You also can't see things that are too far away. Turn around or move to explore. "
    #     "After a few observations, if you're lost, use a 'D' action to get a new instruction. "
    #     "Headings are described in degrees, with positive meaning to the right and "
    #     "negative meaning to the left. You are on a discrete hex grid, each turn is 60 degrees. "
    #     "You get a new observation each time you move. Do not hit 'done' until "
    #     "you have completed the instruction. The leader can see things you can't, "
    #     "so trust instructions. After each ACTION line, you will get a new observation, so you can explore a bit. "
    #     "There are no traps, so explore freely. Outposts, and cities are just waypoints, and have no clues."
    # )
    system_prompt = "You are playing a text-based videogame.\nThe rules of the game are as follows. The game is played by two players (leader and follower). Here I am the leader and you are the follower. This is an environment made up of hexagonal tiles, which means each player has only 6 orientations. The objective of cerealbar is to earn points by selecting valid sets of cards. A valid set has three cards with distinct color, shape, and count. When the only cards selected in the world form a valid set, the players receive a point, the selected cards disappear, three new cards are added randomly, and the number of remaining turns increases. The increase in turns decays for each set completed. An agent stepping on a card flips its selection status. The players form sets together. You only see a first-person view of the environment, and requiree instructions from the follower. You cannot see things behind you or to the sides. You also can't see things that are too far away. Turn around or move to explore. And you don't know the specific content of the card until you select it. Standing on the card means you select/get the card, stepping away from it and then standing on the card again means you deselect the card. I chooses the next target set, plans which of us should get which card, and instructs you. First you must type in your thoughts. You can do this by starting a line with 'THOUGHTS:' and then typing your thoughts on the same line (important!). Then type in your intended actions. You can do this by starting a line with 'ACTION:' and then typing only a single action, ending with newline. E.G. one of: 'R', 'L', 'F', 'B', or 'D' which mean (Right, Left, Forward, Back, Done). Action MUST be one of these single-letters. Use 'D' to mark the instruction as completed. You can perform a turn back(turn around) by typing 'R' or 'L' three times, as this is an environment made up of hexagonal tiles. If the house/lake/tree/lamp one tile in front of you or one tile behind you, then you cannot choose 'F' or 'B'. After each ACTION line, you will get a new observation of yuor first-person view. Do not choose 'D' until you have completed the instruction. You can take up to 10 actions in your turn.Take advantage of these 10 actions to complete the instruction."
    return system_prompt

def FollowerSystemVisionPrompt() -> str:
    system_prompt = """You are playing a text-based videogame.The rules of the game are as follows. 
    The game is played by two players (leader and follower). Here I am the leader and you are the follower. 
    This is an environment made up of hexagonal tiles, which means each player has only 6 orientations. 
    The objective of cerealbar is to earn points by selecting valid sets of cards. 
    A valid set has three cards with distinct color, shape, and count. 
    When the only cards selected in the world form a valid set, the players receive a point.
    The selected cards disappear, three new cards are added randomly, and the number of remaining turns increases. 
    The increase in turns decays for each set completed. An agent stepping on a card flips its selection status. The players form sets together. 
    You can only see the picture information of your first perspective. 
    I will provide you with the first perspective picture at that time when providing instructions. 
    There is an area surrounded by a fan in the picture, and the things within the fan-shaped area are what you can see. 
    The vertex of the fan is the number '22' and a solid red circle. 
    That is where you stand, and the direction of the fan is your direction. 
    Turn around or move to explore. And you don't know the specific content of the card until you select it. 
    Standing on the card means you select/get the card, stepping away from it and then standing on the card again means you deselect the card. 
    I chooses the next target set, plans which of us should get which card, and instructs you. First you must type in your thoughts. 
    You can do this by starting a line with 'THOUGHTS:' and then typing your thoughts on the same line (important!). 
    Then type in your intended actions. You can do this by starting a line with 'ACTIONS:' and then typing a comma-separate list of actions, ending with newline. 
    E.G. a List of: 'R', 'L', 'F', 'B', or 'D' which mean (Right, Left, Forward, Back, Done). 
    Action MUST be one of these single-letters. Use 'D' to mark the instruction as completed. 
    You can perform a turn back(turn around) by typing 'R' or 'L' three times, as this is an environment made up of hexagonal tiles. 
    If the house/lake/tree/lamp one tile in front of you or one tile behind you, then you cannot choose 'F' or 'B'. 
    After each ACTION line, you will get a new observation of your first-person view. 
    Do not choose 'D' until you have completed the instruction. 
    You can take up to 10 actions in your turn.Take advantage of these 10 actions to complete the instruction.
    """
    return system_prompt

def SingleActionSystemVisionPrompt() -> str:
    system_prompt = """You are playing a text-based videogame.The rules of the game are as follows. 
    The game is played by two players (leader and follower). Here I am the leader and you are the follower. 
    This is an environment made up of hexagonal tiles, which means each player has only 6 orientations. 
    The objective of cerealbar is to earn points by selecting valid sets of cards. 
    A valid set has three cards with distinct color, shape, and count. 
    When the only cards selected in the world form a valid set, the players receive a point.
    The selected cards disappear, three new cards are added randomly, and the number of remaining turns increases. 
    The increase in turns decays for each set completed. An agent stepping on a card flips its selection status. The players form sets together. 
    You can only see the picture information of your first perspective. 
    Before you perform each action, I will provide you with your first-person perspective picture. 
    There is an area surrounded by a fan in the picture, and the things within the fan-shaped area are what you can see. 
    The vertex of the fan is the number '22' and a solid red circle. 
    That is where you stand, and the direction of the fan is your direction. 
    Turn around or move to explore. And you don't know the specific content of the card until you select it. 
    Standing on the card means you select/get the card, stepping away from it and then standing on the card again means you deselect the card. 
    I chooses the next target set, plans which of us should get which card, and instructs you. First you must type in your thoughts. 
    You can do this by starting a line with 'THOUGHTS:' and then typing your thoughts on the same line (important!). 
    Then type in your intended actions. You can do this by starting a line with 'ACTION:' and then typing only a single action, ending with newline. 
    E.G. one of: 'R', 'L', 'F', 'B', or 'D' which mean (Right, Left, Forward, Back, Done). 
    Action MUST be one of these single-letters. Use 'D' to mark the instruction as completed. 
    You can perform a turn back(turn around) by typing 'R' or 'L' three times, as this is an environment made up of hexagonal tiles. 
    If the house/lake/tree/lamp one tile in front of you or one tile behind you, then you cannot choose 'F' or 'B'. 
    After each ACTION line, you will get a new observation of your first-person view. 
    Do not choose 'D' until you have completed the instruction. 
    You can take up to 10 actions in your turn.Take advantage of these 10 actions to complete the instruction.
    """
    return system_prompt

def SystemPrompt() -> str:
    system_prompt = """
    You are an embodied agent collaborating with a human leader within an interactive environment. Your primary task is to perform actions based on the instructions provided by the leader. You have the capability to navigate through the environment by moving to adjacent hexagonal tiles and by pivoting to adjust your orientation. Additionally, you can interact with cards by moving over them to select or deselect them. 

    While the leader has access to a comprehensive overhead view of the entire environment, your perspective is limited to a first-person view directly in front of you. You cannot see things behind you or to the sides. You also can't see things that are too far away. This difference in vantage points may result in a vision gap, where the leader's instructions might not be immediately clear from your current perspective. To address this, you may need to take exploratory actions, such as moving forward or turning, to gain a new view and better understand the leader's instructions. 

    In this environment, you can perform the following basic actions:
    - Turn Right: Rotate your direction to the right without moving from the current tile.
    - Turn Left: Rotate your direction to the left without moving from the current tile.
    - Move Forward: Advance one step in your current direction while maintaining the same orientation.
    - Move Back: Retreat one step in the opposite direction of your current orientation, without changing your facing direction.

    You can perform a turn back(turn around) by turn right or left three times, as this is an environment made up of hexagonal tiles.
    If the house/lake/tree/streetlight/rock one tile in front of you or one tile behind you, then you cannot choose move forward or backward. 

    Standing on the card means you select/get the card, stepping away from it and then standing on the card again means you deselect the card.

    # Here is your Objective: \n
    Act as a follower by interpreting and executing the leader's instructions through explorative interaction with the environment.Your goal is to successfully complete the task. \n
    Focus on gaining new perspectives step-by-step to effectively manage and execute these tasks.\n
    To complete the task step by step, you need to break down the original instruction into two categories:
    1. Immediate Tasks: Tasks that are achievable within your current perspective and can be completed promptly.
    2. Deferred Tasks: Tasks that necessitate a change in perspective or additional insights to be accomplished. If there are no deferred tasks, record the output as "NULL". \n

    # Structured String to describe your first-view map :\n
    This environment consists of a hexagonal tile grid, allowing each player six possible orientations. \n
    Due to limited visibility, your perspective is restricted to a fan-shaped area directly in front of you. \n
    The environment's first-person view can be represented by a structured string with the following components:
    - MAP DIMENSIONS: A {map.rows}x{map.cols} hexagon map with {num_cards_in_view} cards in follower's view.
    - CARDS DESCRIPTIONS: A list of tiles with type of CARD
        - SELECTED CARDS: A list of selected cards, formatted "Tile at heading {direction:.0f}° and distance {distance:.1f}: CARD. Shape {shape.name}, color {color.name}, count {count}"
        - UNSELECTED CARDS: A list of unselected cards, formatted "Tile at heading {direction:.0f}° and distance {distance:.1f}: CARD"
    - MAP DESCRIPTION: Descriptions of the lake, mountain, city, and leader visible within the view.
    - NEARBY TILES:
        - Left Tile: At a heading of 60 and a distance of 1.0, named {AssetId(tile.asset_id).name}
        - Right Tile: At a heading of -60 and a distance of 1.0, named {AssetId(tile.asset_id).name}
        - Forward Tile: Directly ahead at a heading of 0 and a distance of 1.0, named {AssetId(tile.asset_id).name}
        (Note: The tile behind you is not visible.)
    - FURTHER TILES: A list of distant tiles, formatted as "Tile at heading {direction:.0f}° and distance {distance:.1f}: {AssetId(tile.asset_id).name}"
    
    A direction of heading greater than 0 means the tile is to your left, less than 0 means the tile is to your right, and equal to 0 means the tile is directly in front of you.
    
    
    # PNG Image of your first-view map: \n
    In addition to the first-view structured map string, you can also get a first-view PNG image, which can help you better understand the relative position of tiles in the map.

    # Definition and format of Immediate Tasks: \n
    These are three types pf tasks that can be completed within your current field of view: \n
    Type 1: Change Direction: You can only turn left, right or around. \n
    Choose one of the following directions:
    - Turn Left
    - Turn Right
    - Turn Around
    Your output should be like: \n
    Change Direction: Turn <direction>

    Type 2: Move to a Specific Location within the visible area of the map.  \n 
    **Important Note:** The location should be extracted from the NEARBY TILES or FURTHER TILES part of the structured string provided. \n
    Here are the situations where you need to move to a specific location: \n
    The leader instructs you to move to a specific location, you should provide the location based on your current view.  
    Or the leader asks you to find a card, but the card is not in your view, you should move to a new location based on the leaders instructions to find it. \n
    Describe the location using the format: Tile at heading <angle> and distance <distance>: <GROUND_TILE_TYPE>. \n
    Your output should be like: \n
    Next Location: Tile at heading <angle> and distance <distance>: <GROUND_TILE_TYPE>

    Type 3: Interact with a Card at a Specific Location within the visible area of the map.  \n
    **Important Note:** The location of the card should be extracted from the CARDS DESCRIPTION part of the structured string provided.\n
    Here are the situations where you need to interact with a card: \n
    If the leader instructs you to pick a card, you should check whether the card is in your view. If it is, you can directly point out the card's location.
    If you are standing on a card, sometimes your leader would ask you to deselect it.To deselect a card, step away from it and then stand on it again. \n
    Choose one of the interactions:
    - Select Card
    - Deselect Card
    Your output should be like: \n
    Card Interaction: <interaction> at Tile at heading <angle> and distance <distance>: CARD

    # Definition and format Deferred Tasks: \n
    These are tasks that require a change in perspective, which becomes possible after completing the immediate tasks. Describe these tasks in natural language not more than 20 words. \n
    You should notice you would get a new view after you complete the immediate tasks, so you should not provide the location from the current view in the deferred tasks. \n

    # Process for Breaking Down Instructions Workflows:
    1、Initial Assessment:
    First, Review the leader's instructions carefully. Then analyze the structured string representing your first-view map.
    Identify any Immediate Tasks that can be executed based on the current perspective.
    2、Identify Immediate Tasks: You can choose one of the three types of immediate tasks to complete.
    3、Identify Deferred Tasks: Summarize any remaining tasks that cannot be completed yet as deferred tasks. You should notice you would get a new view after you complete the immediate tasks, so you should not provide the location extracted from the current structured string in the deferred tasks. \n
    Imagine you have completed the immediate tasks and get a new view, the description should be based on the new view. \n
    If there are no such tasks, record the output for this section as "NULL".
    4、Review and Adjust:
        - Ensure that your immediate and deferred tasks accurately represent the leader’s instructions. Adjust them if necessary to capture all details.
        - Eure that the immediate task location you describe comes from the structured string provided. If not, please choose the correct location and update the description accordingly.

    # Note:
    Provide your answer in JSON format with the following keys:Immediate Task, Deferred Task. Other formats are not accepted.
    Expected Output Format:
    {"Immediate Task": Either a direction change (e.g., "Turn left/right/Around") or a specific location (e.g., "Next Location: Tile at heading <angle> and distance <distance>: <GROUND_TILE_TYPE>") or a card interaction (e.g., "Card Interaction: Select/Deselect Card at Tile at heading <angle> and distance <distance>: CARD"),
    "Deferred Task": "NULL" or a consice description of the remaining instructions in no more then 20 words.}
    Again, you must not calculate the heading and distance to a certain position. You only need to search the structured first-view string and get the best fit "Tile at heading <angle> and distance <distance>: <GROUND_TILE_TYPE>" according to your structured first-view string, first-view PNG image and instructions.
    
    # Example:
    Here are some examples, It should be noted that the Loactions in the example come from the structured first-view string provided each time and is not made up at will. \n:
    instruction from the leader: "Move to the tree and pick up the card." \n
    Output: {"Immediate Task": "Next Location: Tile at heading 0 and distance 4.0: GROUND_TILE", "Deferred Task": "Pick up the card near the tree."} \n

    instruction from the leader: "Get the card that you can see across from you right now; it will be on the right side of the lake"
    Output: {"Immediate Task": "Card Interaction: Select Card at Tile at heading -60 and distance 4.0: CARD", "Deferred Task": "NULL"} \n

    instruction from the leader: "Turn 180 degrees and get the card immediately in front of you" \n
    Output: {"Immediate Task": "Change Direction: Turn Around", "Deferred Task": "Get the card in front of you."} \n

    instruction from the leader: "unselect the card you are on" \n
    Output: {"Immediate Task": "Card Interaction: Deselect Card at Tile at heading 0 and distance 0: CARD", "Deferred Task": "NULL"} \n

    instruction from the leader: "turn left twice and grab that card" \n
    Output: {"Immediate Task": "Change Direction: Turn Left", "Deferred Task": "Turn left again then grab the card nearest to you."} \n

    instruction from the leader: "get the card by the dark green tree" \n
    Output: {"Immediate Task": "Card Interaction: Select Card at Tile at heading 14 and distance 3.6: CARD", "Deferred Task": "NULL"} \n

    instruction from the leader: "take five steps forward then turn around" \n
    Output: {"Immediate Task": "Next Location: Tile at heading 0 and distance 5.0: GROUND_TILE_PATH", "Deferred Task": "turn around"} \n
    
    instruction from the leader: "select the card on distant right of you" \n
    Output: {"Immediate Task": "Card Interaction: Select Card at Tile at heading -60 and distance 5.0: CARD", "Deferred Task": "NULL"} \n
    
    instruction from the leader: "waite" \n
    Output: {"Immediate Task": "Next Location: Tile at heading 0 and distance 0: GROUND_TILE", "Deferred Task": "NULL"} \n
    """
    return system_prompt

def DescribeMap(
    map_update: MapUpdate,
    prop_update: PropUpdate,
    instructions: List[ObjectiveMessage],
    turn_state: TurnState,
    follower: Actor,
    leader: Actor = None,
) -> str:
    """Returns a string describing the given map."""
    header = f"MAP DIMENSIONS:\n\t{map_update.rows}x{map_update.cols} hexagon map with {len(prop_update.props)} props. \n"

    cards = [prop for prop in prop_update.props if prop.prop_type == PropType.CARD]

    fog_end = map_update.fog_end
    if fog_end is None:
        # Create a config object and use the default value.
        default_config = Config()
        fog_end = default_config.fog_end

    # Describe the cards
    selected_card_descriptions = []
    unselected_card_descriptions = []
    for prop in prop_update.props:
        if prop.prop_type == PropType.CARD:
            direction = (
                        follower.heading_degrees()
                        - follower.location().degrees_to_precise(prop.prop_info.location)
                        - 60
                        ) % 360
            if direction > 180:
                direction -= 360
            if direction < -180:
                direction += 360
            if abs(direction) == 0:
                direction = 0
            distance = follower.location().distance_to(prop.prop_info.location)
            # Only show shape, color, count for selected cards.
            if prop.card_init.selected:
                selected_card_descriptions.append(
                    f"Tile at heading {direction:.0f} and distance {distance:.1f}: CARD. Shape {prop.card_init.shape.name}, color {prop.card_init.color.name}, count {prop.card_init.count}"
                )
            else:
                unselected_card_descriptions.append(
                    f"Tile at heading {direction:.0f} and distance {distance:.1f}: CARD"
                )
    # Describe the map metadata
    metadata = map_update.metadata
    metadata_descriptions = []
    for lake in metadata.lakes:
        if CoordinateIsVisible(
            HecsCoord.from_offset(lake.r, lake.c), follower, fog_end
        ):
            location_description = DescribeLocationFromActor(
                HecsCoord.from_offset(lake.r, lake.c), follower, map_update, cards
            )
            metadata_descriptions.append(
                f"Lake of size {lake.size} and shape {lake.type.name} at {location_description}."
            )
    for mountain in metadata.mountains:
        if CoordinateIsVisible(
            HecsCoord.from_offset(mountain.r, mountain.c), follower, fog_end
        ):
            location_description = DescribeLocationFromActor(
                HecsCoord.from_offset(mountain.r, mountain.c),
                follower,
                map_update,
                cards,
            )
            metadata_descriptions.append(
                f"{mountain.type.name} mountain{' (snowy)' if mountain.snowy else ''} at {location_description}."
            )
    for city in metadata.cities:
        if CoordinateIsVisible(
            HecsCoord.from_offset(city.r, city.c), follower, fog_end
        ):
            location_description = DescribeLocationFromActor(
                HecsCoord.from_offset(city.r, city.c), follower, map_update, cards
            )
            metadata_descriptions.append(
                f"City of size {city.size} at {location_description}."
            )
    for outpost in metadata.outposts:
        if CoordinateIsVisible(
            HecsCoord.from_offset(outpost.r, outpost.c), follower, fog_end
        ):
            location_description = DescribeLocationFromActor(
                HecsCoord.from_offset(outpost.r, outpost.c), follower, map_update, cards
            )
            metadata_descriptions.append(f"Outpost at {location_description}.")

    # If provided, and if visible, describe the leader.
    if leader:
        if CoordinateIsVisible(leader.location(), follower, fog_end):
            leader_location = DescribeLocationFromActor(
                leader.location(), follower, map_update, cards
            )
            metadata_descriptions.append(f"Leader at {leader_location}.")

    # Describe nearby tiles
    nearby_tiles = []
    follower_forward = follower.location().neighbor_at_heading(
        follower.heading_degrees()
    )
    follower_left = follower.location().neighbor_at_heading(
        follower.heading_degrees() - 60
    )
    follower_right = follower.location().neighbor_at_heading(
        follower.heading_degrees() + 60
    )
    further_tiles = []
    for tile in map_update.tiles:
        direction = (
            follower.heading_degrees()
            - follower.location().degrees_to_precise(tile.cell.coord)
            - 60
        )% 360
        if direction > 180:
            direction -= 360
        if direction < -180:
            direction += 360
        if abs(direction) == 0:
            direction = abs(direction)
        distance = follower.location().distance_to(tile.cell.coord)
        if tile.cell.coord == follower_forward:
            nearby_tiles.append(f"Forward Tile: Tile at heading 0 and distance 1.0: {AssetId(tile.asset_id).name}")
        elif tile.cell.coord == follower_left:
            nearby_tiles.append(f"Left Tile: Tile at heading 60 and distance 1.0: {AssetId(tile.asset_id).name}")
        elif tile.cell.coord == follower_right:
            nearby_tiles.append(f"Right Tile: Tile at heading -60 and distance 1.0: {AssetId(tile.asset_id).name}")
        else:
            further_tiles.append(f"Tile at heading {direction:.0f} and distance {distance:.1f}: {AssetId(tile.asset_id).name}")

    # Combine all descriptions
    prompt = (
        header
        + "CARDS DESCRIPTIONS\n\t"
        + "SELECTED CARDS:\n\t"
        + "\n\t\t".join(selected_card_descriptions)
        + "\n\tUNSELECTED CARDS:\n\t"
        + "\n\t\t".join(unselected_card_descriptions)
        + "\nMAP DESCRIPTION\n\t"
        + "\n\t".join(metadata_descriptions)
        + "\nNEARBY TILES\n\t"
        + "\n\t".join(nearby_tiles)
        + "\nFURTHER TILES\n\t"
        + "\n\t".join(further_tiles)
    )
    return prompt
