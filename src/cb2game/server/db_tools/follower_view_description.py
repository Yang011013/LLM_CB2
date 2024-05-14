# Creates a set of graphics where an instruction is displayed on the left, and
# the follower's pathway is displayed on the right.
import json
import logging
import pathlib
from typing import List
from cb2game.server.config.config import Config
import fire
import peewee
import pygame
import pygame.freetype
import dataclasses
import cb2game.server.config.config as config
import cb2game.server.db_tools.db_utils as db_utils
import cb2game.server.messages.live_feedback as live_feedback_msg
import cb2game.server.messages.map_update as map_update_msg
import cb2game.server.messages.prop as prop_msg
import cb2game.server.schemas.defaults as defaults_db
from cb2game.server.card import Card
from cb2game.server.hex import HecsCoord
from cb2game.server.map_tools import visualize
from cb2game.server.messages.action import Action
from cb2game.server.messages.objective import ObjectiveMessage
from cb2game.server.schemas import base
from cb2game.server.schemas.event import Event, EventType
from cb2game.pyclient.client_utils import crop_non_white_square, DescribeMap
from cb2game.server.map_utils import AssetId, NatureAssetIds, TreeAssetIds
from cb2game.server.messages.prop import PropType, PropUpdate
from collections import deque
from cb2game.server.messages.map_update import MapUpdate
logger = logging.getLogger(__name__)

pygame.freetype.init()
INSTRUCTION_FONT = pygame.freetype.SysFont("Times New Roman", 30)

# The below imports are used to import pygame in a headless setup, to render map
# updates as images for game recordings.
import os

# set SDL to use the dummy NULL video driver,
#   so it doesn't need a windowing system.
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame.transform

if 1:
    # some platforms might need to init the display for some parts of pygame.
    import pygame.display

    pygame.display.init()
    screen = pygame.display.set_mode((1, 1))

SCREEN_SIZE = 800
UNITY_COORDINATES_SCALE = 3.46
FOLLOWER_FOV = 96.5
def CoordinateNeighborCells(location, orientation):
    # Get the two neighboring cells to the left and right. Special case them.
    return [
        location.neighbor_at_heading((orientation - 60) % 360),
        location.neighbor_at_heading((orientation + 60) % 360),
    ]

def CoordinateInViewingDistance(coord, follower_location, fog_end):
    """Returns true if the given coordinate should be visible to the given follower with the given config."""
    view_depth = fog_end / UNITY_COORDINATES_SCALE
    # Check distance.
    distance = coord.distance_to(follower_location)
    # Add 0.5 to round up to the next hex cell.
    return distance <= (view_depth + 0.5)

def CoordinateInFov(coord, follower_location, follower_orientation, config):
    # Check FOV.
    follower_orientation = follower_orientation - 60
    degrees_to = follower_location.degrees_to_precise(coord) % 360
    left = (follower_orientation - FOLLOWER_FOV / 2) % 360
    right = (follower_orientation + FOLLOWER_FOV / 2) % 360
    if left < right:
        return left <= degrees_to <= right
    else:
        return left <= degrees_to or degrees_to <= right

def VisibleCoordinates(follower_location, follower_orientation, config):
    """Given a follower's location and orientation, returns all HecsCoords that are visible."""
    visible_coords = []

    # Get the two neighboring cells to the left and right. Special case them.
    neighbor_coords = CoordinateNeighborCells(follower_location, follower_orientation)
    visible_coords.extend(neighbor_coords)

    # BFS from the follower's location, find all visible coordinates.
    next_coords = deque([follower_location])
    already_visited = set()
    while len(next_coords) > 0:
        coord = next_coords.popleft()
        if coord in already_visited:
            continue
        already_visited.add(coord)
        if coord in visible_coords:
            continue
        if (coord != follower_location) and (
            not CoordinateInViewingDistance(coord, follower_location, config.fog_end)
            or not CoordinateInFov(coord, follower_location, follower_orientation, config)
        ):
            continue
        visible_coords.append(coord)
        for neighbor in coord.neighbors():
            next_coords.append(neighbor)
    return visible_coords

def CensorFollowerMap(map_update, follower_location, follower_orientation, config):
    config.fog_end / UNITY_COORDINATES_SCALE

    visible_coords = VisibleCoordinates(follower_location, follower_orientation, config)
    new_tiles = []
    for coord in visible_coords:
        tile = map_update.tile_at(coord)
        if tile is None:
            continue
        new_tiles.append(tile)
    filtered_map_update = MapUpdate(
        map_update.rows, map_update.cols, new_tiles, map_update.metadata
    )
    return filtered_map_update

def CoordinateIsVisible(coord, follower_location, follower_orientation, fog_end):
    # Get the two neighboring cells to the left and right. Special case them.
    if coord in CoordinateNeighborCells(follower_location, follower_orientation):
        return True

    """  Returns true if the given coordinate should be visible to the given follower with the given config. """
    view_depth = fog_end / UNITY_COORDINATES_SCALE

    # Check distance.
    distance = coord.distance_to(follower_location)
    # Add 0.5 to round up to the next hex cell.
    if distance > (view_depth + 0.5):
        return False
    # Special case distance == 0 to avoid weird FOV calculations.
    if distance == 0:
        return True
    # Check FOV.
    degrees_to = follower_location.degrees_to_precise(coord) % 360
    left = (follower_orientation - FOLLOWER_FOV / 2) % 360
    right = (follower_orientation + FOLLOWER_FOV / 2) % 360
    if left < right:
        return left <= degrees_to <= right
    else:
        return left <= degrees_to or degrees_to <= right

def CensorFollowerProps(props, follower_location, follower_orientation, config):
    """Removes all props which aren't visible to the follower.
    """
    new_props = []
    for prop in props:
        if CoordinateIsVisible(prop.prop_info.location, follower_location, follower_orientation, config.fog_end):
            new_props.append(dataclasses.replace(prop))
    return new_props

def find_path_to_card(location: HecsCoord, follower_location, follower_orientation, map, cards):
    start_location = follower_location
    end_location = location
    location_queue = deque()
    location_queue.append((start_location, [start_location]))
    card_locations = set([card.prop_info.location for card in cards])
    if start_location in card_locations:
        card_locations.remove(start_location)
    if end_location in card_locations:
        card_locations.remove(end_location)
    visited_locations = set()
    while len(location_queue) > 0:
        current_location, current_path = location_queue.popleft()
        if current_location in visited_locations:
            continue
        if current_location in card_locations: # 绕过card？？
            continue
        visited_locations.add(current_location)
        if current_location == end_location:
            return current_path
        tile = map.tile_at(current_location)
        for neighbor in tile.cell.coord.neighbors():
            if tile.cell.boundary.get_edge_between(tile.cell.coord, neighbor):
                continue
            neighbor_tile = map.tile_at(neighbor)
            # This can happen if routing on a follower view with limited map visibility.
            if neighbor_tile is None:
                continue
            if neighbor_tile.cell.boundary.get_edge_between(neighbor, tile.cell.coord):
                continue
            location_queue.append((neighbor, current_path + [neighbor]))
    return None
def get_instruction_to_location(
    location: HecsCoord,
    follower_location,
    follower_orientation,
    map,
    cards,
    game_endpoint=None,
    default_instruction="random, random, random, random, random, random",
):
    distance_to_follower = lambda c: c.prop_info.location.distance_to(
        follower_location
    )
    path = find_path_to_card(location, follower_location, follower_orientation, map, cards)
    if not path:
        return default_instruction
    game_vis = game_endpoint.visualization() if game_endpoint else None
    if game_vis is not None:
        game_vis.set_trajectory([(coord, 0) for coord in path])
    heading = follower_orientation - 60
    instructions = []
    for idx, location in enumerate(path):
        next_location = path[idx + 1] if idx + 1 < len(path) else None
        if not next_location:
            break
        degrees_away = location.degrees_to(next_location) - heading
        if degrees_away < 0:
            degrees_away += 360
        if degrees_away > 180:
            degrees_away -= 360
        # Pre-defined shortcuts to introduce backstepping.
        if degrees_away == 180:
            instructions.append("backward")
            location = next_location
            continue
        if degrees_away == 120:
            instructions.extend(["right"] * 2)
            instructions.append("forward")
            heading += 120
            location = next_location
            continue
        if degrees_away == -120:
            instructions.extend(["left"] * 2)
            instructions.append("forward")
            heading -= 120
            location = next_location
            continue
        # General-case movement pattern.
        if degrees_away > 0:
            instructions.extend(["right"] * int(degrees_away / 60))
        else:
            instructions.extend(["left"] * int(-degrees_away / 60))
        heading += degrees_away
        instructions.append("forward")
        location = next_location

    return ", ".join(instructions)

def DescribeBearingFromActor(location: HecsCoord, follower_location, follower_orientation) -> str:
    """Returns a string describing the given location from the perspective of the given actor."""
    distance = round(follower_location.distance_to(location), 1)
    direction = (
        -(
            round(
                follower_orientation
                - follower_location.degrees_to_precise(location)
                - 60,
                1,
            )
        )
        % 360
    )
    if direction > 180:
        direction -= 360
    elif direction < -180:
        direction += 360
    # If we're standing on the tile, give a heading of zero.
    if distance == 0:
        direction = 0
    return f"distance: {distance} units and heading: {direction} degrees"


def DescribeLocationFromActor(location: HecsCoord, follower_location, follower_orientation, map, cards) -> str:
    """Returns a string describing the given location from the perspective of the given actor."""
    if follower_location == location:
        return "You are standing here."
    
    default_instruction = "<cannot reach>"
    instruction = get_instruction_to_location(
        location, follower_location, follower_orientation, map, cards, default_instruction=default_instruction
    )

    if instruction == default_instruction:
        # neighbor_cells = follower_location.neighbors()
        neighbor_cells = location.neighbors()
        atomic_to_neighbors = ["To up right", "To right", "To down right", "To down left", "To left", "To up left"]
        for i, neighbor in enumerate(neighbor_cells):
            instruction = get_instruction_to_location(
                neighbor, follower_location, follower_orientation, map, cards, default_instruction=default_instruction
            )
            if instruction != default_instruction:
                atomic_to_neighbors[i] = f"{atomic_to_neighbors[i]}: {instruction}"
                break
            else:
                atomic_to_neighbors[i] = f"{atomic_to_neighbors[i]}: cannot reach"
        return f"""{default_instruction}. 
                {atomic_to_neighbors[0]}
                {atomic_to_neighbors[1]}
                {atomic_to_neighbors[2]}
                {atomic_to_neighbors[3]}
                {atomic_to_neighbors[4]}
                {atomic_to_neighbors[5]}"""
    return f"Path to reach: {instruction}"
def DescribeMap(
    map_update: MapUpdate,
    props,
    instructions: List[ObjectiveMessage],
    follower_location = None,
    follower_orientation = None,
    only_map = True,
    turn_state = None,
) -> str:
    """Returns a string describing the given map."""
    header = f"MAP DIMENSIONS:\n\t{map_update.rows}x{map_update.cols} hexagon map with {len(props)} props. \n"

    cards = [prop for prop in props if prop.prop_type == PropType.CARD]

    fog_end = map_update.fog_end
    if fog_end is None:
        # Create a config object and use the default value.
        default_config = Config()
        fog_end = default_config.fog_end
    default_config = Config()
    visible_coord = VisibleCoordinates(follower_location, follower_orientation, default_config)

    # Describe the cards
    selected_cards_descriptions = []
    unselected_cards_descriptions = []
    for prop in props:
        if prop.prop_type == PropType.CARD:
            location_description = DescribeLocationFromActor( # 返回一个字符串，描述follower到prop的路径
                prop.prop_info.location, follower_location, follower_orientation, map_update, cards
            )
            direction = (
                    follower_orientation
                    - follower_location.degrees_to_precise(prop.prop_info.location)
                    - 60
            ) % 360
            if direction > 180:
                direction -= 360
            elif direction < -180:
                direction += 360
            if abs(direction) == 0:
                direction = 0
            distance = follower_location.distance_to(prop.prop_info.location)
            # Only show shape, color, count for selected cards.
            if prop.card_init.selected:
                selected_cards_descriptions.append(f"Tile at heading {direction:.0f} and distance {distance:.1f}: CARD\n\t\t{prop.prop_info.location}\n\t\t{location_description}")
            else:
                unselected_cards_descriptions.append(f"Tile at heading {direction:.0f} and distance {distance:.1f}: CARD\n\t\t{prop.prop_info.location}\n\t\t{location_description}")
                # prop_descriptions.append(f"Card at {location_description}.")

    # Describe nearby tiles
    nearby_tiles = []
    follower_forward = follower_location.neighbor_at_heading(
        follower_orientation
    )
    follower_left = follower_location.neighbor_at_heading(
        follower_orientation - 60
    )
    follower_right = follower_location.neighbor_at_heading(
        follower_orientation + 60
    )
    further_tiles = []
    follower_mapu = CensorFollowerMap(map_update, follower_location, follower_orientation, default_config)
    print("follower_mapu tiles: ", len(follower_mapu.tiles))
    for tile in follower_mapu.tiles:
        direction = (
                follower_orientation
                - follower_location.degrees_to_precise(tile.cell.coord)
                - 60
        ) % 360
        if direction > 180:
            direction -= 360
        elif direction < -180:
            direction += 360
        if abs(direction) == 0:
            direction = 0
        distance = follower_location.distance_to(tile.cell.coord)

        if tile.cell.coord == follower_forward:
            nearby_tiles.append(f"Tile at heading 0 and distance 1.0: {AssetId(tile.asset_id).name}\n\t\t{tile.cell.coord}\n\t\tPath to reach: forward")
        elif tile.cell.coord == follower_left:
            nearby_tiles.append(f"Tile at heading 60 and distance 1.0: {AssetId(tile.asset_id).name}\n\t\t{tile.cell.coord}\n\t\tPath to reach: left, forward")
        elif tile.cell.coord == follower_right:
            nearby_tiles.append(f"Tile at heading -60 and distance 1.0: {AssetId(tile.asset_id).name}\n\t\t{tile.cell.coord}\n\t\tPath to reach: right, forward")
        elif tile.cell.coord == follower_location:
            continue
        else:
            if tile.cell.coord not in visible_coord:
                continue
            tile_description = DescribeLocationFromActor(tile.cell.coord, follower_location, follower_orientation,
                                                        map_update, cards)
            further_tiles.append(
                f"Tile at heading {direction:.0f} and distance {distance:.1f}: {AssetId(tile.asset_id).name}\n\t\t{tile.cell.coord}\n\t\t{tile_description}"
            )
    # Combine all descriptions
    prompt = (
        header
        + "CARDS DESCRIPTIONS\n\t"
        + "SELECTED CARDS:\n\t"
        + "\n\t".join(selected_cards_descriptions)
        + "\n\tUNSELECTED CARDS:\n\t"
        + "\n\t".join(unselected_cards_descriptions)
        + "\nNEARBY TILES\n\t"
        + "\n\t".join(nearby_tiles)
        + "\nFURTHER TILES\n\t"
        + "\n\t".join(further_tiles)
    )
    return prompt

def draw_follower_view(
    instruction: ObjectiveMessage,
    moves: List[Action],
    feedbacks: List[live_feedback_msg.LiveFeedback],
    map_update: map_update_msg.MapUpdate,
    file_path: str,
    game_id: int,
    props: List[prop_msg.Prop],  # 只限card
    config,
    description,
    only_map, # 只描述地图信息，不描述turn state和instruction
):

    trajectory = [(move.location, move.orientation) for move in moves]
    n = 0
    for (location, orientation) in trajectory:
        display = visualize.GameDisplay(SCREEN_SIZE)
        display.set_config(config)

        follower_map_update = CensorFollowerMap(map_update, location, orientation, config)
        display.set_map(follower_map_update)
        follower_props = CensorFollowerProps(props, location, orientation, config)
        display.set_props(follower_props)

        display.draw()
        display.visualize_follower_location_orientation(location, orientation)

        n += 1
        instr = instruction.text
        if '/' in instr:
            instr = instr.replace('/', ' or ')
        if '.' in instr:
            instr = instr.replace('.', '')
        if '"' in instr:
            instr = instr.replace('"', "'")

        filename = file_path / f"{instr}_{n}.png"
        print(f"Saving image to {filename}")

        pygame.display.flip()
        pygame.image.save(display.screen(), filename)


        if description:
            description_filename = file_path / f"{instr}_{n}_description.txt"
            map_description = DescribeMap(
                follower_map_update, follower_props, [instruction], location, orientation, only_map,
            )
            # 保存到文件
            print(description_filename)
            with open(description_filename, "w") as file:
                file.write(map_description)

        crop_non_white_square(filename)
        # break

def main(
    max_instructions=-1,
    config_filepath="C:/Users/keyang/Desktop/yan0/Agent/cb2/follower_bots/pretraining_data/cb2-data-base/config/human_model.yaml",
    output_dir="C:/Users/keyang/Desktop/yan0/Agent/cb2/follower_view/human_model/",
    research_only=True,
):
    logging.basicConfig(level=logging.INFO)
    if config_filepath == "":
        cfg = config.Config()
        logger.warning(
            f"No config was provided. Using default database located at: {cfg.database_path()}"
        )
    else:
        cfg = config.ReadConfigOrDie(config_filepath)
    # Setup the sqlite database used to record game actions.
    base.SetDatabase(cfg)
    base.ConnectDatabase()
    base.CreateTablesIfNotExists(defaults_db.ListDefaultTables())

    output_dir = pathlib.Path(output_dir).expanduser()
    # Create the directory if it doesn't exist.
    output_dir.mkdir(parents=False, exist_ok=True)

    words = set()
    instruction_list = []

    if research_only:
        games = db_utils.ListAnalysisGames(cfg)
    else:
        games = [
            game for game in db_utils.ListGames() if db_utils.IsConfigGame(cfg, game)
        ]
    print(f"Found {len(games)} games.")
    ParentEvent = Event.alias()
    # For each game.
    for game in games:
        # Create a directory for the game.
        game_dir = output_dir / str(game.id)
        game_dir.mkdir(parents=False, exist_ok=True)
        # Do a self-join to link parent events. ParentEvent is an alias of Event defined above.
        game_events = (
            Event.select()
            .join(
                ParentEvent,
                peewee.JOIN.LEFT_OUTER,
                on=(Event.parent_event == ParentEvent.id),
            )
            .where(Event.game_id == game.id)
            .order_by(Event.server_time)
        )
        map_events = (
            game_events.select()
            .where(Event.type == EventType.MAP_UPDATE)
            .order_by(Event.server_time)
        )
        if map_events.count() == 0:
            print(f"Skipping game {game.id} because it has no map update events.")
            continue
        prop_updates = game_events.where(Event.type == EventType.PROP_UPDATE).order_by(
            Event.server_time
        )
        if not prop_updates.exists():
            print(f"Skipping game {game.id} because it has no prop update events.")
            continue

        follower_action_events = game_events.where((Event.type == EventType.ACTION) & (Event.role == 'Follower')).order_by(
            Event.server_time
        )

        first_map_update = map_update_msg.MapUpdate.from_json(map_events.get().data)
        first_prop_update = prop_msg.PropUpdate.from_json(prop_updates.get().data)
        initial_cards = [Card.FromProp(prop) for prop in first_prop_update.props]
        instructions = game_events.where(Event.type == EventType.INSTRUCTION_SENT)

        for instruction in instructions:
            activation_query = instruction.children.where(
                Event.type == EventType.INSTRUCTION_ACTIVATED
            )
            if not activation_query.exists():
                print(
                    f"Skipping instruction {instruction.id} because it was never activated."
                )
                continue
            activation = activation_query.get()

            cards_by_location = {card.location: card for card in initial_cards}
            card_events = game_events.where(
                Event.type << [EventType.CARD_SPAWN, EventType.CARD_SET],
                Event.server_time <= activation.server_time,
            ).order_by(Event.server_time)
            for event in card_events:
                if event.type == EventType.CARD_SPAWN:
                    card = Card.from_json(event.data)
                    cards_by_location[card.location] = card
                elif event.type == EventType.CARD_SET:
                    data_obj = json.loads(event.data)
                    cards = [Card.from_dict(card) for card in data_obj["cards"]]
                    for card in cards:
                        cards_by_location[card.location] = None
            props = [
                card.prop() for card in cards_by_location.values() if card is not None
            ]

            moves = instruction.children.where(Event.type == EventType.ACTION) #所有action事件,role='Role.FOLLOWER'
            roles = [move.role for move in moves] # "Role.FOLLOWER"
            feedbacks = (
                game_events.select()
                .where(
                    Event.parent_event << moves, Event.type == EventType.LIVE_FEEDBACK
                )
                .order_by(Event.server_time)
            )

            dt_string = instruction.server_time.strftime("%Y-%m-%d_%H-%M-%S")
            filepath = game_dir / f"instruction_vis_{dt_string}.png"
            instruction_obj = ObjectiveMessage.from_json(instruction.data)

            follower_file_path = game_dir #
            draw_follower_view(
                instruction_obj,
                moves, #所有action事件
                feedbacks, #所有反馈事件
                first_map_update, #第一个地图更新的数据
                follower_file_path,
                game.id,
                props, #只限card
                cfg,
                description=True,
                only_map=True,
            )

            instruction_list.append(instruction_obj.text)
            for word in instruction_obj.text.split(" "):
                words.add(word)
            if max_instructions == 0:
                break
            max_instructions -= 1
            break
        break
    # print how many instructions and unique words there are.
    print(f"{len(instruction_list)} instructions")
    print(f"{len(words)} unique words")


if __name__ == "__main__":
    fire.Fire(main)
