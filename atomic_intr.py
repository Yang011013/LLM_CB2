# 无视地图中的properties，只考虑follower和target的位置关系，返回automic instruction
from src.cb2game.server.hex import HecsCoord, HexBoundary, HexCell
from collections import deque

def get_edge_between(a, b):
    # a-->b; b-->a.nagete()
    displacement = HecsCoord(b.a ^ a.negate().a,
                             b.r + a.negate().r + (b.a & a.negate().a),
                             b.c + a.negate().c + (b.a & a.negate().a))
    edge = HexBoundary.DIR_TO_EDGE.get(displacement, None)
    if edge is None:
        raise ValueError(
            f"HecsCoords {a}, {b} passed to set_edge_between are not adjacent."
        )
    return edge

def find_path_to_card(target_location: HecsCoord, follower_loaction: HecsCoord):
    start_location = follower_loaction
    end_location = target_location
    location_queue = deque()
    location_queue.append((start_location, [start_location]))
    visited_locations = set()
    while len(location_queue) > 0:
        current_location, current_path = location_queue.popleft()
        if current_location in visited_locations:
            continue
        visited_locations.add(current_location)
        if current_location == end_location:
            return current_path
        current_neighbors = current_location.neighbors()
        for neighbor in current_neighbors:
            location_queue.append((neighbor, current_path + [neighbor]))
    return None


def get_instruction_to_location(
    target_location: HecsCoord,# (a,r,c)
    follower_location: HecsCoord,
    follower_orientation,
    map,
    cards,
    game_endpoint=None,
    default_instruction="random, random, random, random, random, random",
):

    path = find_path_to_card(target_location, follower_location)
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
            instructions.append("left")
            instructions.append("backward")
            heading -= 60
            location = next_location
            continue
        if degrees_away == -120:
            instructions.append("right")
            instructions.append("backward")
            heading += 60
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


def DescribeLocationFromActor(location: HecsCoord, follower_location: HecsCoord, follower_orientation, map, cards) -> str:
    """Returns a string describing the given location from the perspective of the given actor."""
    if follower_location == location:
        return "You are standing here."
    default_instruction = "<Cannot reach>"
    instruction = get_instruction_to_location(
        location, follower_location, follower_orientation, map, cards, default_instruction=default_instruction
    )
    return f"automic instruction: {instruction}"

def main(target_location, follower_location, follower_orientation, map, cards):
    print(DescribeLocationFromActor(target_location, follower_location, follower_orientation, map, cards))

if __name__ == '__main__':
    follower_location = HecsCoord(a=1, r=8, c=9)
    follower_orientation = 0
    # target_location = HecsCoord(a=0, r=7, c=13)
    target_location = HecsCoord(a=1, r=5, c=8)
    main(target_location, follower_location, follower_orientation, None, None)