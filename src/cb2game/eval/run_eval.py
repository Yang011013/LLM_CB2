import json
import logging
import time
import math
from datetime import datetime, timedelta
from typing import List
import os
from typing import Union, Tuple, Any
import fire
from tqdm import tqdm

from cb2game.agents.agent import Agent, RateLimitException, Role
from cb2game.agents.config import (
    AgentConfigData,
    AgentConfig,
    LoadAgentFromConfig,
    ReadAgentConfigOrDie,
    SerializeAgentConfig,
)
from cb2game.eval.eval_schema import Eval, InstructionEvaluation, RunSource
from cb2game.pyclient.endpoint_pair import EndpointPair
from cb2game.pyclient.local_game_coordinator import LocalGameCoordinator

from cb2game.server.config.config import Config, ReadServerConfigOrDie
from cb2game.server.db_tools.db_utils import ListGames, ListAnalysisGames
from cb2game.server.lobbies.open_lobby import OpenLobby
from cb2game.server.lobby_consts import LobbyInfo, LobbyType
from cb2game.server.messages.objective import ObjectiveMessage
from cb2game.server.messages.prop import PropType
from cb2game.server.messages.turn_state import TurnState
from cb2game.server.card import Card
from cb2game.server.scenario_util import (
    GameStateFromScenario,
    ReconstructScenarioFromEvent,
    GetSelectedCardsBetweenEvents,
)
from cb2game.server.schemas import base
from cb2game.server.schemas.event import Event, EventType
from cb2game.server.state_utils import (
    FOLLOWER_MOVES_PER_TURN,
    FOLLOWER_SECONDS_PER_TURN,
)
from cb2game.server.util import GetCommitHash, PackageVersion
from cb2game.agents.agent_utils import *
from cb2game.util.log_config import logger

# This is the default lobby name. Should equal the eval lobby defined in
# server/config/config.py. Must match the eval lobby on the remote server.
REMOTE_LOBBY_NAME = "eval-lobby"


def SwitchToDatabase(db):
    base.SetDatabaseByPath(db)
    base.ConnectDatabase()


def follower_eval_start(instruction: Event) -> Event:
    first_follower_move = (
        Event.select()
        .where(
            (Event.game == instruction.game) & (Event.type == EventType.ACTION) & (Event.parent_event_id == instruction.id)
        )
        .order_by(Event.server_time)
        .first()
    )
    if first_follower_move is None:
        logger.info("No follower move found.")
        return None
    event_before = (
        Event.select()
        .where(
            (Event.game == first_follower_move.game)
            & (Event.server_time < first_follower_move.server_time)
        )
        .order_by(Event.server_time.desc())
        .first()
    )
    return event_before


def final_follower_move(instruction: Event) -> Event:
    last_follower_move = (
        Event.select()
        .where(
            (Event.game == instruction.game) & (Event.type == EventType.ACTION) & (Event.parent_event_id == instruction.id)
        )
        .order_by(Event.server_time.desc())
        .first()
    )
    if last_follower_move is None:
        # No follower move found. Just get the INSTRUCTION_DONE event. Return null if
        # that doesn't exist either (instruction cancelled or game ended).
        instruction_complete_event = (
            Event.select()
            .where(
                (Event.type == EventType.INSTRUCTION_DONE)
                & (Event.parent_event_id == instruction.id)
            )
            .first()
        )
        return instruction_complete_event

    # Get the event after this one.
    event_after = (
        Event.select()
        .where(
            (Event.game == last_follower_move.game)
            & (Event.server_time > last_follower_move.server_time)
        )
        .order_by(Event.server_time)
        .first()
    )
    return event_after

def event_between_start_and_final(instruction: Event) -> Event:
    first_follower_move = (
        Event.select()
        .where(
            (Event.game == instruction.game) & (Event.type == EventType.ACTION) & (Event.parent_event_id == instruction.id)
        )
        .order_by(Event.server_time)
        .first()
    )
    last_follower_move = (
        Event.select()
        .where(
            (Event.game == instruction.game) & (Event.type == EventType.ACTION) & (Event.parent_event_id == instruction.id)
        )
        .order_by(Event.server_time.desc())
        .first()
    )
    between_moves = (
        Event.select()
        .where(
            (Event.type == EventType.CARD_SELECT) & (Event.role == Role.LEADER) &
            (Event.game == instruction.game) & (Event.server_time > first_follower_move.server_time) & (Event.server_time < last_follower_move.server_time)
        )
        .order_by(Event.server_time)
        .first()
    )
    return between_moves

def event_cancel(instruction: Event) -> bool:
    child_event = (
        Event.select()
        .where(
            (Event.game == instruction.game) & (Event.parent_event_id == instruction.id) & (Event.type == EventType.INSTRUCTION_CANCELLED)
        )
        .order_by(Event.server_time)
        .first()
    )
    # print((child_event is None))
    if child_event is not None:
        return True
    else:
        return False

def CompareCardSelections(a: List[Card], b: List[Card]) -> bool:
    selected_ids_a = set([card.id for card in a if card.selected])
    selected_ids_b = set([card.id for card in b if card.selected])
    return selected_ids_a == selected_ids_b
def CompareCards(a, b) -> bool:
    a = sorted(a)
    b = sorted(b)
    return a == b
def get_active_instruction(instructions):
    for instruction in instructions:
        if not instruction.completed and not instruction.cancelled:
            return instruction
    return None
def distance(loc1,loc2):
    return math.sqrt((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2)


def filter_cards(card_ids):
    if len(card_ids) == 0:
        return card_ids
    count = {}
    for number in card_ids:
        if number in count:
            count[number] += 1
        else:
            count[number] = 1
    result = [number for number in card_ids if count[number] % 2 != 0]
    return result
def RunEval(
        agent: Agent,
        output_prefix: str = "eval_",
        server_config_path: str = "",
        limit1: int = 0,
        limit2: int = 100,
        # Optional information about the agent that will be saved in the eval JSON output.
        agent_config: AgentConfig = None,
        up_score: int = 100,
        low_score: int = -1,
        instruction_length: int = -1,
):
    """Runs an eval against the given agent.

    Server configuration is required. This allows us to preserve the settings,
    software version, and lobby configuration that were used to collect the game
    data. Without this, an eval would be impossible to reproduce.

    Args:
        agent: The agent to run the eval against.
        output_prefix: The prefix to use for the output file.
        limit: The maximum number of instructions to evaluate. If -1, no limit.
        server_config_path: The path to the server config file.
    """
    follower_view_path = f"follower_view/follower_view_{low_score}_{up_score}.png"
    if server_config_path == "":
        config = Config()
        logger.warning(
            f"Server config path not provided. Using default config. Database path: {config.data_directory()}"
        )
    else:
        config = ReadServerConfigOrDie(server_config_path)


    base.SetDatabase(config)
    base.ConnectDatabase()

    with open("human_human_game_ids.txt", "r") as f:
        game_ids = f.readlines()
        game_ids = [int(game_id.strip()) for game_id in game_ids]
    # game_ids = [game.id for game in games]
    instructions = Event.select().where(
        (Event.type == EventType.INSTRUCTION_SENT) & (Event.game_id << game_ids)
    )  

    instructions = [instruction for instruction in instructions if
                    len(ObjectiveMessage.from_json(instruction.data).text.split()) > instruction_length]
    print(
        f"A total of {len(game_ids)} games({low_score} =< score =< {up_score}), and a total of {len(instructions)} instructions(length > {instruction_length})")

    instructions = instructions[limit1:limit2]

    if len(instructions) == 0:
        print("No instructions found.")
        return

    # Create an eval run entry in the database.
    eval_run = Eval(
        run_source=RunSource.LOCAL,
        commit_version=GetCommitHash() or PackageVersion(),
        agent_config=SerializeAgentConfig(agent_config),
        agent_role=agent.role(),
        server_config=config.to_json(),
    )
    # This object will help us launch local games.
    coordinator = LocalGameCoordinator(
        config,
        render_leader=True,
        render_follower=True,
    )

    eval_lobby = OpenLobby(
        LobbyInfo(
            name="eval virtual lobby",
            type=LobbyType.OPEN,
            comment="Ephemeral lobby used for eval runs.",
            game_capacity=1,
            sound_clip_volume=0,
        )
    )

    agent_instructions_passed = []
    results = []
    unmatched_instructions = {}
    selected_cards = {}
    skipped_instructions = []
    total_distance_loss_m = 0
    passed_distance_loss_m = 0
    unpassed_distance_loss_m = 0

    for i, instruction in enumerate(tqdm(instructions)):
        coordinator = LocalGameCoordinator(
            config,
            render_leader=True,
            render_follower=True,
        )
        try:
            objective = ObjectiveMessage.from_json(instruction.data)
            logger.info(f"-------------------------------------------------------------")

            logger.info(
                f"Evaluating agent {agent_config.agent_type} on instruction {instruction.id} and game_id = {instruction.game_id}"
            )
            logger.info(f"Instruction text: {objective.text}")
            if agent.role() == Role.LEADER:
                # Leader eval not yet supported.
                logger.info(f"Leader eval not yet supported.")
                return
            if event_cancel(instruction):
                logger.info("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                reason = "instruction cancelled"
                logger.info(f"skipping instruction: text={objective.text}, uuid={str(objective.uuid)} game_id={instruction.game_id},for {reason}")
                skipped_instructions.append(
                    {"instruction test": objective.text, "uuid": str(objective.uuid), "reason": reason,
                     "game_id": instruction.game_id})
                continue
            eval_start_event = follower_eval_start(instruction) 
            final_baseline_state = final_follower_move(instruction) 
            if eval_start_event is None or final_baseline_state is None:
                reason = "Skipping instruction. Invalid start or end states. This could be due to the instruction being cancelled or the game ending."
                logger.info(reason)
                skipped_instructions.append({"instruction test": objective.text, "uuid": str(objective.uuid), "reason": reason, "game_id": instruction.game_id})
                continue

            between_s_and_f_cards, card_selected_by_leader, card_selected_by_follower = GetSelectedCardsBetweenEvents(eval_start_event, final_baseline_state)
            between_s_and_f_cards = [card.id for card in between_s_and_f_cards] 

            card_selected_by_leader_id = [card.id for card in card_selected_by_leader]
            card_selected_by_follower_id = [card.id for card in card_selected_by_follower]

            if len(card_selected_by_follower_id) == 0:
                logger.info(f"**********Skipping instruction**********")
                reason = f"card_selected_by_follower_id={card_selected_by_follower_id} is empty. Follower didn't select any card."
                logger.info(reason)
                logger.info(between_s_and_f_cards)
                skipped_instructions.append({"instruction test": objective.text, "uuid": str(objective.uuid), "reason": reason, "game_id": instruction.game_id})
                continue

            # ---------------- start scenario and card ids
            start_scenario, err = ReconstructScenarioFromEvent(eval_start_event.id)
            start_baseline_state = GameStateFromScenario(start_scenario)
            start_baseline_props = start_baseline_state.props
            
            start_baseline_cards = [
                Card.FromProp(prop)
                for prop in start_baseline_props
                if prop.prop_type == PropType.CARD
            ]
            start_card_ids = [card.id for card in start_baseline_cards if card.selected]
            # ---------------- final scenario and card ids
            final_scenario, err = ReconstructScenarioFromEvent(final_baseline_state.id)
            final_baseline_state = GameStateFromScenario(final_scenario)
            ground_truth_follower_loc = [actor for actor in final_baseline_state.actors if actor.role().name == "FOLLOWER"][0].location().cartesian()
            final_baseline_props = final_baseline_state.props
            
            final_baseline_cards = [
                Card.FromProp(prop)
                for prop in final_baseline_props
                if prop.prop_type == PropType.CARD
            ]
            final_card_ids = [card.id for card in final_baseline_cards if card.selected]

            game_name = coordinator.CreateGameFromDatabase(
                eval_start_event.id.hex, log_to_db=False, lobby=eval_lobby
            )
            # Due to a known bug (now patched) where TURN_STATE events were not
            # being logged, we need to force the current turn state to be at the
            # beginning of the follower's turn, with full moves and time.
            state_machine = coordinator._state_machine_driver(
                game_name
            ).state_machine()  # pylint: disable=protected-access
            state_machine._send_turn_state(
                TurnState(  # pylint: disable=protected-access
                    Role.FOLLOWER,
                    FOLLOWER_MOVES_PER_TURN,
                    1,  # As long as next turn isn't game over.
                    datetime.utcnow() + timedelta(seconds=FOLLOWER_SECONDS_PER_TURN),
                    datetime.utcnow(),
                    0,  # Let's start each eval with a score of zero.
                    0,
                    False,
                    0,
                )
            )

            endpoint_pair = EndpointPair(coordinator, game_name)
            endpoint_pair.initialize()
            follower_game_state = endpoint_pair.initial_state()
            _, _, _, instrs, _, _ = follower_game_state
            if instrs[-1].text != objective.text:
                reason = "Skipping instruction. instrs[-1].text != objective.text. This could be due to the instruction being cancelled."
                logger.info(reason)
                skipped_instructions.append({"instruction test": objective.text, "uuid": str(objective.uuid), "reason": reason, "game_id": instruction.game_id})
                continue

            agent_cards = []
            agent_cards.extend(start_card_ids)

            if follower_game_state.turn_state.turn != agent.role():
                logger.error(
                    f"Agent role {agent.role()} does not match turn eval run state {follower_game_state.turn_state.turn}"
                )
                continue

            # Keep running until the current turn is over. We check for this inside
            # the loop because the game state may change in the middle of the loop.
            agent_actions = []
            responses = []
            card_states = []
            selected_card_types = []
            action_num = 0
            dataset = "human_human" if "human_human" in server_config_path else "train"
            image_folder = f"follower_view/{agent.model_name}/{dataset}/{instruction.game_id}/{objective.text}/"
            endpoint_pair.leader()._render()
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)


            while not endpoint_pair.over():
                # If the turn is over, then the eval for this instruction is done.
                if follower_game_state.turn_state.turn != agent.role():
                    break
                image_path = image_folder + f"{action_num}.png"
                endpoint_pair.leader().visualization().save_screen(image_path)
                if "baseline" in agent.model_name:
                    response, action = agent.choose_action(game_state=follower_game_state,
                                                            action_mask=endpoint_pair.follower().action_mask(), test=True)
                    game_state, follower_game_state = endpoint_pair.step(action)
                else:
                    response, action = agent.choose_action(follower_game_state, endpoint_pair.follower(),
                                                            follower_view_path=image_path, test=True)
                    game_state, follower_game_state = endpoint_pair.step(action)
                responses.append(response)
                action_num += 1

                selected_card_props = [Card.FromProp(prop) for prop in game_state.props if
                                       prop.prop_type == PropType.CARD]
                selected_card_ids = [card.id for card in selected_card_props if card.selected]
                selected_card_type = [(card.shape, card.color, card.count) for card in selected_card_props if
                                      card.selected]
                card_states.append(selected_card_ids)
                selected_card_types.append(selected_card_type)

                agent_actions.append(str(action))
                if "INSTRUCTION_DONE" in str(action):
                    break

            agent_selected_cards_in_game = [card_id for (card_id, selected) in state_machine.agent_selected_cards]

            logger.info(f"Agent actions: {agent_actions}")
            logger.info(f"card states after each action: {card_states}")
            logger.log(logging.INFO, f"\n selected_card_type after each action: {selected_card_types}")

            agent_follower_loc = [actor for actor in follower_game_state.actors if actor.role().name == "FOLLOWER"][0].location().cartesian()
            distance_loss = distance(ground_truth_follower_loc, agent_follower_loc)
            logger.log(logging.INFO, f"\nground truth follower loc: {ground_truth_follower_loc}\nagent follower loc: {agent_follower_loc}\ndistance loss: {distance_loss}")


            selected_cards[str(instruction.id)] = [instruction.game_id, objective.text, final_card_ids]
            agent_cards = agent_selected_cards_in_game + start_card_ids + card_selected_by_leader_id
            agent_cards = filter_cards(agent_cards)
            ground_truth_cards = start_card_ids + between_s_and_f_cards
            ground_truth_cards = filter_cards(ground_truth_cards)
            cards_match = CompareCards(agent_cards, ground_truth_cards)

            logger.log(logging.INFO,
                       f"\n Compare Cards: \nstart_card_ids={start_card_ids}\nground_truth_cards={ground_truth_cards}\nagent_cards={agent_cards}")
            logger.log(logging.INFO,
                       f"\n agent_selected_cards_in_game={agent_selected_cards_in_game}\ncard_selected_by_leader_id={card_selected_by_leader_id}\nbetween_s_and_f_cards={between_s_and_f_cards}")
            logger.log(logging.INFO,
                       f"\ncards_match={cards_match}")

            if not cards_match:
                unmatched_instructions[str(instruction.id)] = [instruction.game_id, objective.text, responses,
                                                               agent_actions]

            print(f"---------------cards_match={cards_match}----------------")         

            if cards_match:
                agent_instructions_passed.append(instruction.id)
                passed_distance_loss_m += distance_loss
            else:
                unpassed_distance_loss_m += distance_loss
            total_distance_loss_m += distance_loss
            
            results.append(
                InstructionEvaluation(
                    instruction_uuid=instruction.short_code,
                    agent_actions=str(agent_actions),
                    event_uuid=eval_start_event.id.hex,
                    success=cards_match,
                )
            )
        except RuntimeError as e:
            # Log the exception, with stack trace and instruction ID.
            logger.error(
                f"Runtime error in eval run {eval_run.id} for instruction {objective.text}."
            )
            logger.error(e, exc_info=True)
            continue
        except RateLimitException:
            logger.info(f"Rate limit error. Waiting 60 seconds.")
            time.sleep(60)
            continue
        except Exception as e:
            # Log the exception, with stack trace and instruction ID.
            logger.error(
                f"Exception in eval run {eval_run.id} for instruction {instruction.id}."
            )
            logger.error(e, exc_info=True)
            continue
        if i % 2 == 0 and i != 0:
            if len(agent_instructions_passed) > 0 and len(results) > 0 and (len(results)-len(agent_instructions_passed))>0:
                logger.info(
                    f"total instruction: {i + 1}. count of passed instructions:{len(agent_instructions_passed)}. ({100 * len(agent_instructions_passed) / len(results)}%)")
                print(
                    f"total instruction: {i + 1}. count of passed instructions:{len(agent_instructions_passed)}. ({100 * len(agent_instructions_passed) / len(results)}%)")

                print(f"total mean distance loss={total_distance_loss_m/len(results)}")
                print(f"passed mean distance loss={passed_distance_loss_m/len(agent_instructions_passed)}")
                print(f"unpassed mean distance loss={unpassed_distance_loss_m/(len(results)-len(agent_instructions_passed))}")
    eval_run.percent_passed = (
        (100 * len(agent_instructions_passed) / len(results)) if len(results) > 0 else 0
    )
    eval_run.total_instructions = len(results)
    eval_run.instruction_evals = results

    logger.info(f"Eval run {eval_run.id} complete.")
    logger.info(
        f"A total of {len(game_ids)} games({low_score} =< score =< {up_score}), and a total of {len(instructions)} instructions[{limit1}:{limit2}]")
    if len(results) > 0 and len(agent_instructions_passed) > 0:
        logger.info(
            f"Instructions passed: {len(agent_instructions_passed)}. ({100 * len(agent_instructions_passed) / len(results)}%)"
        )
        logger.info(f"total mean distance_loss={total_distance_loss_m/len(results)} of {len(results)} total instructions")
        logger.info(f"passed mean distance_loss={passed_distance_loss_m/len(agent_instructions_passed)} of {len(agent_instructions_passed)} passed instructions")
        logger.info(f"unpassed mean distance_loss={unpassed_distance_loss_m/(len(results)-len(agent_instructions_passed))} of {len(results)-len(agent_instructions_passed)} unpassed instructions")
        logger.info(f"Total instructions: {len(results)}")


def main(
        agent_config_path: str,
        output_prefix: str = "eval_",
        server_config: str = "",
        limit1: int = 0,
        limit2: int = 100,
        up_score: int = 100,
        low_score: int = -1,
        instruction_length: int = -1,
):
    agent_config_data = ReadAgentConfigOrDie(agent_config_path)
    agent = LoadAgentFromConfig(agent_config_data)
    RunEval(
        agent,
        output_prefix,
        server_config,
        limit1,
        limit2,
        agent_config_data,
        up_score,
        low_score,
        instruction_length,
    )

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time of evaluation: {elapsed_time} seconds")