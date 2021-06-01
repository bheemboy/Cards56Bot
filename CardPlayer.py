import threading
import json
from signalrcore.hub_connection_builder import HubConnectionBuilder
from enum import IntEnum
from OneHotState import OneHot
from PPO import Memory
import numpy as np
from torch.distributions import Categorical
import torch


SERVER_URL = "https://localhost/Cards56Hub"


class PlayerState(IntEnum):
    NOT_CONNECTED = 0
    CONNECTED = 1
    REGISTERED = 2
    JOINED_TABLE = 3


class GameStage(IntEnum):
    WAITING_FOR_PLAYERS = 1
    BIDDING = 2
    SELECT_TRUMP = 3
    PLAY_CARD = 4
    GAME_OVER = 5


class GameAction(IntEnum):
    PASS = 0
    BID_28 = PASS + 1
    BID_57 = PASS + 30          # Should be 30 to enable thani
    TRUMP_CARD_1 = BID_57 + 1
    TRUMP_CARD_8 = BID_57 + 8
    PLAY_CARD_1 = TRUMP_CARD_8 + 1
    PLAY_CARD_8 = TRUMP_CARD_8 + 8
    SHOW_TRUMP = PLAY_CARD_8 + 1
    MAX_ACTIONS = SHOW_TRUMP + 1


class ServerMethod(IntEnum):
    REGISTERPLAYER = 1
    JOINTABLE = 2
    PLACEBID = 3
    PASSBID = 4
    SELECTTRUMP = 5
    PLAYCARD = 6
    SHOWTRUMP = 7
    STARTNEXTGAME = 8


STATE_DIM4 = 250
STATE_DIM6 = 368
ACTION_DIM = int(GameAction.MAX_ACTIONS)  # 48
REWARD_FOR_MISTAKE = 0
WINNING_SCORE_MULTIPLIER = 100
INCLUDE_LAST_ROUND_REWARD = True
INCLUDE_BAD_MOVES = True


class CardPlayer:
    def __init__(self, id, policy, print_moves=False):
        self.id = id
        self.name = f"BOT-{self.id}"
        self.policy = policy
        self.print_moves = print_moves
        self.lang = "en-US"
        self.watch_only = False
        self.memory = Memory()

        self.player_state = PlayerState.NOT_CONNECTED

        self.connected_event = threading.Event()
        self.registered_event = threading.Event()
        self.joined_table = threading.Event()
        self.table_full_event = threading.Event()
        self.turn_to_play = threading.Event()
        self.take_action_complete = threading.Event()

        self.game_started_event = threading.Event()
        self.game_started_event.clear()
        self.game_over_event = threading.Event()
        self.game_over_event.clear()
        self.json_state = ""
        self.player_position = -1
        self.high_bid = -1
        self.high_bidder = -1
        self.winning_team = -1
        self.last_played_round = -1
        self.game_cancelled = False
        self.last_json_state = ""

        self._hub_connection: HubConnectionBuilder = HubConnectionBuilder()\
            .with_url(SERVER_URL, options={"verify_ssl": False})\
            .with_automatic_reconnect({
                    "type": "raw",
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                    "max_attempts": 5
                })\
            .build()

        # Register websockets events
        self._hub_connection.on_open(self._on_connected)
        self._hub_connection.on_close(self._on_close)

        self._hub_connection.on("OnError", self._on_error)
        self._hub_connection.on("OnRegisterPlayerCompleted", self._on_register_player_completed)
        self._hub_connection.on("OnStateUpdated", self._on_state_updated)

        self.connected_event.clear()
        self._hub_connection.start()
        if self.connected_event.wait(10):
            self._register_player()
            self._join_table(0, "")
        else:
            print(f"Player {self.name} could not connect.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._hub_connection.stop()
        except:
            print(f"Player {self.name} error in __exit__()")

    def __del__(self):
        try:
            self._hub_connection.stop()
        except:
            print(f"Player {self.name} error in __del__()")

    def _on_connected(self):
        # print(f"Player {self.name} connected")
        self.player_state = PlayerState.CONNECTED
        self.connected_event.set()

    def _on_close(self):
        # print(f"Player {self.name} connection closed")
        self.player_state = PlayerState.NOT_CONNECTED

    def _register_player(self):
        if self.player_state == PlayerState.CONNECTED:
            self.registered_event.clear()
            self._hub_connection.send("RegisterPlayer", [self.name, self.lang, self.watch_only])
            if not self.registered_event.wait(10):
                print(f"Could not register player {self.name}")
        else:
            print(f"Cannot register {self.name}. Current state is {self.player_state}")

    def _on_register_player_completed(self, data):
        # print(f"Player {self.name} registered")
        self.player_state = PlayerState.REGISTERED
        self.registered_event.set()

    def _join_table(self, table_type, table_name):
        if self.player_state == PlayerState.REGISTERED:
            self.joined_table.clear()
            self._hub_connection.send("JoinTable", [table_type, table_name])
            if not self.joined_table.wait(10):
                print(f"Player {self.name} could not join table")
        else:
            print(f"Player {self.name} cannot join table. Current state is {self.player_state}")

    def _start_next_game(self):
        # print(f"Player {self.name} start_next_game")
        while not self.table_full_event.is_set():
            self.table_full_event.wait(1)

        if self.id == 0:
            self.game_started_event.wait(0.1)
            if not self.game_started_event.is_set():
                self._hub_connection.send("StartNextGame", [])

        # wait for game to start
        self.game_started_event.wait(2)
        while not self.game_started_event.is_set():
            self._hub_connection.send("StartNextGame", [])
            self.game_started_event.wait(1)

    def _game_started(self):
        self.game_started_event.set()
        self.game_over_event.clear()

    def _game_over(self):
        self.game_started_event.clear()
        self.game_over_event.set()

    def _on_error(self, data):
        error_code = data[0]
        method_id = data[1]
        error_message = data[2]
        error_data = data[3]
        self._print_moves(f"Player {self.name} Error {error_code} in method {method_id}: {error_message}")
        if ServerMethod.PLACEBID <= method_id <= ServerMethod.SHOWTRUMP:
            if self.memory.count() > 0:
                if INCLUDE_BAD_MOVES:
                    self.memory.rewards[self.memory.count() - 1] = REWARD_FOR_MISTAKE
                else:
                    self.memory.pop()
            self._on_state_updated([self.json_state])

    def _on_state_updated(self, data):
        self.player_state = PlayerState.JOINED_TABLE
        self.joined_table.set()
        self.turn_to_play.clear()
        self.json_state = data[0]
        if self.json_state:
            # determine if it is this players turn to play
            state = json.loads(self.json_state)
            self.player_position = state['PlayerPosition']
            self.high_bid = state['TableInfo']['Bid']['HighBid']
            self.high_bidder = state['TableInfo']['Bid']['HighBidder']
            if state['TableFull']:
                self.table_full_event.set()
                # GameStage {Unknown=0, WaitingForPlayers=1, Bidding=2, SelectingTrump=3, PlayingCards=4, GameOver=5}
                game_stage = state['GameStage']
                if game_stage == GameStage.BIDDING or game_stage == GameStage.SELECT_TRUMP:
                    self._game_started()
                    if self.player_position == state['TableInfo']['Bid']['NextBidder']:
                        # It is player turn to play
                        # print(f"Player {self.name} turn to play set")
                        self.turn_to_play.set()
                elif game_stage == GameStage.PLAY_CARD:
                    self._game_started()
                    self._update_last_round_reward(state)
                    current_round = len(state['TableInfo']['Rounds']) - 1
                    if self.player_position == state['TableInfo']['Rounds'][current_round]['NextPlayer']:
                        # It is player turn to play
                        self.turn_to_play.set()

                elif game_stage == GameStage.GAME_OVER:
                    self._update_last_round_reward(state)
                    self.game_cancelled = state['TableInfo']['GameCancelled']
                    self._game_over()
                elif game_stage == GameStage.WAITING_FOR_PLAYERS:
                    pass
                else:
                    print(f"Unexpected game_stage in _on_state_updated? {game_stage}")
            else:      # Cannot play
                self.table_full_event.clear()

        self.take_action_complete.set()

    def play_game(self, i):
        self.memory.clear_memory()
        self.last_played_round = -1
        self._start_next_game()

        while not self.game_over_event.is_set():
            # wait for the players turn
            if self.turn_to_play.wait(1):
                self.turn_to_play.clear()
                state = json.loads(self.json_state)
                action = self._get_action(state)
                self._take_action(state, action)

        self._update_rewards()

    def _get_next_action(self):
        # get an action that is not in self.selected_actions
        action = torch.tensor(self.selectable_actions[self.selectable_actions_idx])
        self.selectable_actions_idx += 1
        return action, Categorical(self.action_probs).log_prob(action)

    def _get_action(self, state):
        if self.json_state != self.last_json_state:
            self.last_json_state = self.json_state
            state_one_hot = OneHot(state).one_hot
            self.state_tensor, self.action_probs = self.policy.act(state_one_hot)
            self.selectable_actions = np.random.choice(range(GameAction.MAX_ACTIONS),
                                                       GameAction.MAX_ACTIONS,
                                                       replace=False,
                                                       p=self.action_probs.detach().numpy())
            self.selectable_actions_idx = 0
        # end if

        action, log_probs = self._get_next_action()

        while not self._is_valid_action(state, action.item()):
            if INCLUDE_BAD_MOVES:
                self.memory.append(state=self.state_tensor, action=action, logprobs=log_probs,
                                   reward=REWARD_FOR_MISTAKE, done=False)
            action, log_probs = self._get_next_action()

        self.memory.append(state=self.state_tensor, action=action, logprobs=log_probs, reward=0, done=False)

        return action.item()

    def _take_action(self, state, action):
        self.take_action_complete.clear()
        if action == GameAction.PASS:
            self._print_moves(f"Player {self.name} Passing")
            self._hub_connection.send("PassBid", [])
            self.take_action_complete.wait()
        elif GameAction.BID_28 <= action <= GameAction.BID_57:
            bid = action - GameAction.BID_28 + 28
            self._print_moves(f"Player {self.name} Bidding {bid} with {state['PlayerCards']}")
            self._hub_connection.send("PlaceBid", [bid])
            self.take_action_complete.wait()
        elif GameAction.TRUMP_CARD_1 <= action <= GameAction.TRUMP_CARD_8:    # Select one of my cards as trump
            total_cards_count = len(state['PlayerCards'])
            selected_card_num = action-GameAction.TRUMP_CARD_1
            card = ""
            if selected_card_num < total_cards_count:
                card = state['PlayerCards'][selected_card_num]
            self._print_moves(f"Player {self.name} Selecting Trump {card} from {state['PlayerCards']}")
            self._hub_connection.send("SelectTrump", [card])
            self.take_action_complete.wait()
        elif GameAction.PLAY_CARD_1 <= action <= GameAction.PLAY_CARD_8:    # Play one of my cards
            total_cards_count = len(state['PlayerCards'])
            selected_card_num = action-GameAction.PLAY_CARD_1
            card = ""
            if selected_card_num < total_cards_count:
                card = state['PlayerCards'][selected_card_num]
            self.last_played_round = len(state['TableInfo']['Rounds'])-1
            self._print_moves(f"Player {self.name} Playing Card {card} "
                              f"in round {self.last_played_round} from {state['PlayerCards']}")
            self._hub_connection.send("PlayCard", [card, 0])
            self.take_action_complete.wait()
        elif action == GameAction.SHOW_TRUMP:
            self._print_moves(f"Player {self.name} ShowTrump")
            self._hub_connection.send("ShowTrump", [0])
            self.take_action_complete.wait()
        else:
            print(f"Unexpected action in _take_action? {action}")

    @staticmethod
    def _is_valid_action(state, action):
        game_stage = state['GameStage']
        if game_stage == GameStage.BIDDING:
            if GameAction.PASS == action:
                if state['TableInfo']['Bid']['NextMinBid'] > 28:
                    return True
            elif GameAction.BID_28 <= action <= GameAction.BID_57:
                bid = action - GameAction.BID_28 + 28
                min_bid = state['TableInfo']['Bid']['NextMinBid']
                if bid >= min_bid:
                    return True
        elif game_stage == GameStage.SELECT_TRUMP:
            if GameAction.TRUMP_CARD_1 <= action <= GameAction.TRUMP_CARD_8:
                return True
        elif game_stage == GameStage.PLAY_CARD:
            if GameAction.PLAY_CARD_1 <= action <= GameAction.PLAY_CARD_8 or action == GameAction.SHOW_TRUMP:
                total_cards_count = len(state['PlayerCards'])
                selected_card_num = action - GameAction.PLAY_CARD_1
                if selected_card_num < total_cards_count:
                    return True
        else:
            print(f"Unexpected game_stage in is_valid_action {game_stage}")
        return False

    def _update_last_round_reward(self, state):
        if not INCLUDE_LAST_ROUND_REWARD:
            return
        if self.last_played_round < 0 or self.memory.count() <= 0:
            return

        if self.memory.rewards[self.memory.count() - 1] == 0:  # successful play
            if len(state['TableInfo']['Rounds']) > self.last_played_round:
                game_round = state['TableInfo']['Rounds'][self.last_played_round]
                winner = game_round["Winner"]
                score = game_round["Score"]
                if (winner % 2) == (self.player_position % 2):
                    self.memory.rewards[self.memory.count() - 1] = score

    def _update_rewards(self):
        state = json.loads(self.json_state)
        if self.memory.count() > 0:
            self.winning_team = state['TableInfo']['WinningTeam']
            winning_score = state['TableInfo']['WinningScore'] * WINNING_SCORE_MULTIPLIER
            if (self.player_position % 2) == self.winning_team:
                self.memory.rewards[self.memory.count() - 1] += winning_score
            else:
                self.memory.rewards[self.memory.count() - 1] += -winning_score
            self.memory.is_terminals[self.memory.count()-1] = True

    def _print_moves(self, msg):
        if self.print_moves:
            print(msg)


def play_one_game(game_num, players):
    threads = [threading.Thread(target=p.play_game, args=(game_num,)) for p in players]
    for t in threads:
        t.daemon = True
        t.start()
    for t in threads:
        t.join()

    # grab the results
    bids = [0, 0]
    wins = [0, 0]
    rewards = [0, 0]
    game_cancelled = True
    for player_num, p in enumerate(players):
        game_cancelled = p.game_cancelled
        if game_cancelled:
            break

        # find player who has a high bid value
        if p.high_bid != -1:  # disregard players who did not participate in the game at all (Thani game)
            bids[p.high_bidder % 2] = p.high_bid
            wins[p.winning_team] = 1
            rewards[player_num % 2] += np.sum(p.memory.rewards)
    return game_cancelled, bids, wins, rewards
