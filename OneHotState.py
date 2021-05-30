import numpy as np
import json


class OneHot:
    def __init__(self, state):
        self.MAX_PLAYERS = state['TableInfo']['MaxPlayers']
        if self.MAX_PLAYERS == 4:
            self.RANKS = [10, 1, 9, 11]
        else:
            self.RANKS = [12, 13, 10, 1, 9, 11]
        self.CARDS_PER_PLAYER = 8
        self.SUITS = ['c', 'd', 's', 'h']
        self.DECK_SIZE = self.MAX_PLAYERS * self.CARDS_PER_PLAYER
        self.ONE_HOT_CARD_SIZE = len(self.SUITS) + 1
        self.GAME_STAGES = [2, 3, 4]  # Game stages of interest = Bidding=2, SelectingTrump=3, PlayingCards=4

        self.player_position = state['PlayerPosition']
        # state will be relative to the players position
        # so all other players' positions will be adjusted.

        # dealer position
        self.dealer_position_idx = 0
        dealer_position = state['TableInfo']['DealerPos']
        dealer_position = (dealer_position - self.player_position) % self.MAX_PLAYERS
        one_hot_dealer_position = [1 if i == dealer_position else 0 for i in range(self.MAX_PLAYERS)]
        self.one_hot = one_hot_dealer_position

        # game stage
        self.game_stage_idx = len(self.one_hot)
        one_hot_game_stage = np.zeros(len(self.GAME_STAGES))
        game_stage = state['GameStage']
        if game_stage in self.GAME_STAGES:
            one_hot_game_stage[self.GAME_STAGES.index(game_stage)] = 1
        self.one_hot = np.concatenate((self.one_hot, one_hot_game_stage))

        # trump exposed
        self.trump_exposed_idx = len(self.one_hot)
        trump_exposed = state['TrumpExposed']
        self.one_hot = np.concatenate((self.one_hot, [1 if trump_exposed else 0]))

        # high bidder
        self.high_bidder_idx = len(self.one_hot)
        high_bidder = state['TableInfo']['Bid']['HighBidder']
        high_bidder = (high_bidder - self.player_position) % self.MAX_PLAYERS
        one_hot_high_bidder = [1 if i == high_bidder else 0 for i in range(self.MAX_PLAYERS)]
        self.one_hot = np.concatenate((self.one_hot, one_hot_high_bidder))

        # trump card
        self.trump_card_idx = len(self.one_hot)
        trump_card = state['TrumpCard']
        one_hot_card = self.get_one_hot_card(trump_card)
        self.one_hot = np.concatenate((self.one_hot, one_hot_card))

        # my cards
        self.player_cards_idx = len(self.one_hot)
        player_cards = state['PlayerCards']
        one_hot_cards = self.get_one_hot_cards(player_cards)
        self.one_hot = np.concatenate((self.one_hot, one_hot_cards))

        # highest bids
        self.highest_bids_idx = len(self.one_hot)
        bid_history = state['TableInfo']['Bid']['BidHistory']
        one_hot_highest_bids = self.get_one_hot_highest_bids(bid_history)
        self.one_hot = np.concatenate((self.one_hot, one_hot_highest_bids))

        # next_min_bid
        self.next_min_bid_idx = len(self.one_hot)
        next_min_bid = state['TableInfo']['Bid']['NextMinBid']
        self.one_hot = np.concatenate((self.one_hot, [next_min_bid]))

        # Current Round Suit
        self.current_round_suit_idx = len(self.one_hot)
        current_round_suit = state['CurrentRoundSuit']
        one_hot_current_round_suit = np.zeros(len(self.SUITS))
        try:
            suit = self.SUITS.index(current_round_suit)
            one_hot_current_round_suit[suit] = 1
        except ValueError:
            pass
        self.one_hot = np.concatenate((self.one_hot, one_hot_current_round_suit))

        # played rounds
        self.rounds_idx = len(self.one_hot)
        rounds = state['TableInfo']['Rounds']
        one_hot_rounds = self.get_one_hot_rounds(rounds)
        self.one_hot = np.concatenate((self.one_hot, one_hot_rounds))

    def get_card_suit_and_rank(self, card):
        suit = self.SUITS.index(card[:1])
        rank = self.RANKS.index(int(card[1:]))
        return suit, rank

    def get_one_hot_card(self, card):
        retval = np.zeros(self.ONE_HOT_CARD_SIZE)
        if len(card) > 0:
            suit, rank = self.get_card_suit_and_rank(card)
            retval[suit] = 1
            retval[self.ONE_HOT_CARD_SIZE-1] = rank
        return retval

    def get_one_hot_cards(self, cards):
        retval = np.zeros(self.DECK_SIZE)
        for card in cards:
            suit, rank = self.get_card_suit_and_rank(card)
            card_index = suit*len(self.RANKS) + rank
            if retval[card_index] == 0:
                retval[card_index] = 1
            else:
                retval[card_index+int(self.DECK_SIZE/2)] = 1
        return retval

    def get_one_hot_highest_bids(self, bid_history):
        one_hot_highest_bids = np.zeros(self.MAX_PLAYERS)
        for item in bid_history:
            position = item['Position']
            position = (position - self.player_position) % self.MAX_PLAYERS
            bid = item['Bid']
            if bid > one_hot_highest_bids[position]:
                one_hot_highest_bids[position] = bid
        return one_hot_highest_bids

    def get_one_hot_rounds(self, rounds):
        one_hot_trump_exposed = np.zeros(self.DECK_SIZE)
        one_hot_round_cards = np.zeros(self.DECK_SIZE*self.ONE_HOT_CARD_SIZE)

        for round_idx, round_info in enumerate(rounds):
            first_player = round_info["FirstPlayer"]
            trump_exposed = round_info["TrumpExposed"]
            # Cards played by this player (player_position) will be at position zero.
            player_offset = (first_player-self.player_position) % self.MAX_PLAYERS
            round_offset = round_idx * self.MAX_PLAYERS
            for trump_exposed_idx, val in enumerate(trump_exposed):
                trump_exposed_offset = (trump_exposed_idx+player_offset) % self.MAX_PLAYERS
                one_hot_trump_exposed_index = round_offset + trump_exposed_offset
                one_hot_trump_exposed[one_hot_trump_exposed_index] = 1 if val else 0

            played_cards = round_info["PlayedCards"]
            round_offset = round_idx * self.ONE_HOT_CARD_SIZE * self.MAX_PLAYERS
            for card_idx, card in enumerate(played_cards):
                card_offset = ((card_idx+player_offset) % self.MAX_PLAYERS) * self.ONE_HOT_CARD_SIZE
                one_hot_round_cards_index = round_offset + card_offset
                suit, rank = self.get_card_suit_and_rank(card)
                one_hot_round_cards[one_hot_round_cards_index + suit] = 1
                one_hot_round_cards[one_hot_round_cards_index + self.ONE_HOT_CARD_SIZE-1] = rank

        return np.concatenate((one_hot_trump_exposed, one_hot_round_cards))

    def print_player_cards(self):
        player_cards = ""
        one_hot_player_cards = self.one_hot[self.player_cards_idx:self.player_cards_idx+self.DECK_SIZE]
        for card_idx, card in enumerate(one_hot_player_cards):
            if card == 1:
                idx = card_idx % self.DECK_SIZE
                suit = int(idx / len(self.RANKS))
                rank = idx % len(self.RANKS)
                player_cards += f"{self.SUITS[suit]}{self.RANKS[rank]},"
        print(f"player cards={player_cards}")

    def print_rounds(self):
        # print played round cards
        one_hot_trump_exposed = self.one_hot[self.rounds_idx:
                                             self.rounds_idx+self.DECK_SIZE]
        one_hot_round_cards = self.one_hot[self.rounds_idx+self.DECK_SIZE:
                                           self.rounds_idx+self.DECK_SIZE+self.DECK_SIZE*self.ONE_HOT_CARD_SIZE]
        cards = ""
        for card_num in range(self.DECK_SIZE):
            card_start_idx = card_num*self.ONE_HOT_CARD_SIZE
            one_hot_card = one_hot_round_cards[card_start_idx: card_start_idx+self.ONE_HOT_CARD_SIZE]
            card = "--"
            for i in range(len(self.SUITS)):
                if one_hot_card[i] == 1:
                    suit = self.SUITS[i]
                    rank = self.RANKS[int(one_hot_card[self.ONE_HOT_CARD_SIZE - 1])]
                    card = f"{suit}{rank}"
                    break
            cards += f"{card},"
        print(f"played rounds={cards}")


if __name__ == '__main__':
    with open('test/4PlayerGameState.json') as json_file:
        data = json.load(json_file)
        player4_one_hot = OneHot(data)
        player4_one_hot.print_player_cards()
        player4_one_hot.print_rounds()

    with open('test/6PlayerGameState.json') as json_file:
        data = json.load(json_file)
        player4_one_hot = OneHot(data)
        player4_one_hot.print_player_cards()
        player4_one_hot.print_rounds()


