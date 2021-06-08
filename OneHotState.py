import numpy as np
import json


class OneHot:
    def __init__(self, state):
        self.one_hot = []
        self.MAX_PLAYERS = state['TableInfo']['MaxPlayers']
        if self.MAX_PLAYERS == 4:
            self.RANKS = [10, 1, 9, 11]
        else:
            self.RANKS = [12, 13, 10, 1, 9, 11]
        self.CARDS_PER_PLAYER = 8
        self.SUITS = ['c', 'd', 's', 'h']
        self.DECK_SIZE = self.MAX_PLAYERS * self.CARDS_PER_PLAYER
        self.ONE_HOT_CARD_SIZE = len(self.SUITS) + len(self.RANKS)
        self.GAME_STAGES = [2, 3, 4]  # Game stages of interest = Bidding=2, SelectingTrump=3, PlayingCards=4

        self.player_position = state['PlayerPosition']
        # state will be relative to the players position
        # so all other players' positions will be adjusted.

        # game stage
        self.game_stage_idx = len(self.one_hot)
        game_stage = state['GameStage']
        one_hot_game_stage = [1 if game_stage == stage else 0 for stage in self.GAME_STAGES]
        self.one_hot = np.concatenate((self.one_hot, one_hot_game_stage))

        # trump card
        self.trump_card_idx = len(self.one_hot)
        trump_card = state['TrumpCard']
        one_hot_card = self.get_one_hot_suit(trump_card)
        self.one_hot = np.concatenate((self.one_hot, one_hot_card))

        # my cards
        self.player_cards_idx = len(self.one_hot)
        player_cards = state['PlayerCards']
        one_hot_cards = self.get_one_hot_cards(player_cards)
        self.one_hot = np.concatenate((self.one_hot, one_hot_cards))

        # Current Round
        self.current_round_idx = len(self.one_hot)
        rounds = state['TableInfo']['Rounds']
        played_cards = []
        if len(rounds) > 0:
            played_cards = rounds[-1]["PlayedCards"]    # Get last round
        one_hot_current_round = self.get_current_round(played_cards)
        self.one_hot = np.concatenate((self.one_hot, one_hot_current_round))

        # cards played so far
        self.cards_played_idx = len(self.one_hot)
        rounds = state['TableInfo']['Rounds']
        one_cards_played = self.get_cards_played(rounds)
        self.one_hot = np.concatenate((self.one_hot, one_cards_played))

    def get_card_suit_and_rank(self, card):
        suit = self.SUITS.index(card[:1])
        rank = self.RANKS.index(int(card[1:]))
        return suit, rank

    def get_one_hot_suit(self, card):
        retval = np.zeros(len(self.SUITS))
        if len(card) > 0:
            suit, _ = self.get_card_suit_and_rank(card)
            retval[suit] = 1
        return retval

    def set_card_in_deck(self, deck, card):
        suit, rank = self.get_card_suit_and_rank(card)
        card_index = suit * len(self.RANKS) * 2 + rank * 2
        if deck[card_index] == 0:
            deck[card_index] = 1
        else:
            deck[card_index + 1] = 1
        return deck

    def get_one_hot_cards(self, cards):
        one_hot_cards = np.zeros(self.DECK_SIZE)
        for card in cards:
            one_hot_cards = self.set_card_in_deck(one_hot_cards, card)
        return one_hot_cards

    def get_current_round(self, played_cards):
        # one card is represented as 4 bit suit [len(self.SUITS)] + 4 (or 6) bit rank [len(self.RANKS)]
        # there are a maximum of 3 (or 5) cards [self.MAX_PLAYERS-1]
        one_hot_current_round = np.zeros((len(self.SUITS)+len(self.RANKS)) * (self.MAX_PLAYERS-1))
        for i, card in enumerate(played_cards):
            suit, rank = self.get_card_suit_and_rank(card)
            # The previous payer always gets the last spot regardless of how many plays have happened
            card_offset = ((self.MAX_PLAYERS-1)-(len(played_cards))+i) * (len(self.SUITS)+len(self.RANKS))
            one_hot_current_round[card_offset+suit] = 1
            one_hot_current_round[card_offset+len(self.SUITS)+rank] = 1
        return one_hot_current_round

    def get_cards_played(self, rounds):
        one_hot_get_cards_played = np.zeros(self.DECK_SIZE)
        for r in rounds:
            for card in r["PlayedCards"]:
                one_hot_get_cards_played = self.set_card_in_deck(one_hot_get_cards_played, card)
        return one_hot_get_cards_played

    def get_cards_from_on_hot(self, one_hot_deck):
        cards = ""
        for card_idx, card in enumerate(one_hot_deck):
            if card == 1:
                idx = card_idx % self.DECK_SIZE
                suit = int(idx / (len(self.RANKS)*2))
                rank = int((idx % (len(self.RANKS)*2)) / 2)
                cards += f"{self.SUITS[suit]}{self.RANKS[rank]},"
        return cards

    def print_player_cards(self):
        one_hot_player_cards = self.one_hot[self.player_cards_idx:self.player_cards_idx+self.DECK_SIZE]
        print(f"Player cards={self.get_cards_from_on_hot(one_hot_player_cards)}")

    def print_played_cards(self):
        one_hot_played_cards = self.one_hot[self.cards_played_idx:self.cards_played_idx+self.DECK_SIZE]
        print(f"Played cards={self.get_cards_from_on_hot(one_hot_played_cards)}")


if __name__ == '__main__':
    with open('data/4PlayerGameState.json') as json_file:
        data = json.load(json_file)
        player4_one_hot = OneHot(data)
        print(f"4 Player one hot size = {len(player4_one_hot.one_hot)}")
        player4_one_hot.print_player_cards()
        player4_one_hot.print_played_cards()

    with open('data/6PlayerGameState.json') as json_file:
        data = json.load(json_file)
        player6_one_hot = OneHot(data)
        print(f"6 Player one hot size = {len(player6_one_hot.one_hot)}")
        player6_one_hot.print_player_cards()
        player6_one_hot.print_played_cards()


