import numpy as np


class GameStats:
    def __init__(self, stat_size):
        self.stat_size = stat_size

        self.last_bids = []
        self.last_wins = []
        self.last_rewards = []

        self.bidding_team = -1
        self.bidder_wins = []
        self.last_bidder_wins = []

        self.ave_bids = []
        self.ave_wins = []
        self.ave_rewards = []
        self.ave_bidder_wins = []

    def update(self, bids, wins, rewards):
        self.last_bids.append(bids)
        while len(self.last_bids) > self.stat_size:
            self.last_bids.pop(0)
        np_last_bids = np.array(self.last_bids)
        self.ave_bids = np_last_bids.sum(axis=0) / (np_last_bids != 0).sum(axis=0)

        self.last_wins.append(wins)
        while len(self.last_wins) > self.stat_size:
            self.last_wins.pop(0)
        self.ave_wins = np.sum(self.last_wins, axis=0)

        self.last_rewards.append(rewards)
        while len(self.last_rewards) > self.stat_size:
            self.last_rewards.pop(0)
        self.ave_rewards = np.average(self.last_rewards, axis=0)

        self.bidding_team = 0 if bids[0] > 0 else 1
        self.bidder_wins = [0, 0]
        if wins[self.bidding_team] > 0:
            self.bidder_wins[self.bidding_team] = 1
        self.last_bidder_wins.append(self.bidder_wins)
        while len(self.last_bidder_wins) > self.stat_size:
            self.last_bidder_wins.pop(0)
        self.ave_bidder_wins = np.sum(self.last_bidder_wins, axis=0) / (np_last_bids != 0).sum(axis=0)

