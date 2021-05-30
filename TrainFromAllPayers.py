from CardPlayer import *
from PPO import PPO, Memory
import torch
from GameStats import GameStats
import numpy as np


def main():
    ############## Hyperparameters ##############
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    k_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    memory = Memory()
    layer_dims = [STATE_DIM4, 128, ACTION_DIM]
    ppo = PPO(layer_dims, lr, betas, gamma, k_epochs, eps_clip)

    file_name = 'Card56Bot.pth'
    temp_file = f'Temp-{layer_dims}.pth'
    ppo.load_policy_from_file(temp_file)

    n_players = 4
    n_total_games = 15000
    n_update_every = 25
    best_ave_bidder_wins = 0.55
    stat_size = 100
    stats = GameStats(stat_size)

    players = [CardPlayer(id=i, policy=ppo.policy_old) for i in range(n_players)]
    for game_num in range(1, n_total_games+1):
        # play one game on multiple threads
        game_cancelled, bids, wins, rewards = play_one_game(game_num=game_num, players=players)
        while game_cancelled:  # try again if game was cancelled
            game_cancelled, bids, wins, rewards = play_one_game(game_num=game_num, players=players)

        # update stats from the played game
        stats.update(bids, wins, rewards)

        # Report results
        bid_str = f"{bids[0]}" if bids[0] > 0 else "  "
        bid_str += f"{'*' if stats.bidder_wins[0] > 0 else ' '}:"
        bid_str += f"{bids[1]}" if bids[1] > 0 else "  "
        bid_str += f"{'*' if stats.bidder_wins[1] > 0 else ' '}"
        print(f"[{game_num}]:\t"
              f"Bid[{bid_str}],\t"
              f"Ave Bids[{stats.ave_bids[0]:0.0f},{stats.ave_bids[1]:0.0f}],\t"
              f"Bid Wins[{stats.ave_bidder_wins[0]:0.0%},{stats.ave_bidder_wins[1]:0.0%}],\t"
              f"Wins[{stats.ave_wins[0]:0.0f},{stats.ave_wins[1]:0.0f}],\t"
              f"Rewards[{stats.ave_rewards[0]:0.0f},{stats.ave_rewards[1]:0.0f}]")

        # stop training if bidder wins 55% of the last 100 games
        if len(stats.last_wins) >= stat_size and np.average(stats.ave_bidder_wins) >= best_ave_bidder_wins:
            best_ave_bidder_wins = np.ave(stats.ave_bidder_wins)
            print(f"Save model with {best_ave_bidder_wins:0.0%} wins.")
            torch.save(ppo.policy.state_dict(), file_name)

        # collect memories from the teams
        for p in players:
            memory.extend(states=p.memory.states, actions=p.memory.actions, logprobs=p.memory.logprobs,
                          rewards=p.memory.rewards, dones=p.memory.is_terminals)

        # update the model every x number of games
        if game_num % n_update_every == 0:
            print(f"Updating model")
            ppo.update(memory)
            memory.clear_memory()
            torch.save(ppo.policy.state_dict(), temp_file)

    print(f"=============== Training completed ===============")


if __name__ == '__main__':
    main()
