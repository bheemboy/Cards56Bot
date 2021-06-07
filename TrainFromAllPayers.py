from CardPlayer import *
from PPO import PPO, Memory
import torch
from GameStats import GameStats
import numpy as np


def main():
    ############## Hyperparameters ##############
    lr = 0.0001
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    k_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    table_type = 0

    memory = Memory()
    layer_dims = [STATE_DIM[table_type], 128, 64, ACTION_DIM]
    ppo = PPO(layer_dims, lr, betas, gamma, k_epochs, eps_clip)

    file_name = 'Card56Bot.pth'
    temp_file = f'Temp-{layer_dims}.pth'
    ppo.load_policy_from_file(temp_file)

    n_players = 4
    n_total_games = 100000
    n_update_every = 100
    best_ave_bidder_wins = 0.5
    stat_size = 100
    stats = GameStats(stat_size)

    players = [CardPlayer(id=i, table_type=table_type, policy=ppo.policy_old) for i in range(n_players)]
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
              f"Reward[{rewards[0]:0.0f},{rewards[1]:0.0f}],\t"
              f"Ave Bids[{stats.ave_bids[0]:0.0f},{stats.ave_bids[1]:0.0f}],\t"
              f"Wins[{stats.ave_bidder_wins[0]:0.0%},{stats.ave_bidder_wins[1]:0.0%}]")

        # save model if bidder wins more than previous winner
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
