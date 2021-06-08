from CardPlayer import *
from PPO import PPO, Memory
import torch
from GameStats import GameStats


def train_model():
    ############## Hyperparameters ##############
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    k_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################
    table_type = 0
    train_ppo = 1
    n_players = 4 + 2*table_type
    layer_dims = [[STATE_DIM[table_type], 64, 64, ACTION_DIM],
                  [STATE_DIM[table_type], 128, ACTION_DIM]]

    memory = Memory()
    policy_files = [f'Card56Bot-{dim}-{i}.pth' for i, dim in enumerate(layer_dims)]
    temp_files = [f'Temp-{dim}-{i}.pth' for i, dim in enumerate(layer_dims)]
    ppos = [PPO(dim, lr, betas, gamma, k_epochs, eps_clip) for dim in layer_dims]

    players_temp = [CardPlayer(id=i, table_type=table_type, policy=ppos[i % 2].policy_old) for i in range(n_players)]
    # rearrange players based on their table position
    players = [None, None, None, None]
    for p in players_temp:
        players[int(p.player_position)] = p

    for player_num, p in enumerate(players):
        p.policy = ppos[player_num % 2].policy_old

    n_total_games = 20000
    n_update_every = 25
    target_rewards = 100
    stat_size = 200
    better_policy_not_found_count = 0
    loop_num = 0

    while better_policy_not_found_count < 2:
        loop_num += 1
        stats = GameStats(stat_size)
        better_policy_found = False

        print(f"Start training model [{train_ppo}] - {layer_dims[train_ppo]}...")
        for i, ppo in enumerate(ppos):
            ppo.load_policy_from_file(saved_file=policy_files[i])

        # try to find a better policy
        for game_num in range(1, n_total_games+1):
            # play one game on multiple threads
            game_cancelled, bids, wins, rewards = play_one_game(game_num=game_num, players=players)
            while game_cancelled:   # try again if game was cancelled
                game_cancelled, bids, wins, rewards = play_one_game(game_num=game_num, players=players)

            # update stats from the played game
            stats.update(bids, wins, rewards)

            # Report results
            bid_str = f"{bids[0]}" if bids[0] > 0 else "  "
            bid_str += f"{'*' if stats.bidder_wins[0] > 0 else ' '}:"
            bid_str += f"{bids[1]}" if bids[1] > 0 else "  "
            bid_str += f"{'*' if stats.bidder_wins[1] > 0 else ' '}"
            print(f"[{better_policy_not_found_count}.{loop_num}.{game_num}]:\t"
                  f"Bid[{bid_str}],\t"
                  f"Ave Bids[{stats.ave_bids[0]:0.0f},{stats.ave_bids[1]:0.0f}],\t"
                  f"Bid Wins[{stats.ave_bidder_wins[0]:0.0%},{stats.ave_bidder_wins[1]:0.0%}],\t"
                  f"Wins[{stats.ave_wins[0]:0.0f},{stats.ave_wins[1]:0.0f}],\t"
                  f"Rewards[{stats.ave_rewards[0]:0.0f},{stats.ave_rewards[1]:0.0f}]")

            # if the other team has been beaten save the model and break out of the for loop
            if len(stats.last_rewards) >= stat_size and \
                    stats.ave_rewards[train_ppo] >= stats.ave_rewards[(train_ppo+1) % 2] + target_rewards:
                more_rewards = stats.ave_rewards[train_ppo] - stats.ave_rewards[(train_ppo+1) % 2]
                print(f"Better policy with {more_rewards:0.0f} more rewards found.\t"
                      f"Average bid=[{stats.ave_bids[0]: 0.0f},{stats.ave_bids[1]: 0.0f}]")
                better_policy_found = True
                torch.save(ppos[train_ppo].policy.state_dict(), policy_files[train_ppo])
                train_ppo = (train_ppo + 1) % 2
                break

            # collect memories from the team being trained
            for player_num, p in enumerate(players):
                if player_num % 2 == train_ppo:
                    memory.extend(states=p.memory.states, actions=p.memory.actions, logprobs=p.memory.logprobs,
                                  rewards=p.memory.rewards, dones=p.memory.is_terminals)

            # update the model every x number of games
            if game_num % n_update_every == 0:
                print(f"Updating model [{train_ppo}] - {layer_dims[train_ppo]}")
                ppos[train_ppo].update(memory)
                memory.clear_memory()
                torch.save(ppos[train_ppo].policy.state_dict(), temp_files[train_ppo])

        # end for game_num in range(1, n_total_games+1):
        if better_policy_found:
            better_policy_not_found_count = 0
        else:
            better_policy_not_found_count += 1

    # end while better_policy_not_found_count < 2:
    print(f"=============== Training completed ===============")


if __name__ == '__main__':
    train_model()
