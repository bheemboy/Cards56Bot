from CardPlayer import *
from PPO import PPO


def play_game():
    ############## Hyperparameters ##############
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    k_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################
    table_type = 0
    layer_dims = [STATE_DIM[table_type], 64, 64, ACTION_DIM]
    ppo = PPO(layer_dims, lr, betas, gamma, k_epochs, eps_clip)

    temp_file = f'Card56Bot-{layer_dims}.pth'
    ppo.load_policy_from_file(temp_file)

    n_players = 2
    number_of_games = 1

    players = [CardPlayer(id=i, table_type=table_type, policy=ppo.policy_old, print_moves=True) for i in range(n_players)]
    for game_num in range(number_of_games):
        play_one_game(game_num=game_num, players=players)


if __name__ == '__main__':
    play_game()
