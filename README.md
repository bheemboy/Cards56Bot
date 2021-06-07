# Cards56Bot
#### Description
This is a deep reiforcement learning project that uses PPO to train an AI bot to play 56 cards game.

#### Setting up development environment

1. Install miniconda from https://docs.conda.io/

2. Install Build-Essentials
```bash
sudo apt install build-essential
```
3. Clone this project
```bash
git clone https://github.com/bheemboy/Cards56Bot.git
```

4. Create a conda environment and install pre-requisites
 ```bash
cd Cards56Bot
conda create -n Cards56Bot python=3.8
conda install swig
```

5. Install PyTorch
```bash
conda activate Cards56Bot
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

6. Install OpenAI gym
```bash
pip install gym
pip install gym[box2d]
```

7. Install SignalRCore
```bash
pip install signalrcore
```

NOTE: The project uses SERVER_URL in CardPlayer.py as the address of the game server. 
The default value assumes that you are running a local copy of https://github.com/bheemboy/Cards56 on port 5051. If your environment is different you need to update its value. 
