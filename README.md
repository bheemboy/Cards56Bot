# Cards56Bot
#### Description
This is a deep reiforcement learning project that uses PPO to train an AI bot to play 56 cards game.

#### Setting up development environment

Clone this project
```
git clone gh repo clone bheemboy/Cards56Bot
```

Install miniconda

Create a conda environment and install pre-requisites
 ```
cd Cards56Bot
conda create --name Cards56Bot --file requirements.txt
```

Install PyTorch
```
conda activate Cards56Bot
conda install pytorch-1.8.1 torchvision-0.9.1 torchaudio-0.8.1 cpuonly-1.0 -c pytorch
```

Use PIP to install OpenAI GYM
```
pip install gym==0.18.3
pip install gym[box2d]
```

Install SignalRCore
```
pip install signalrcore==0.9.2
```
