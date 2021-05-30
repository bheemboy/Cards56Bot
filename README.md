# Cards56Bot
#### Description
This is a deep reiforcement learning project that uses PPO and train a AI bot to play a game of cards called 56.

#### Setting up development environment

1. Clone this project
```
git clone gh repo clone bheemboy/Cards56Bot'
```

2. Install miniconda

3. Create a conda environment and install pre-requisites
 ```
cd Cards56Bot
conda create --name Cards56Bot --file requirements.txt
```

4. Install PyTorch
```
conda activate Cards56Bot
conda install pytorch-1.8.1 torchvision-0.9.1 torchaudio-0.8.1 cpuonly-1.0 -c pytorch-lts
```

5. Use PIP to install OpenAI GYM
```
pip install gym==0.18.3
pip install gym[box2d]
```

6. Install SignalRCore
```
pip install signalrcore==0.9.2
```
