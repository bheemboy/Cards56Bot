# Cards56Bot

### Getting Started

1. Launch miniconda command prompt and create a new environment.

        conda create -n Cards56Bot python=3.8.3
        activate Cards56Bot
        conda install pytorch=1.5.1 torchvision=0.6.1 cpuonly -c pytorch
        pip install gym==0.17.2
        conda install swig
        pip install gym[box2d]
        pip install signalrcore==0.8.5
