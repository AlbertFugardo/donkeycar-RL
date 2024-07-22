# Autonomous Racing with Reinforcement Learning
The objective of this project is using Reinforcement Learning models and engineering reward functions for the task of autonomous racing around a track. The simulator used is Donkey Car.
![alt text](./images/donkeyFramework.png)

## Results
After training a PPO algorithm (with a custom reward engineered by us) for 600k steps, the car was able to complete laps around the track at a high speed. <br>
A video of the behaviour of the car throughout the training steps can be found here:

The paper in which the details of the project are presented can be found in:

## Quick start
In this section there are the necessary steps for the installation, testing and running of the models.
### Installation
To install all the necessary packages, it is recommended the creation of a virtual environment. To do so, just execute from a terminal:

```bash
python3 -m venv donkey_env
source donkey_env/bin/activate
```
If you want to exit the venv, just do:
```bash
deactivate
```

To install the packages (it takes quite a while):
```
pip install -r requirements.txt
```

Moreover, the simulator interface is needed to be able to run and train models. This interface runs in Unity. <br>
From https://github.com/tawnkramer/gym-donkeycar/releases, download and unzip ``DonkeySimLinux.zip``, ``DonkeySimMac.zip`` and ``DonkeySimWin.zip`` depending on your operating system (and change the permissions when necessary, as stated in the link). <br>
To check that the interface has been downloaded properly, open the app, and something similar to the following image should appear:
![alt text](./images/donkeysim.png)

### Variational Autoencoder
In the development of this project, we have seen that using a VAE to encode image observations is much better for the learning process of the model (that's seen in more detail in our paper). <br>
The VAE architecture and code that we have used in this project has been obtained from: https://github.com/araffin/aae-train-donkeycar/tree/live-twitch-2 <br>
If you want to use our trained VAEs, you can find them in ``/src/gym-donkeycar/RL/autoencoders/``. For example, you can ``use ae-32_100k.pkl``, which is an autoencoder trained using 100k images thoroughly collected driving the car.

### Training a model

### Testing a model
You can find one model in ``/src/gym-donkeycar/RL/models_test/ppo_trained_650k.zip``, a model trained for 650k steps.

## Future Work
The main restriction found in this project was the amount of computational power needed to train and run the models, which we didn't have.

## Credits

pip install -e <path> per instal·lar els paquets com en local.
-e means: Install a project in editable mode (i.e. setuptools “develop mode”) from a local project path or a VCS url.
https://pip.pypa.io/en/stable/cli/pip_install/
