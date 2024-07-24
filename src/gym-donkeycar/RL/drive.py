import gym
import gym_donkeycar
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
from callbacks import LapTimeCallback, CheckpointCallback, LearningRateCallback
from wrappers import AutoencoderWrapper, HistoryWrapper
from rewards import Reward
from torch import nn
import pickle
import cv2
import os
from datetime import datetime
import yaml
from prettytable import PrettyTable


def test_model(model, env, num_tests = 10, save_images = [False]):
    if save_images[0]:
        now = datetime.now()
        unique_id = now.strftime("%Y%m%d%H%M%S")
        frame_skip = 2
        render = True
        
        frames = save_images[1]
        output_folder = save_images[2]
        obs = env.reset()
        for i in range(frames):
            action, _ = model.predict(obs, deterministic=True)
            for _ in range(frame_skip):
                obs, _, done, info = env.step(action)
                if done:
                    break
            if render:
                env.render()
            img = info[0]["image"]
            path = os.path.join(output_folder, f"{unique_id+i}.jpg")
            # Convert to BGR
            cv2.imwrite(path, img[:, :, ::-1])

        obs = env.reset()
    else:
        print(f"Trying {num_tests} laps!")
        times = []
        laps = 0
        obs = env.reset()
        while laps < num_tests:
            done = False
            ep_laps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                lap_time = info[0]['last_lap_time']
                episode_laps = info[0]['lap_count']
                if episode_laps > ep_laps:
                    print("LAP COMPLETED!")
                    ep_laps = episode_laps
                    times.append(lap_time)
                    laps = laps + 1
                    if laps >= num_tests:
                        break
                if done:
                    laps = laps + 1

        avg_time = sum(times)/len(times) if len(times)>0 else "No lap has been completed!"
        print(f"Done testing! Times: {times} \n Average time: {avg_time}")
        obs = env.reset()


def read_config(config_path):
    # we read the config of the run
    with open(config_path, "r") as f:
        config_yaml = yaml.safe_load(f)
        print("Configuration read:\n" + yaml.dump(config_yaml, default_flow_style=False))
        model_name = config_yaml["model_name"]
        config = config_yaml["config"]
        config["model_path"] = os.path.join(config["model_path"], model_name)
        config["checkpoints"][1] = os.path.join(config["checkpoints"][1], model_name)
        config["checkpoints"][2] = os.path.join(config["checkpoints"][2], model_name)
        conf = config_yaml["conf"]
        hyperparameters = config_yaml["hyperparameters"]
        hyperparameters["policy_kwargs"] = eval(hyperparameters["policy_kwargs"])
        return config, conf, hyperparameters


def create_env(config, conf, center_line):
    # we read the center track line points
    with open(center_line, "rb") as fp:
        center_line = pickle.load(fp)
    
    def make_env():
        env = gym.make(config["env_name"], conf=conf)
        r = Reward(center_line)
        env.set_reward_fn(r.time_sector_reward)
        env = Monitor(env)  # record stats such as returns
        if config["use_autoencoder"]:
            env = AutoencoderWrapper(env, ae_path=config["ae_path"])
        if config["use_history"]:
            env = HistoryWrapper(env, horizon=config["horizon"])
        return env

    env = DummyVecEnv([make_env])
    if config["normalize"]:
        vec_env_path = os.path.join(config["load_model"][4],config["load_model"][1])+".pkl"
        if config["load_model"][0] and os.path.exists(vec_env_path):
            print("Loading saved VecNormalize stats")
            env = VecNormalize.load(vec_env_path, env)
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.0, clip_reward=100.0)
    return env


def count_parameters(model): # function to print information about a model
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
        
        
if __name__ == "__main__":
    config, conf, hyperparameters = read_config("./config.yaml")
    env = create_env(config, conf, "center_line/center_line_1452")
                
    test = True
    if test:
        model = PPO.load(os.path.join(config["load_model"][2],config["load_model"][1]), env=env)
        print("Model Policy:\n", model.policy, sep="")
        print("Model Policy Action Distribution:\n", model.policy.action_dist, sep="")
        print("Model Parameters:")
        count_parameters(model.policy)

        test_model(model, env, num_tests=5, save_images = config["save_images"])
    else:
        run = wandb.init(
            project="testing",
            config=dict(config=config,hyperparameters=hyperparameters,conf=conf), # merge of the three dictionaries
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

        model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}", **hyperparameters)
        if config["load_model"][0]:
            print("Loading pretrained agent")

            # if "policy_kwargs" in hyperparameters.keys():
            #     del hyperparameters["policy_kwargs"]

            model = PPO.load(os.path.join(config["load_model"][2],config["load_model"][1]), env=env, verbose=1, tensorboard_log=f"runs/{run.id}", **hyperparameters)
            replay_buffer_path = os.path.join(config["load_model"][3],config["load_model"][1])+".pkl"
            if os.path.exists(replay_buffer_path):
                print("Loading replay buffer")
                model.load_replay_buffer(replay_buffer_path)
            
        print(model.policy)
        print(model.policy.action_dist)
        callbacks = [WandbCallback(gradient_save_freq=100,verbose=2,),
                    LapTimeCallback(), 
                    CheckpointCallback(save_freq=10000,
                                        save_path=config["model_path"],
                                        save_replay_buffer=config["checkpoints"][0],
                                        replay_buffer_path=config["checkpoints"][1],
                                        normalize=config["normalize"],
                                        normalize_path=config["checkpoints"][2]),]
                    #LearningRateCallback()]
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
        )

        test_model(model, env)
        model.save(config["model_path"])
        run.finish()

    env.close()