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

def test_model(model, env, log = False, num_tests = 10, save_images = [False]):
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
            path = os.path.join(output_folder, f"{90000+i}.jpg")
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
                # if action[0][1] < 0.5:
                #      print("A")
                obs, reward, done, info = env.step(action)
                #print(reward)
                lap_time = info[0]['last_lap_time']
                episode_laps = info[0]['lap_count']
                if episode_laps > ep_laps:
                    print("LAP DONE!")
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

model_name = "ppo_report"
loaded_model_name = "ppo_report_best"
normalize = True
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_name": "donkey-generated-track-v0",
    "model_path": os.path.join("models_report", model_name),
    "use_autoencoder": True,
    "ae_path": "./autoencoders/ae-32_100k.pkl",
    "use_history": True,
    "horizon": 2,
    "save_images": [False,15000,"autoencoders/test_images/"], # [if save images, num of steps/frames, output folder]
    "normalize": normalize,
    "load_model": [True, loaded_model_name, "models_report", "replay_buffers", "vec_envs"], # [if loading a model, model name, model folder, replay buffer folder, vec env folder]
    "checkpoints": [False, os.path.join("replay_buffers", model_name), normalize, os.path.join("vec_envs", model_name)] # [save_replay_buffer, save_replay_buffer path, normalize, vec_env path]
}

conf = { # configuration of the environment
    "body_style": "donkey", # body_style = "donkey" | "bare" | "car01" | "f1" | "cybertruck"
    "body_rgb": (247, 152, 29), # body_rgb  = (128, 128, 128) tuple of ints
    "car_name": "", # car_name = "string less than 64 char"
    "font_size": 40,
    "steer_limit": 0.5, # default was 1.0, but it is better to change it, that much is not necessary
    "throttle_min": -0.2, # default is 0.0
    "throttle_max": 1.5, # default is 1.0
}

hyperparameters = {
    "learning_rate": 5e-5, # default 3e-4
    "n_steps": 2048, # default 2048 (rollout buffer size)
    "batch_size": 64, # default 64
    "n_epochs": 10, # default 10
    "gamma": 0.99, # default 0.99
    "clip_range": 0.2, # default 0.2
    "ent_coef": 0.0, # default 0.0
    "vf_coef": 0.5, # default 0.5
    #"policy_kwargs": dict(log_std_init=0.0, net_arch=[dict(pi=[256, 256], vf=[256, 256])], activation_fn=nn.Tanh), # default dict(log_std_init=0.0, net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=nn.Tanh)
    "policy_kwargs": dict(log_std_init=0.0, net_arch=[dict(pi=[256, 256], vf=[256, 256])], activation_fn=nn.Tanh, # default dict(log_std_init=0.0, net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=nn.Tanh)
                         optimizer_kwargs=dict(betas=(0.9,0.999), weight_decay=0)),  # default dict(betas=(0.9,0.999), weight_decay=0.0))
}

# we read the center track line points
with open("center_line/center_line_1452", "rb") as fp:
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

            
test = True
if test:
    #model = PPO.load("models/ppo_sectors_variable_lr_350k")
    model = PPO.load(os.path.join(config["load_model"][2],config["load_model"][1]), env=env)
    print(model.policy)
    print(model.policy.action_dist)
    
    # from prettytable import PrettyTable
    # def count_parameters(model):
    #     table = PrettyTable(["Modules", "Parameters"])
    #     total_params = 0
    #     for name, parameter in model.named_parameters():
    #         if not parameter.requires_grad: continue
    #         params = parameter.numel()
    #         table.add_row([name, params])
    #         total_params+=params
    #     print(table)
    #     print(f"Total Trainable Params: {total_params}")
    #     return total_params
    # print(count_parameters(model.policy))

    test_model(model, env, log=False, num_tests=5, save_images = config["save_images"])

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
                                    normalize=config["checkpoints"][2],
                                    normalize_path=config["checkpoints"][3]),]
                 #LearningRateCallback()]
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
    )

    test_model(model, env, log=True)
    model.save(config["model_path"])
    run.finish()

env.close()