#!/usr/bin/env python3
import isaacgym

assert isaacgym
import torch
import numpy as np
import imageio # For videos
    
import glob
import pickle as pkl
import os.path

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

from tqdm import tqdm

from scripts.config_env import config_env

import yaml

import wandb

import os
import imageio
from PIL import Image, ImageDraw, ImageFont

print("Lets play!")

def get_logdir(pretrain=False):
    label = "gait-conditioned-agility/%s/train" % ( "pretrain-v0" if pretrain else "2*")

    if os.path.isdir("./.git"):
        dirs = glob.glob(f"./runs/{label}/*")
    else:
        dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[-1]  # [0]
    print("selected run: %s" % logdir)
    return logdir


def load_policy(logdir, step='latest'):
    print("LOADING POLICY FROM " + logdir)
    run_id = os.listdir(logdir + '/checkpoints')[0]
    body = torch.jit.load(logdir + f'/checkpoints/{run_id}/body_{step}.jit')
    adaptation_module = torch.jit.load(logdir + f'/checkpoints/{run_id}/adaptation_module_{step}.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(logdir, headless=False):
    # logdir = get_logdir()
    print("----------------LOADING ENV FROM parameters.pkl---------------------")
    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    # our part
    # config_env(Cfg) # THIS IS WHERE YOU NEED TO ADD MALFUNCTIONS

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg) 
    env = HistoryWrapper(env)

    # load policy
    # from ml_logger import logger
    # from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy

def get_video_env(env):
    import copy
    video_cfg = copy.deepcopy(env.cfg)
    
    # turn off DR for evaluation script
    video_cfg.domain_rand.push_robots = False
    video_cfg.domain_rand.randomize_friction = False
    video_cfg.domain_rand.randomize_gravity = False
    video_cfg.domain_rand.randomize_restitution = False
    video_cfg.domain_rand.randomize_motor_offset = False
    video_cfg.domain_rand.randomize_motor_strength = False
    video_cfg.domain_rand.randomize_friction_indep = False
    video_cfg.domain_rand.randomize_ground_friction = False
    video_cfg.domain_rand.randomize_base_mass = False
    video_cfg.domain_rand.randomize_Kd_factor = False
    video_cfg.domain_rand.randomize_Kp_factor = False
    video_cfg.domain_rand.randomize_joint_friction = False
    video_cfg.domain_rand.randomize_com_displacement = False

    video_cfg.env.num_recording_envs = 1
    video_cfg.env.num_envs = 1
    video_cfg.terrain.num_rows = 5
    video_cfg.terrain.num_cols = 5
    video_cfg.terrain.border_size = 0
    video_cfg.terrain.center_robots = True
    video_cfg.terrain.center_span = 1
    video_cfg.terrain.teleport_robots = True

    video_cfg.domain_rand.lag_timesteps = 6
    video_cfg.domain_rand.randomize_lag_timesteps = True
    video_cfg.control.control_type = "actuator_net"

    video_env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=video_cfg)
    video_env = HistoryWrapper(video_env)

    return video_env

def load_env_from_yaml(logdir, headless=False):
    print("----------------LOADING ENV FROM YAML---------------------")
    # from go1_gym.envs.base.legged_robot_config import Cfg
    from copy import deepcopy
    # local_Cfg = deepcopy(Cfg)
    with open(logdir + "/config.yaml", 'r') as file:
        yaml_cfg = yaml.safe_load(file)
        print(yaml_cfg.keys())
        try:
            cfg = yaml_cfg["Cfg"]['value']
        except KeyError:
            cfg = yaml_cfg["Cfg"]

        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    # print(key2)
                    # print(getattr(Cfg, key))
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    # our part
    config_env(Cfg) # THIS IS WHERE YOU NEED TO ADD MALFUNCTIONS

    # from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    # from ml_logger import logger
    # from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    # actor_critic = ActorCritic(env.num_obs,
    #                            env.num_privileged_obs,
    #                            env.num_obs_history,
    #                            env.num_actions).to('cuda:0')
    
    # actor_critic.load_state_dict(torch.load(logdir + '/checkpoints/ac_weights_last.pt'))

    # policy = load_policy(logdir, step)
    # policy = actor_critic.act_inference

    return env


def get_play_frames(env, policy, num_eval_steps=450):
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()

    frames = []
    
    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()

        img = env.render(mode="rgb_array")
        frames.append(np.array(img))  # Store the frame
        # if i % 300 == 0:
        #     env.reset()

    return frames


def download_run_files(run, download_dir='wandb'):
    
    # Ensure download directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    # Download config 
    config = run.config
    config_path = os.path.join(download_dir, "config.yaml")
    # Save config as YAML
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
        print(f"Configuration for run {run_id} saved to {config_path}")

    # Download files
    print("Downloading files from wandb run...")
    for file in run.files():
        file.download(root=download_dir, replace=True)
    
    print(f"Downloaded files to: {download_dir}")
    return download_dir


def add_caption_to_frames(frames, caption):
    """
    Adds a caption to each frame in the video.
    """
    font = ImageFont.load_default()  # Load a default font
    captioned_frames = []
    
    for frame in frames:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        
        # Use ImageFont.getbbox to calculate text dimensions
        text_bbox = font.getbbox(caption)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        
        x = (image.width - text_width) // 2
        y = image.height - text_height - 10  # Place text near the bottom
        
        draw.text((x, y), caption, fill="black", font=font)
        captioned_frames.append(np.array(image))
    
    return captioned_frames



def play_go1_from_files(run_id, steps='latest', output_video_name=f'play_video_combined.mp4', headless=True):
    """
    Create a video for one or multiple steps, stitching them together with captions.
    """
    # Define the download directory for the run
    download_dir = f'../wandb/{run_id}'
    api = wandb.Api()
    run = api.run(f"zoharmilman/robot-training/{run_id}")  

    # Check if the directory exists; if not, download files from wandb
    if not os.path.exists(download_dir):
        print(f"Directory {download_dir} not found. Downloading files...")
        download_run_files(run, download_dir=download_dir)
    else:
        print(f"Directory {download_dir} already exists. Skipping download.")

    # Ensure steps is a list for uniform processing
    if isinstance(steps, str) or isinstance(steps, int):
        steps = [steps]

    all_frames = []
    env = load_env_from_yaml(download_dir, headless=headless)

    for step in steps:
        print(f"Processing step: {step}")
        # Load environment and policy for the current step
        
        try: 
            policy = load_policy(download_dir, step)
            # Generate frames and add caption
            env.reset()
            frames = get_play_frames(env, policy)
            caption = f"Run id: {run_id}, Step: {step}"
            frames_with_caption = add_caption_to_frames(frames, caption)

            # Append frames to the overall list
            all_frames.extend(frames_with_caption)
        except ValueError:
            print('-----Tried to load a non-existant step------')

    # Save stitched video in the media directory
    media_dir = os.path.join(download_dir, 'media')
    os.makedirs(media_dir, exist_ok=True)
    output_filename = os.path.join(media_dir, output_video_name)
    imageio.mimsave(output_filename, all_frames, fps=30)
    print("Saved video to: " + output_filename)



def play_go1(env, policy, num_eval_steps=900, headless=False):
    # Get a video enviorment with DR turned off
    video_env = get_video_env(env)
    frames = get_play_frames(video_env, policy, num_eval_steps=num_eval_steps)

    return frames

# if __name__ == '__main__':
#     # to see the environment rendering, set headless=False
#     play_go1_from_files(logdir="wandb/run-20241103_191407-fn5iwsaf/files", step='000999', headless=False)



if __name__ == '__main__':
    # vye0j20m - From nothing with nothing 1 
    # 3h1exxz0 - From nothing with nothing 2
    # wy1pnf38 - From nothing with nothing 3
    # dfs31hyt - From nothing with nothing 4
    # xxoral21 - From nothing with nothing 5

    # bu20t6v5 - from nothing with only pos rewards 1
    # 8l5sx020 - from nothing with only pos rewards 2
    # c65ybabb - from nothing with only pos rewards 3
    # jz8kdqr7 - from nothing with only pos rewards 4

    # 958kz3g9 - From nothing with -1 raribert heuristic scale 1
    # k1zkcjlg - From nothing with -1 raribert heuristic scale 2
    # kwwxvace - From nothing with -1 raribert heuristic scale 3 
    # lw46f0dh - From nothing with -1 raribert heuristic scale 4

    # lpu7s93c - From nothing with fault on 2 std=1 mean=1 1
    # m47r5l7u - From nothing with fault on 2 std=1 mean=1 2
    # x4mw4i4y - From nothing with fault on 2 std=1 mean=1 3
    # usf4i72p - From pretrain fault on 2 std=1 mean1 1 1 
    # 3rhaayes - From pretrain fault on 2 std=1 mean1 1 2
    # l93ss9p4 - From pretrain fault on 2 std=1 mean1 1 3

    # 70d8l0br - from nothing with limit on joint 2 from -pi/4 to pi/4 1
    # x7ruwrcj - from nothing with limit on joint 2 from -pi/4 to pi/4 2:
    #    Walked on 2000
    # 125mit4e - from nothing with limit on joint 2 from -pi/4 to pi/4 3
    # qdlgafl1 - from nothing with limit on joint 2 from -pi/4 to pi/4 4
    # j2lj4qbg - from pretrain with limit on joint 2 from -pi/4 to pi/4 1
    # z5de61zd - from pretrain with limit on joint 2 from -pi/4 to pi/4 2
    # oeii2foh - from pretrain with limit on joint 2 from -pi/4 to pi/4 3
    # zu85qxrw - from pretrain with limit on joint 2 from -pi/4 to pi/4 4

    # nnc1rfh5 - From nothing joint 1,4,7,10 restricted between -pi/2 and pi/2 1 
    # nicros41 - From nothing joint 1,4,7,10 restricted between -pi/2 and pi/2 2
    # crsj54o6 - From nothing joint 1,4,7,10 restricted between -pi/2 and pi/2 3
    # 5peeme6x - From pretrain joint 1,4,7,10 restricted between -pi/2 and pi/2 1
    # jurylyvy - From pretrain joint 1,4,7,10 restricted between -pi/2 and pi/2 2
    # 9urkk5sp - From pretrain joint 1,4,7,10 restricted between -pi/2 and pi/2 3
    # 6hdfdv97 - From pretrain joint 1,4,7,10 restricted between -pi/2 and pi/2 4

    # 6zbe64lw - from pretrain with limit on joint 1, 4 from -pi/4 to pi/4 1
    # o7ck8njf - from pretrain with limit on joint 1, 4 from -pi/4 to pi/4 2
    # ajqsarup - from pretrain with limit on joint 1, 4 from -pi/4 to pi/4 3
    # pvcwfwxz - from pretrain with limit on joint 1, 4 from -pi/4 to pi/4 4

    # 7eyo5sv7 - from pretrain with limit on joint 1, 4 from 0 to pi/4 1
    # 617hqdc3 - from pretrain with limit on joint 1, 4 from 0 to pi/4 2
    # 2v42qgh8 - from pretrain with limit on joint 1, 4 from 0 to pi/4 3
    # zxw8x7km - from pretrain with limit on joint 1, 4 from 0 to pi/4 4

    max_step = 50000
    save_interval = 1000
    steps = [f"{i:06}" for i in range(0, max_step + 1, save_interval)] + ['latest', 'best']
    print(steps)
    
    run_id = "vye0j20m"

    play_go1_from_files(run_id=run_id, steps=steps, output_video_name='play_video_combined_diff_commands_2.mp4', headless=False)


    