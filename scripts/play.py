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
    # body = torch.jit.load(logdir + '/checkpoints/body_000800.jit')
    adaptation_module = torch.jit.load(logdir + f'/checkpoints/{run_id}/adaptation_module_{step}.jit')
    # adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_000800.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


# def load_env(logdir, headless=False):
#     # logdir = get_logdir()
#     print("----------------LOADING ENV FROM parameters.pkl---------------------")
#     with open(logdir + "/parameters.pkl", 'rb') as file:
#         pkl_cfg = pkl.load(file)
#         print(pkl_cfg.keys())
#         cfg = pkl_cfg["Cfg"]
#         print(cfg.keys())

#         for key, value in cfg.items():
#             if hasattr(Cfg, key):
#                 for key2, value2 in cfg[key].items():
#                     setattr(getattr(Cfg, key), key2, value2)

#     # turn off DR for evaluation script
#     Cfg.domain_rand.push_robots = False
#     Cfg.domain_rand.randomize_friction = False
#     Cfg.domain_rand.randomize_gravity = False
#     Cfg.domain_rand.randomize_restitution = False
#     Cfg.domain_rand.randomize_motor_offset = False
#     Cfg.domain_rand.randomize_motor_strength = False
#     Cfg.domain_rand.randomize_friction_indep = False
#     Cfg.domain_rand.randomize_ground_friction = False
#     Cfg.domain_rand.randomize_base_mass = False
#     Cfg.domain_rand.randomize_Kd_factor = False
#     Cfg.domain_rand.randomize_Kp_factor = False
#     Cfg.domain_rand.randomize_joint_friction = False
#     Cfg.domain_rand.randomize_com_displacement = False

#     Cfg.env.num_recording_envs = 1
#     Cfg.env.num_envs = 1
#     Cfg.terrain.num_rows = 5
#     Cfg.terrain.num_cols = 5
#     Cfg.terrain.border_size = 0
#     Cfg.terrain.center_robots = True
#     Cfg.terrain.center_span = 1
#     Cfg.terrain.teleport_robots = True

#     Cfg.domain_rand.lag_timesteps = 6
#     Cfg.domain_rand.randomize_lag_timesteps = True
#     Cfg.control.control_type = "actuator_net"

#     # our part
#     # config_env(Cfg) # THIS IS WHERE YOU NEED TO ADD MALFUNCTIONS

#     from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

#     env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg) 
#     env = HistoryWrapper(env)

#     # load policy
#     # from ml_logger import logger
#     # from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

#     policy = load_policy(logdir)

#     return env, policy

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

def load_env_from_yaml(logdir, step, headless=False):
    print("----------------LOADING ENV FROM YAML---------------------")
    with open(logdir + "/config.yaml", 'r') as file:
        yaml_cfg = yaml.safe_load(file)
        print(yaml_cfg.keys())
        cfg = yaml_cfg["Cfg"]['value']
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

    policy = load_policy(logdir, step)
    # policy = actor_critic.act_inference

    return env, policy


def get_play_frames(env, policy, num_eval_steps=450):
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
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



def play_go1_from_files(logdir, step='latest', headless=True):
    # from ml_logger import logger

    from go1_gym import MINI_GYM_ROOT_DIR
    import imageio

    # Create a media directory if it doesn't exist
    media_dir = os.path.join(logdir, 'media')
    os.makedirs(media_dir, exist_ok=True)

    # Load environment and policy
    env, policy = load_env_from_yaml(logdir, step, headless=headless)

    # Generate frames
    frames = get_play_frames(env, policy)

    # Save video in the media directory
    output_filename = os.path.join(media_dir, f'play_video_{step}.mp4')
    imageio.mimsave(output_filename, frames, fps=30)
    print("Saved video to: " + output_filename)

    # # plot target and measured forward velocity
    # from matplotlib import pyplot as plt
    # import matplotlib
    # matplotlib.use('TKAgg')
    # fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    # axs[0].legend()
    # axs[0].set_title("Forward Linear Velocity")
    # axs[0].set_xlabel("Time (s)")
    # axs[0].set_ylabel("Velocity (m/s)")

    # axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions[:,:6], linestyle="-", label=[str(i) for i in range(6)])
    # axs[1].set_title("Joint Positions")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Joint Position (rad)")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()



def play_go1(env, policy, num_eval_steps=900, headless=False):
    # Get a video enviorment with DR turned off
    video_env = get_video_env(env)
    frames = get_play_frames(video_env, policy, num_eval_steps=num_eval_steps)

    return frames

# if __name__ == '__main__':
#     # to see the environment rendering, set headless=False
#     play_go1_from_files(logdir="wandb/run-20241103_191407-fn5iwsaf/files", step='000999', headless=False)



if __name__ == '__main__':
    logdir = "wandb/latest-run/files"
    # run_id = os.listdir(logdir + '/checkpoints')[0]

    # # Find all steps in the checkpoints directory
    # checkpoint_dir = os.path.join(os.path.join(logdir, "checkpoints"), run_id)
    # print(checkpoint_dir)
    # checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "adaptation_module_*.jit"))
    
    # # Extract steps from filenames
    # steps = []
    # for file_path in checkpoint_files:
    #     filename = os.path.basename(file_path)
    #     step = filename.split('_')[-1].split('.')[0]
    #     steps.append(step)
    
    # print(f"Found steps: {steps}")

    # # Run play_go1_from_files for each step
    # for step in steps:
    #     print('Playing step: ' + step)
    play_go1_from_files(logdir=logdir, step='latest', headless=False)