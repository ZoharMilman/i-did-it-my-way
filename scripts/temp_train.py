#!/usr/bin/env python3
from datetime import datetime
from config_env import config_env, config_log
import wandb

#bla bla

def initialize_env_config(Cfg, headless=True):
    print("Importing functions for enviorment initialization...")
    from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    print("Done importing for enviorment initialization")

    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    Cfg.commands.distributional_commands = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.env.priv_observe_motion = False
    Cfg.env.priv_observe_gravity_transformed_motion = False
    Cfg.domain_rand.randomize_friction_indep = False 
    Cfg.domain_rand.randomize_restitution = True
    Cfg.env.priv_observe_restitution = True
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.env.priv_observe_base_mass = False
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    Cfg.domain_rand.randomize_gravity = True
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    Cfg.domain_rand.gravity_impulse_duration = 0.99
    Cfg.env.priv_observe_gravity = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    Cfg.env.priv_observe_com_displacement = False
    Cfg.domain_rand.randomize_ground_friction = True
    Cfg.env.priv_observe_ground_friction = False
    Cfg.env.priv_observe_ground_friction_per_foot = False
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cfg.env.priv_observe_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = True
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cfg.env.priv_observe_motor_offset = False
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.env.priv_observe_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.env.priv_observe_Kd_factor = False
    Cfg.env.priv_observe_body_velocity = False
    Cfg.env.priv_observe_body_height = False
    Cfg.env.priv_observe_desired_contact_states = False
    Cfg.env.priv_observe_contact_forces = False
    Cfg.env.priv_observe_foot_displacement = False
    Cfg.env.priv_observe_gravity_transformed_foot_displacement = False

    Cfg.env.num_privileged_obs = 2
    Cfg.env.num_observation_history = 30
    Cfg.reward_scales.feet_contact_forces = 0.0

    Cfg.domain_rand.rand_interval_s = 4
    Cfg.commands.num_commands = 15
    Cfg.env.observe_two_prev_actions = True
    Cfg.env.observe_yaw = False
    Cfg.env.num_observations = 70
    Cfg.env.num_scalar_observations = 70
    Cfg.env.observe_gait_commands = True
    Cfg.env.observe_timing_parameter = False
    Cfg.env.observe_clock_inputs = True

    Cfg.domain_rand.tile_height_range = [-0.0, 0.0]
    Cfg.domain_rand.tile_height_curriculum = False
    Cfg.domain_rand.tile_height_update_interval = 1000000
    Cfg.domain_rand.tile_height_curriculum_step = 0.01
    Cfg.terrain.border_size = 0.0
    Cfg.terrain.mesh_type = "trimesh"
    Cfg.terrain.num_cols = 30
    Cfg.terrain.num_rows = 30
    Cfg.terrain.terrain_width = 5.0
    Cfg.terrain.terrain_length = 5.0
    Cfg.terrain.x_init_range = 0.2
    Cfg.terrain.y_init_range = 0.2
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 4
    Cfg.terrain.horizontal_scale = 0.10
    Cfg.rewards.use_terminal_foot_height = False
    Cfg.rewards.use_terminal_body_height = True
    Cfg.rewards.terminal_body_height = 0.05
    Cfg.rewards.use_terminal_roll_pitch = True
    Cfg.rewards.terminal_body_ori = 1.6

    Cfg.commands.resampling_time = 10

    Cfg.reward_scales.feet_slip = -0.04
    Cfg.reward_scales.action_smoothness_1 = -0.1
    Cfg.reward_scales.action_smoothness_2 = -0.1
    Cfg.reward_scales.dof_vel = -1e-4
    Cfg.reward_scales.dof_pos = -0.0
    Cfg.reward_scales.jump = 10.0
    Cfg.reward_scales.base_height = 0.0
    Cfg.rewards.base_height_target = 0.30
    Cfg.reward_scales.estimation_bonus = 0.0
    Cfg.reward_scales.raibert_heuristic = -10.0
    Cfg.reward_scales.feet_impact_vel = -0.0
    Cfg.reward_scales.feet_clearance = -0.0
    Cfg.reward_scales.feet_clearance_cmd = -0.0
    Cfg.reward_scales.feet_clearance_cmd_linear = -30.0
    Cfg.reward_scales.orientation = -5.0 # edited
    Cfg.reward_scales.orientation_control = -50.0 # edited
    Cfg.reward_scales.tracking_stance_width = -0.0
    Cfg.reward_scales.tracking_stance_length = -0.0
    Cfg.reward_scales.lin_vel_z = -0.02
    Cfg.reward_scales.ang_vel_xy = -0.001
    Cfg.reward_scales.feet_air_time = 0.0
    Cfg.reward_scales.hop_symmetry = 0.0
    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.
    Cfg.rewards.gait_vel_sigma = 10.
    #Cfg.reward_scales.tracking_contacts_shaped_force = 4.0  # NOTE: removed to check if cause the robot to freeze
    #Cfg.reward_scales.tracking_contacts_shaped_vel = 4.0
    Cfg.reward_scales.collision = -5.0

    Cfg.rewards.reward_container_name = "CoRLRewards"
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = True
    Cfg.rewards.sigma_rew_neg = 0.02



    Cfg.commands.lin_vel_x = [-1.0, 1.0]
    Cfg.commands.lin_vel_y = [-0.6, 0.6]
    Cfg.commands.ang_vel_yaw = [-1.0, 1.0]
    Cfg.commands.body_height_cmd = [-0.25, 0.15]
    Cfg.commands.gait_frequency_cmd_range = [2.0, 4.0]
    Cfg.commands.gait_phase_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_offset_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    Cfg.commands.footswing_height_range = [0.03, 0.35]
    Cfg.commands.body_pitch_range = [-0.4, 0.4]
    Cfg.commands.body_roll_range = [-0.0, 0.0]
    Cfg.commands.stance_width_range = [0.10, 0.45]
    Cfg.commands.stance_length_range = [0.35, 0.45]

    Cfg.commands.limit_vel_x = [-5.0, 5.0]
    Cfg.commands.limit_vel_y = [-0.6, 0.6]
    Cfg.commands.limit_vel_yaw = [-5.0, 5.0]
    Cfg.commands.limit_body_height = [-0.25, 0.15]
    Cfg.commands.limit_gait_frequency = [2.0, 4.0]
    Cfg.commands.limit_gait_phase = [0.0, 1.0]
    Cfg.commands.limit_gait_offset = [0.0, 1.0]
    Cfg.commands.limit_gait_bound = [0.0, 1.0]
    Cfg.commands.limit_gait_duration = [0.5, 0.5]
    Cfg.commands.limit_footswing_height = [0.03, 0.35]
    Cfg.commands.limit_body_pitch = [-0.4, 0.4]
    Cfg.commands.limit_body_roll = [-0.0, 0.0]
    Cfg.commands.limit_stance_width = [0.10, 0.45]
    Cfg.commands.limit_stance_length = [0.35, 0.45]

    Cfg.commands.num_bins_vel_x = 21
    Cfg.commands.num_bins_vel_y = 1
    Cfg.commands.num_bins_vel_yaw = 21
    Cfg.commands.num_bins_body_height = 1
    Cfg.commands.num_bins_gait_frequency = 1
    Cfg.commands.num_bins_gait_phase = 1
    Cfg.commands.num_bins_gait_offset = 1
    Cfg.commands.num_bins_gait_bound = 1
    Cfg.commands.num_bins_gait_duration = 1
    Cfg.commands.num_bins_footswing_height = 1
    Cfg.commands.num_bins_body_roll = 1
    Cfg.commands.num_bins_body_pitch = 1
    Cfg.commands.num_bins_stance_width = 1

    Cfg.normalization.friction_range = [0, 1]
    Cfg.normalization.ground_friction_range = [0, 1]
    Cfg.terrain.yaw_init_range = 3.14
    Cfg.normalization.clip_actions = 10.0

    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.binary_phases = True
    Cfg.commands.gaitwise_curricula = True

    config_env(Cfg)
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)
    print("Initialized Enviorment")
    return env

def train_go1(headless=True):
    import isaacgym
    assert isaacgym
    import torch
    import wandb

    from go1_gym.envs.base.legged_robot_config import Cfg
    from go1_gym.envs.go1.go1_config import config_go1
    from go1_gym_learn.ppo_cse import Runner
    from go1_gym_learn.ppo_cse.actor_critic import AC_Args
    from go1_gym_learn.ppo_cse.ppo import PPO_Args
    from go1_gym_learn.ppo_cse import RunnerArgs

    # Initialize configuration and environment
    config_go1(Cfg)
    env = initialize_env_config(Cfg, headless=headless)

    # Initialize wandb
    now = datetime.now()
    
    wandb.init(project="robot-training", config={
        "AC_Args": vars(AC_Args),
        "PPO_Args": vars(PPO_Args),
        "RunnerArgs": vars(RunnerArgs),
        "Cfg": vars(Cfg),
    },
    name=now.strftime("%d_%m_%Y__%H_%M_%S"))

    # Log experiment parameters
    wandb.config.update({
        "AC_Args": vars(AC_Args),
        "PPO_Args": vars(PPO_Args),
        "RunnerArgs": vars(RunnerArgs),
        "Cfg": vars(Cfg),
    })

    gpu_id = 0
    runner = Runner(env, device=f"cuda:{gpu_id}")

    num_of_iterations = 100  # Adjust as needed
    print(f"Running for {num_of_iterations} iterations")

    # Start learning process
    runner.learn(num_learning_iterations=num_of_iterations, init_at_random_ep_len=True, eval_freq=100)

    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    from pathlib import Path

    # Setup wandb project and logging
    wandb.login(key="70236d768d6ec323c1df61af26e16d2a71c0f83f")  
    stem = Path(__file__).stem
    # wandb.init(project="robot-training", name=stem, sync_tensorboard=True)

    # to see the environment rendering, set headless=False
    train_go1(headless=False)


