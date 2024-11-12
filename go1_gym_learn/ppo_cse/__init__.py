import time
from collections import deque
import copy
import os

import torch
# from ml_logger import logger
from params_proto import PrefixProto

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage

import wandb
import numpy as np

import imageio

from scripts import play

from datetime import datetime

import multiprocessing

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from go1_gym_learn.ppo.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 500  # check for potential saves every this many iterations
    save_video_interval = 200
    log_freq = 10
    # inference_steps = 900 # save half minute video

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True
    run_id = None # For wandb style logging 
    api_run_path = None # Also for wandb


class Runner:

    def __init__(self, env, device='cpu'):
        from .ppo import PPO

        print("ppo_cse: Rennner.init start")
        self.device = device
        self.env = env
        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      ).to(self.device)

        
        
        self.tot_time = 0
        self.current_learning_iteration = 0

        if RunnerArgs.resume:
            # load pretrained weights from resume_path
            # from ml_logger import ML_Logger
            # loader = ML_Logger(root="http://escher.csail.mit.edu:8080",
            #                    prefix=RunnerArgs.resume_path)
            # weights = loader.load_torch("checkpoints/ac_weights_last.pt")
            # actor_critic.load_state_dict(state_dict=weights)

            print(f'--------------------RESUMING RUN: {RunnerArgs.run_id}--------------------------------')
            actor_critic.load_state_dict(torch.load(RunnerArgs.resume_path + f'/checkpoints/{RunnerArgs.run_id}/ac_weights_last.pt'))

            if 'pretrain' not in RunnerArgs.resume_path:
                api = wandb.Api()
                run = api.run(RunnerArgs.api_run_path)
                history = run.history()
                # Check the last entry in the history to find the last logged iteration
                if not history.empty:
                    self.current_learning_iteration = int(history['iteration'].iloc[-1])  
                    print(f'Last logged iteration: {self.current_learning_iteration}')
                else:
                    print("No history found for this run.")
            

            # if hasattr(self.env, "curricula") and RunnerArgs.resume_curriculum:
            #     distribution_last = history["distribution"].iloc[-1] 
            #     gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
            #     for gait_id, gait_name in enumerate(self.env.category_names):
            #         self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
            #         print(gait_name)

            # if hasattr(self.env, "curricula") and RunnerArgs.resume_curriculum:
            #     # load curriculum state
            #     distributions = loader.load_pkl("curriculum/distribution.pkl")
            #     distribution_last = distributions[-1]["distribution"]
            #     gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
            #     for gait_id, gait_name in enumerate(self.env.category_names):
            #         self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
            #         print(gait_name)


        self.tot_timesteps = 0
        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        # self.tot_timesteps = 0
        # self.tot_time = 0
        # self.current_learning_iteration = 0
        # self.last_recording_it = 0

        self.env.reset()


    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):
        import wandb
        
        print('---------------------------USING WANDB--------------------------')

        # Initialize wandb
        # wandb.init(project="gait-conditioned-agility", config=self.ppo_args)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                            high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout, etc.)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        rew_total_sum = 0

        print(__name__ + ": start iterate")
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            print("ITERATION: ", it)
            

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs], obs_history[:num_train_envs])

                    if eval_expert:
                        actions_eval = self.alg.actor_critic.act_teacher(obs_history[num_train_envs:], privileged_obs[num_train_envs:])
                    else:
                        actions_eval = self.alg.actor_critic.act_student(obs_history[num_train_envs:])

                    ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)

                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    if 'train/episode' in infos:
                        wandb.log(infos['train/episode'], step=it)
                        rew_total_sum += infos['train/episode']['rew_total']
                        rew_total_mean = rew_total_sum/it

                    if 'eval/episode' in infos:
                        wandb.log(infos['eval/episode'], step=it)

                    if 'curriculum' in infos:
                        cur_reward_sum += rewards
                        print("Reward Sum: ", cur_reward_sum)
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                        new_ids_eval = new_ids[new_ids >= num_train_envs]
                        rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                        lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_eval] = 0
                        cur_episode_length[new_ids_eval] = 0

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

                if it % curriculum_dump_freq == 0:
                    wandb.log({"curriculum_info_iteration": it,
                            **caches.slot_cache.get_summary(),
                            **caches.dist_cache.get_summary()})

                    if 'curriculum/distribution' in infos:
                        wandb.log({"distribution": distribution}, step=it)

            # Update the model and record losses
            (mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, 
            mean_decoder_loss_student, mean_adaptation_module_test_loss, 
            mean_decoder_test_loss, mean_decoder_test_loss_student) = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            wandb.log({
                "iteration": it,   
                "time_elapsed": time.time() - wandb.run.start_time,
                "rew_total_sum": rew_total_sum,
                "rew_total_mean": rew_total_mean,
                "time_iter": learn_time,
                "adaptation_loss": mean_adaptation_module_loss,
                "mean_value_loss": mean_value_loss,
                "mean_surrogate_loss": mean_surrogate_loss,
                "mean_decoder_loss": mean_decoder_loss,
                "mean_decoder_loss_student": mean_decoder_loss_student,
                "mean_decoder_test_loss": mean_decoder_test_loss,
                "mean_decoder_test_loss_student": mean_decoder_test_loss_student,
                "mean_adaptation_module_test_loss": mean_adaptation_module_test_loss
            }, step=it)

            if it % RunnerArgs.save_video_interval == 0:
                print("should save eval vid here")
                # self.log_video(it)
                # self.log_video_wandb(it, fps=30)
                # video_process = multiprocessing.Process(target=self.save_video_process, args=(it, ))
                # video_process.start()

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs

            # Periodic logging and checkpoint saving
            if it % RunnerArgs.log_freq == 0:
                wandb.log({"timesteps": self.tot_timesteps, "iterations": it}, step=it)

            if (it) % RunnerArgs.save_interval == 0:
                # Save checkpoints
                now = datetime.now()
                current_time = now.strftime("%d_%m_%Y__%H_%M_%S")
                path = 'checkpoints/' + wandb.run.id + '/'
                ac_weights_checkpoint_path = f"{path}/ac_weights_{it:06d}.pt"
                ac_weights_path = f"{path}/ac_weights_last.pt"
                if os.path.exists(f"{path}"):
                    torch.save(self.alg.actor_critic.state_dict(), ac_weights_checkpoint_path)
                    torch.save(self.alg.actor_critic.state_dict(), ac_weights_path)
                else:
                    os.makedirs(f"{path}")
                    torch.save(self.alg.actor_critic.state_dict(), ac_weights_checkpoint_path)
                    torch.save(self.alg.actor_critic.state_dict(), ac_weights_path)
                    

                # Save other modules as needed
                
                os.makedirs(path, exist_ok=True)

                adaptation_module_checkpoint_path = f'{path}/adaptation_module_{it:06d}.jit'
                adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                traced_script_adaptation_module = torch.jit.script(adaptation_module)
                traced_script_adaptation_module.save(adaptation_module_checkpoint_path)
                traced_script_adaptation_module.save(adaptation_module_path)

                body_checkpoint_path = f'{path}/body_{it:06d}.jit'
                body_path = f'{path}/body_latest.jit'
                body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
                traced_script_body_module = torch.jit.script(body_model)
                traced_script_body_module.save(body_path)
                traced_script_body_module.save(body_checkpoint_path)

                # Use wandb for checkpoint tracking
                wandb.save(ac_weights_path)
                wandb.save(ac_weights_checkpoint_path)
                wandb.save(adaptation_module_path)
                wandb.save(adaptation_module_checkpoint_path)
                wandb.save(body_path)
                wandb.save(body_checkpoint_path)

                # # Delete local shenanigans 
                # os.rmdir(f'{path}')

        self.current_learning_iteration += num_learning_iterations
        print("Learning complete")
        # wandb.finish()


    def log_video(self, it):
        # return  # ignore video
        print("save vid")
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        # Get frames for the training environment
        frames = self.env.get_complete_frames()
        # frames = self.env.get_video_frames()
        print("FRAMES LENGTH: ", len(frames))
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")

            # Convert the frames to a video format wandb can handle
            video_path = f"videos/{it:05d}.mp4"
            fps = 1 / self.env.dt

            output_filename = f"train_step={it:05d}_video.mp4"
            imageio.mimsave(output_filename, frames, fps=fps)
            wandb.log({f"train_step={it:05d}_video": wandb.Video(output_filename, fps=fps, format="mp4")}, step=it)
            # Delete the local video file after logging
            if os.path.exists(output_filename):
                os.remove(output_filename)  # Remove the file    
            # wandb.log({"train_video": wandb.Video(np.array(frames), fps=fps, format="mp4")}, step=it)

        # Get frames for the evaluation environment if it exists
        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")

                # Convert the frames to a video format wandb can handle
                eval_video_path = f"videos/{it:05d}_eval.mp4"
                wandb.log({"eval_video": wandb.Video(np.array(frames), fps=fps, format="mp4")}, step=it)

    
    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference


    def log_video_wandb(self, it, fps=30):
        from scripts import play

        with torch.inference_mode():
            print("---------------------LOGGING VIDEO-----------------------")
            policy = self.get_inference_policy(device=self.device)
            frames = play.play_go1(self.env, policy)

            output_filename = f"train_step={it:05d}_video.mp4"
            imageio.mimsave(output_filename, frames, fps=fps)
            wandb.log({f"train_step={it:05d}_video": wandb.Video(output_filename, fps=fps, format="mp4")}, step=it)

        # with torch.inference_mode():
        #     print("---------------------LOGGING VIDEO-----------------------")
        #     policy = self.get_inference_policy(device=self.device)
        #     print(policy)

        #     # video_env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=True, cfg=self.env.cfg)
        #     # video_env = copy.deepcopy(self.env)
        #     # import pickle
        #     # video_env = pickle.loads(pickle.dumps(self.env))
        #     # video_env = self.env

        #     obs = video_env.reset()
        #     # obs = video_env.get_observations()
                
        #     frames = []
        #     for i in range(fps * video_len):
        #         print(f"STEP {i}")
        #         with torch.no_grad():
        #             actions = policy(obs)

        #         obs, rew, done, info = video_env.step(actions)
        #         img = video_env.render(mode="rgb_array")
        #         frames.append(np.array(img))  # Store the frame

        #     output_filename = f"train_step={it:05d}_video.mp4"
        #     imageio.mimsave(output_filename, frames, fps=30)
        #     wandb.log({f"train_step={it:05d}_video": wandb.Video(output_filename, fps=fps, format="mp4")}, step=it)
        #     # Delete the local video file after logging
        #     if os.path.exists(output_filename):
        #         os.remove(output_filename)  # Remove the file

        #     print("--------------------- DONE LOGGING VIDEO-----------------------")

    def save_video_process(self, it):
        self.log_video_wandb(it, fps=30)

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
