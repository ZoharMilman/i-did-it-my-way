import time
from collections import deque
import copy
import os

import torch
from ml_logger import logger
from params_proto import PrefixProto

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage

import wandb
import numpy as np

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
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 1
    log_freq = 10
    inference_steps = 900 # save half minute video

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True


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

        if RunnerArgs.resume:
            # load pretrained weights from resume_path
            from ml_logger import ML_Logger
            loader = ML_Logger(root="http://escher.csail.mit.edu:8080",
                               prefix=RunnerArgs.resume_path)
            weights = loader.load_torch("checkpoints/ac_weights_last.pt")
            actor_critic.load_state_dict(state_dict=weights)

            if hasattr(self.env, "curricula") and RunnerArgs.resume_curriculum:
                # load curriculum state
                distributions = loader.load_pkl("curriculum/distribution.pkl")
                distribution_last = distributions[-1]["distribution"]
                gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
                for gait_id, gait_name in enumerate(self.env.category_names):
                    self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
                    print(gait_name)

        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0

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

                    if 'eval/episode' in infos:
                        wandb.log(infos['eval/episode'], step=it)

                    if 'curriculum' in infos:
                        cur_reward_sum += rewards
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
                "time_elapsed": time.time() - wandb.run.start_time,
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
                # self.log_video(it)
                self.log_video_wandb(it, fps=30)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs

            # Periodic logging and checkpoint saving
            if it % RunnerArgs.log_freq == 0:
                wandb.log({"timesteps": self.tot_timesteps, "iterations": it}, step=it)

            if it % RunnerArgs.save_interval == 0:
                # Save checkpoints
                ac_weights_checkpoint_path = f"checkpoints/ac_weights_{it:06d}.pt"
                ac_weights_path = f"checkpoints/ac_weights_last.pt"
                if os.path.exists("checkpoints/"):
                    torch.save(self.alg.actor_critic.state_dict(), ac_weights_checkpoint_path)
                    torch.save(self.alg.actor_critic.state_dict(), ac_weights_path)
                else:
                    os.makedirs("checkpoints/")
                    torch.save(self.alg.actor_critic.state_dict(), ac_weights_checkpoint_path)
                    torch.save(self.alg.actor_critic.state_dict(), ac_weights_path)
                    

                # Save other modules as needed
                path = 'checkpoints/'
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
        print("FRAMES LENGTH: ", len(frames))
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")

            # Convert the frames to a video format wandb can handle
            video_path = f"videos/{it:05d}.mp4"
            fps = 1 / self.env.dt
            wandb.log({"train_video": wandb.Video(np.array(frames), fps=fps, format="mp4")}, step=it)

        # Get frames for the evaluation environment if it exists
        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")

                # Convert the frames to a video format wandb can handle
                eval_video_path = f"videos/{it:05d}_eval.mp4"
                wandb.log({"eval_video": wandb.Video(np.array(frames), fps=fps, format="mp4")}, step=it)

    def log_video_wandb(self, it, fps=30):
        # self.alg.actor_critic.adaptation_module
        # self.alg.actor_critic.actor_body
        print("---------------------LOGGING VIDEO-----------------------")
        policy = get_inference_policy(device=self.device)
        video_env = copy.deepcopy(self.env)

        obs = video_env.reset()
        frames = []

        for i in range(inference_steps):
            print(f"STEP {i}")
            with torch.no_grad():
                actions = policy(obs)

            obs, rew, done, info = video_env.step(actions)
            img = video_env.render(mode="rgb_array")
            frames.append(np.array(img))  # Store the frame

        wandb.log({f"train_step={it:05d}_video": wandb.Video(np.array(frames), fps=fps, format="mp4")}, step=it)    
        print("--------------------- DONE LOGGING VIDEO-----------------------")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
