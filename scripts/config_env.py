import functools
import isaacgym
import torch

def config_env(Cfg):
    """ Test our new parameters """
    # 0. common
    import numpy as np
    relevant_joints = [2,] # [0,1,2]
    positional_target = -100 #np.array([0, 1, -2])

    # 1. positional action override - do not really set the position, uses the original controller to decide the torque
    Cfg.control.override_joint_action = False  #  True
    # # 2. torque action override
    Cfg.control.override_torque = True
    Cfg.control.override_torque_value = -100  # zero to neutral, +-const to force extreme position?

    # # rewards (instead of selecting  the action, set a reward). pro: doesnt need to know how much force to apply
    # # 3. positional reward
    # Cfg.reward_scales.fixed_leg = 100.0
    # # 4. torque penalty - addition, if we want to reduce the joints acceleration due to the positional reward
    # Cfg.reward_scales.fixed_leg_torques = 100.0

    # assign common
    Cfg.control.override_action_index = relevant_joints
    Cfg.control.override_torque_index = relevant_joints
    Cfg.rewards.fixed_leg_dof_indices = relevant_joints
    Cfg.control.override_joint_value  = positional_target
    Cfg.rewards.fixed_leg_dof_target  = positional_target

    # Fault related configs 
    Cfg.control.apply_faults = True
    Cfg.control.fault_distribtion_func = functools.partial(torch.normal, mean=0.5, std=1) 
    Cfg.control.fault_min = 0.1 # Minimum engine power to avoid engine shutdown
    Cfg.control.fault_max = 1.4


def config_log(logger, Cfg):
    """ write summary """
    file = ".enfoce_summary.txt"
    if Cfg.control.override_torque:
        logger.log_text(
            f"Torque Enforcing. \n" +
            f"Of indices: {Cfg.control.override_torque_index} \n" +
            f"To: {Cfg.control.override_torque_value} \n\n",
            filename=".enfoce_summary.txt")
    if Cfg.control.override_joint_action:
        logger.log_text(
            f"Action Enforcing. \n" +
            f"Of indices: {Cfg.control.override_action_index} \n" +
            f"To: {Cfg.control.override_joint_value} \n\n",
            filename=".enfoce_summary.txt")
    if not Cfg.control.override_torque and not Cfg.control.override_joint_action:
        logger.log_text(
            f"Original. Not Enforcing. \n\n",
            filename=".enfoce_summary.txt")
    #  add in future if relevant: if Cfg.reward_scales.fixed_leg > 0.0:

