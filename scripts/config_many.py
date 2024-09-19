
# # -- The original list of variations --
# cfg_list =  [ {"joints": j, "value": val, "is_torque": trq} 
#                 for j in [[2,], [0,1,2]]
#                 for val in [0, -100, 100]
#                 for trq in [True, False]
#             ]
# for complex_value in [[0, 1, -2], [0, -1, 2], [0, 100, -100], [0, -100, 100]]:
#     cfg_list.append({"joints":[0,1,2], "value": complex_value, "is_torque": False})

# -- minimal list
#cfg_list = [{"joints": j, "value": 0, "is_torque": False} for j in [[0,1,2], [2,], [1,], [0,]] ] # [] 
cfg_list = [{"joints": [0,1,2], "value": val, "is_torque": False} for val in [-100, 100]]
cfg_list.append({"joints": [], "value": 0, "is_torque": False})
cfg_num = len(cfg_list)


def config_env(Cfg, cfg_idx=0):
    """ Test our new parameters """
    # 0. common
    import numpy as np
    spec = cfg_list[cfg_idx]
    relevant_joints = spec["joints"]
    positional_target = spec["value"]

    # 1. positional action override - do not really set the position, uses the original controller to decide the torque
    Cfg.control.override_joint_action = not spec["is_torque"]
    # # 2. torque action override
    Cfg.control.override_torque = spec["is_torque"]
    Cfg.control.override_torque_value = spec["value"]  # -100  # zero to neutral, +-const to force extreme position?

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


def config_log(logger, Cfg):
    """ write summary """
    file = "enforce_summary.txt"
    if Cfg.control.override_torque:
        logger.log_text(
            f"Torque Enforcing. \n" +
            f"Of indices: {Cfg.control.override_torque_index} \n" +
            f"To: {Cfg.control.override_torque_value} \n\n",
            filename=file)
    if Cfg.control.override_joint_action:
        logger.log_text(
            f"Action Enforcing. \n" +
            f"Of indices: {Cfg.control.override_action_index} \n" +
            f"To: {Cfg.control.override_joint_value} \n\n",
            filename=file)
    if not Cfg.control.override_torque and not Cfg.control.override_joint_action:
        logger.log_text(
            f"Original. Not Enforcing. \n\n",
            filename=file)
    #  add in future if relevant: if Cfg.reward_scales.fixed_leg > 0.0:

