def config_env(Cfg):
    """ Test our new parameters """
    # 0. common
    relevant_joints = range(3)
    positional_target = -0.5

    # 1. positional action override - do not really set the position, uses the original controller to decide the torque
    Cfg.control.override_joint_action = True
    # # 2. torque action override
    # Cfg.control.override_torque = True
    # Cfg.control.override_torque_value = 0  # zero to neutral, +-const to force extreme position?

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