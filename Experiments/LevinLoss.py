import math
import numpy as np

def levin_loss_continuous_with_maxlen(
    trajectory,
    options,
    base_policy=None,
    tol=1e-3
):
    """
    Levin loss for continuous‐action setting, where each option `opt`
    has its own `opt.max_len` and provides action via `opt.act(obs)`.
    """
    T = len(trajectory)
    num_options = len(options)
    M = [float('inf')] * (T + 1)
    M[0] = 0

    for j in range(T):
        obs_j, true_action_j = trajectory[j]

        # primitive (one‐step) coverage
        if base_policy is not None:
            prim_act = base_policy.act(obs_j)  # assume returns NumPy array
            if np.linalg.norm(prim_act - true_action_j) <= tol:
                M[j+1] = min(M[j+1], M[j] + 1)

        # each option covers up to opt.max_len steps
        for opt in options:
            seg_len = 0
            while (
                seg_len < opt.max_len and  # don’t exceed this option’s budget
                j + seg_len < T
            ):
                obs_k, true_act_k = trajectory[j + seg_len]

                # call your option’s act → torch Tensor → NumPy
                pred_act_k = opt.act(obs_k).detach().cpu().numpy()            

                # stop if mis‐match
                if np.linalg.norm(pred_act_k - true_act_k) > tol:
                    break

                seg_len += 1

            if seg_len > 0:
                M[j + seg_len] = min(M[j + seg_len], M[j] + 1)

    num_decisions = M[T]
    depth = T + 1
    choice_count = num_options + (1 if base_policy is not None else 0)
    uniform_p = 1.0 / choice_count
    return math.log(depth) - num_decisions * math.log(uniform_p)




def levin_loss_discrete(trajectory, options, num_actions):
    """
    Calculates the Levin Loss for a given trajectory and a list of agents.
    
    Parameters:
        trajectory (List[Tuple[Any, Any]]): A list of (observation, action) pairs.
        options: List of objects with select_action(observation) method for action prediction.
        num_actions (int): The number of available primitive actions.

    Returns:
        float: The computed Levin Loss.
    """
    if options is None:
        num_options = 0
    else:
        num_options = options.n

    T = len(trajectory)

    # M[j] stores the minimum number of decisions required to cover the first j transitions.
    M = [float('inf')] * (T + 1)
    M[0] = 0  # No decisions are needed to cover an empty trajectory.
    
    for j in range(T):
        # Option 1: Use a primitive action to cover one transition.
        if j + 1 <= T:
            M[j+1] = min(M[j+1], M[j] + 1)
        
        # Option 2: For each agent (mask), try to cover as many consecutive transitions as possible.
        for index in range(num_options):
            segment_length = 0
            while j + segment_length < T:
                observation, true_action = trajectory[j + segment_length]
                predicted_action = options.select_action(observation, index)
                if predicted_action != true_action or options.is_terminated(observation):
                    break
                segment_length += 1
            # If the agent can cover at least one transition:
            if segment_length > 0:
                M[j + segment_length] = min(M[j + segment_length], M[j] + 1)
    
    number_decisions = M[T]
    depth = T + 1  # Total number of "positions" is trajectory length plus one.
    
    # Uniform probability over options: the agents plus the primitive action.
    uniform_probability = 1.0 / (num_options + num_actions)
    # Compute the loss in log space.
    loss = math.log(depth) - number_decisions * math.log(uniform_probability)
    return loss