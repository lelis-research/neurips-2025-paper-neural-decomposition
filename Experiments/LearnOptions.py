import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import numpy as np
import copy

def extract_trajectory(agent, env):
    """
    Generate a single full trajectory by running the agent in the environment until termination.

    Args:
        agent: An object with an `act(observation, greedy=True)` method that returns an action.
        env: An environment following the Gymnasium API, providing `reset()` and `step(action)` methods.

    Returns:
        List of [observation, action] pairs collected along the trajectory.
    """
    full_trajectory = []
    observation, info = env.reset(seed=0)

    while True:
        action = agent.act(observation, greedy=True)
        full_trajectory.append([observation, action])

        next_observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break        
        observation = next_observation
    return full_trajectory

def generate_subtrajectories(traj, min_len, max_len):
    """
    Generate all contiguous sub-trajectories of `traj` whose lengths are 
    between min_len and max_len inclusive.
    
    Args:
        traj (list): List of (state, action) pairs.
        min_len (int): Minimum length of sub-trajectory.
        max_len (int): Maximum length of sub-trajectory.
        
    Returns:
        List of sub-trajectories (each a list of (state, action) pairs).
    """
    subs = []
    t = len(traj)
    for length in range(min_len, max_len + 1):
        for start in range(t - length + 1):
            subs.append(traj[start : start + length])
    return subs

def learn_mask(agent, sub_traj, num_epochs=100, lr=1e-2, pbar=None, tol=1e-3):
    """
    Learn a mask tensor that selectively overrides state features so that
    agent.actor_critic.get_action(masked_state) matches the actions in sub_traj.

    Args:
        agent:     A PPO agent instance with an attribute `actor_critic` whose
                   `get_action(state: Tensor)` returns at least the action.
        sub_traj:  List of (state, action) pairs, where each is a NumPy array.
        num_epochs (int): Number of gradient‐descent iterations to train the mask.
        lr (float):       Learning rate for the mask optimizer.

    Returns:
        torch.Tensor: Learned mask of the same shape as each state (on CPU).
                       Mask entries ≈0 leave that state dimension unchanged;
                       nonzero entries override that dimension.
    """

    example_state_np, _ = sub_traj[0]
    example_state = torch.from_numpy(example_state_np).float()
    mask = nn.Parameter(torch.zeros_like(example_state))
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam([mask], lr=lr)
    
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for state_np, action_np in sub_traj:
            state = torch.from_numpy(state_np).float()
            target_action = torch.from_numpy(action_np).float()

            masked_state = mask + state #torch.where(mask == 0, state, mask)
            pred_action = agent.actor_critic.actor_mean(masked_state)

            total_loss += loss_fn(pred_action, target_action)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if pbar is not None:
            pbar.set_postfix(loss=total_loss.item())
            pbar.update(1)
    
    #test the mask for the sub_traj
    failed = False
    for state_np, action_np in sub_traj:
        state = torch.from_numpy(state_np).float()
        target_action = torch.from_numpy(action_np).float()

        masked_state = mask + state #torch.where(mask == 0, state, mask)
        pred_action = agent.actor_critic.actor_mean(masked_state)
        if torch.norm(pred_action - target_action) > tol:
            failed = True
            break
    
    if failed:
        return None
    return mask.detach().cpu()


def fine_tune_policy(agent, sub_traj, num_epochs=100, lr=1e-2, pbar=None, tol=1e-3):
    """
    Fine-tune a copy of agent.actor_critic so that actor_mean(state) 
    matches the recorded actions in sub_traj.

    Args:
        agent:      PPO agent with .actor_critic, which has methods
                    .actor_mean(state) and attributes .actor, .critic.
        sub_traj:   List of (state_np, action_np) pairs (NumPy arrays).
        num_epochs: How many passes through sub_traj to make.
        lr:         Learning rate for the actor optimizer.
        pbar:       Optional tqdm progress bar (gets updated by batch-size).
        tol:        Tolerance for final per-step error to accept fit.

    Returns:
        actor_critic_copy (nn.Module): The fine-tuned copy, or
        None if it fails to meet tol on the training set.
    """
    # 1) Clone the policy (actor_critic) so original remains untouched
    actor_critic_copy = copy.deepcopy(agent.actor_critic)
    actor_critic_copy.train()

    # 2) Freeze critic (we only want to update the actor)
    for p in actor_critic_copy.critic.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(actor_critic_copy.actor_mean.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 3) Gradient-descent loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        for state_np, action_np in sub_traj:
            state  = torch.from_numpy(state_np).float()
            target = torch.from_numpy(action_np).float()

            pred = actor_critic_copy.actor_mean(state)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if pbar is not None:
            pbar.set_postfix(loss=total_loss)
            pbar.update(1)

    # 4) Quick validation: ensure learned policy reproduces sub_traj within tol
    for state_np, action_np in sub_traj:
        s = torch.from_numpy(state_np).float()
        t = torch.from_numpy(action_np).float()
        if torch.norm(actor_critic_copy.actor_mean(s) - t) > tol:
            return None

    return actor_critic_copy



def find_best_subset(options, loss_fn, max_iters=100, restarts=10):
    """
    Find a subset of `options` that (approximately) minimizes `loss_fn` using
    hill-climbing with random restarts.

    Args:
        options (List[Any]): The full list of option objects.
        loss_fn (Callable[[List[Any]], float]): A function that takes a subset
            of `options` and returns its loss (to be minimized).
        max_iters (int): Maximum number of hill-climbing iterations per restart.
        restarts (int): Number of random restarts.

    Returns:
        Tuple[List[Any], float]: The best subset found and its loss.
    """
    def hill_climb(initial_subset):
        subset = initial_subset[:]
        current_loss = loss_fn(subset)
        for _ in range(max_iters):
            improved = False
            # try flipping each option in or out
            for opt in options:
                if opt in subset:
                    candidate = [o for o in subset if o is not opt]
                else:
                    candidate = subset + [opt]

                candidate_loss = loss_fn(candidate)
                if candidate_loss < current_loss:
                    subset = candidate
                    current_loss = candidate_loss
                    improved = True
                    break  # restart scanning options
            if not improved:
                break
        return subset, current_loss

    best_subset, best_loss = [], float('inf')
    for _ in tqdm(range(restarts)):
        # initialize with a random subset (you can also choose empty or half-size)
        init_size = random.randint(0, len(options))
        initial = random.sample(options, init_size)
        subset, loss = hill_climb(initial)
        if loss < best_loss:
            best_subset, best_loss = subset, loss

    return best_subset, best_loss


def find_best_subset_stochastic(
    options,
    loss_fn,
    max_iters=200,
    restarts=5,
    neighbor_samples=20,
    max_size=None,
):
    """
    Stochastic hill‐climbing: at each step we only try a random
    subset of size `neighbor_samples` of the N possible flips.

    Args:
        options (List): full list of option objects
        loss_fn (callable): takes List[options] -> float loss
        max_iters (int): max flips to try in one climb
        restarts (int): how many random restarts
        neighbor_samples (int): how many random flips to evaluate per iteration

    Returns:
        best_subset, best_loss
    """
    pbar = tqdm(desc="Search", unit="samples")
    best_subset, best_loss = [], float('inf')
    no_option_loss = loss_fn([])
    
    for r in range(restarts):
        # random start subset
        start_size = 0 #if max_size is None else max_size
        subset = set(random.sample(options, start_size)) #initialize with random subset
        curr_loss = loss_fn(list(subset))
        # one climb run
        for iter in range(max_iters):
            # sample some flips
            candidates = random.sample(options, 
                                       k=min(neighbor_samples, len(options)))
            improved = False
            for i, opt in enumerate(candidates):
                if opt in subset:
                    cand = list(subset - {opt})
                else:
                    if max_size is not None and len(subset) >= max_size:
                        continue
                    cand = list(subset | {opt})
                cand_loss = loss_fn(cand)
                
                pbar.set_postfix(
                    cand=f"{i}/{len(candidates)}",
                    cand_loss=f"{cand_loss:.4f}",
                    best_loss=f"{best_loss:.4f}",
                    no_option_loss=f"{no_option_loss:.4f}",
                    num_selected_options=len(best_subset),
                    iterations=f"{iter+1}/{max_iters}",
                    restarts=f"{r+1}/{restarts}",
                )
                pbar.update(1)

                if cand_loss < curr_loss:
                    subset, curr_loss = set(cand), cand_loss
                    improved = True
                    break

            # global best check
            if curr_loss < best_loss:
                best_loss = curr_loss
                best_subset = list(subset)

            if not improved:
                # no local improvement → stop this climb early
                # but we still count this as one of the max_iters
                # and continue hopping back to the next restart
                break
    pbar.close()
    return best_subset, best_loss