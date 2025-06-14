# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import time
import os
import torch
import wandb
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
from environments.environments_combogrid import PROBLEM_NAMES as COMBO_PROBLEM_NAMES, OPTIMAL_TRAJECTORY_LENGTHS, OPTIMAL_TEST_REWARD
from environments.utils import get_single_environment
from utils import utils
from agents.policy_guided_agent import PPOAgent

def try_agent_deterministicly(agent: PPOAgent, options, args, env_seed):
    if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        problem = None
    elif args.env_id == "ComboGrid":
        problem = COMBO_PROBLEM_NAMES[env_seed]
    else:
        raise NotImplementedError
    env = get_single_environment(args, seed=env_seed, problem=problem, is_test=True, options=options)
    trajectory, infos = agent.run(env, 500, deterministic=True)
    reward = infos["reward"]
    entropy = infos["entropy"]
    print(f"Trajectory length: {trajectory.get_length()}")
    print(infos)
    print(f"Steps: {infos['steps']}, Optimal trajectory length: {OPTIMAL_TRAJECTORY_LENGTHS[env_seed]}")
    return reward == OPTIMAL_TEST_REWARD[env_seed] and entropy < 0.5


def train_ppo(envs: gym.vector.SyncVectorEnv, seed, args, model_file_name, device, logger=None, writer=None,  parameter_sweeps=False, deterministic=False):
    hidden_size = args.hidden_size
    l1_lambda = args.l1_lambda
    if not seed:
        seed = args.env_seed
    
    agent = PPOAgent(envs, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=[seed for _ in range(args.num_envs)])

    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            # global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, entropy, value, _ = agent.get_action_and_value(next_obs, deterministic=deterministic)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "action_size" in infos:
                global_step += infos['action_size'].sum()
                
            bootstrap_next_obs = next_obs.clone()
            if "final_info" in infos:
                global_step += infos['final_info']['action_size'].sum()
                returns = []
                lengths = []
                for i, info in enumerate(infos["final_info"]["_episode"]):
                    if info:
                        returns.append(infos["final_info"]["episode"]["r"][i])  # Collect episodic returns
                        # lengths.append(info["episode"]["l"])  # Collect episodic lengths
                        lengths.append(infos["final_info"]["steps"][i])  # Collect episodic lengths
                        bootstrap_next_obs[i] = torch.Tensor(infos["final_obs"][i]).to(device)


                # Log the average episodic return and length, if any episodes ended
                if returns:
                    avg_return = sum(returns) / len(returns)
                    avg_length = sum(lengths) / len(lengths)
                    if args.track:
                        wandb.log({
                            "Charts/episodic_return_avg": avg_return, 
                            "Charts/episodic_length_avg": avg_length
                        }, step=global_step)
                    if writer:
                        writer.add_scalar("Charts/episodic_return", avg_return, global_step)
                        writer.add_scalar("Charts/episodic_length", avg_length, global_step)
                    logger.info(f"global_step={global_step}, episodic_return={avg_return}, episodic_length={avg_length}, entropy={entropy.mean()}")
                    # if parameter_sweeps and OPTIMAL_TEST_REWARD[seed] - avg_return < 10 and entropy.mean() < 0.15: # FIX: just works for ComboGrid
                    #     logger.info("Trying deterministically ...")
                    #     if try_agent_deterministicly(agent, options, args, seed):
                    #         logger.info(f"Optimal trajectory found on step {global_step}")
                    #         envs.close()
                    #         # writer.close()
                    #         os.makedirs(os.path.dirname(model_file_name), exist_ok=True)
                    #         torch.save(agent.state_dict(), model_file_name) # overrides the file if already exists
                    #         logger.info(f"Saved on {model_file_name}")
                    #         return
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(bootstrap_next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds], deterministic=deterministic)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                l1_reg = torch.tensor(0.).to(device)
                for param in agent.actor.parameters():
                    l1_reg += torch.norm(param, 1)

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + l1_lambda * l1_reg

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if writer:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("Charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/l1_reg", l1_reg.item(), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("Charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.track:
            wandb.log({
                "Charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/l1_reg": l1_reg.item(),
                "losses/explained_variance": explained_var,
                "Charts/SPS": int(global_step / (time.time() - start_time))
            }, step=global_step)
        
        if iteration % 1000 == 0:
            logger.info(f"Global steps: {global_step}")
            logger.info(f"SPS: {int(global_step / (time.time() - start_time))}")
            utils.logger_flush(logger)
        if global_step > args.total_timesteps:
            logger.info(f"Global steps: {global_step}")
            logger.info(f"SPS: {int(global_step / (time.time() - start_time))}")
            logger.info("Training finished.")
            break

    envs.close()
    # writer.close()
    os.makedirs(os.path.dirname(model_file_name), exist_ok=True)
    torch.save(agent.state_dict(), model_file_name) # overrides the file if already exists
    logger.info(f"Saved on {model_file_name}")


def train_ppo_async(envs: gym.vector.AsyncVectorEnv, seed, args, model_file_name, device, options=None, logger=None, writer=None, parameter_sweeps=False, deterministic=False):
    hidden_size = args.hidden_size
    l1_lambda = args.l1_lambda
    if not seed:
        seed = args.env_seed
    
    agent = PPOAgent(envs, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=[seed for _ in range(args.num_envs)])

    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            # global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, entropy, value, _ = agent.get_action_and_value(next_obs, deterministic=deterministic)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if 'action_size' in infos:
                global_step += (infos['action_size']).sum()

            bootstrap_next_obs = next_obs.clone()
            if "final_info" in infos:
                global_step += (infos['final_info']['action_size']).sum()
                returns = []
                lengths = []
                for idx, (r, l) in enumerate(zip(infos["final_info"]['episode']['r'], infos["final_info"]['episode']['l'])):
                    if not infos['_final_info'][idx]:
                        continue
                    returns.append(r)  # Collect episodic returns
                    # lengths.append(info["episode"]["l"])  # Collect episodic lengths
                    lengths.append(l)  # Collect episodic lengths
                    bootstrap_next_obs[idx] = torch.Tensor(infos["final_obs"][idx]).to(device)

                # Log the average episodic return and length, if any episodes ended
                if returns:
                    avg_return = sum(returns) / len(returns)
                    avg_length = sum(lengths) / len(lengths)
                    if args.track:
                        wandb.log({
                            "Charts/episodic_return_avg": avg_return, 
                            "Charts/episodic_length_avg": avg_length
                        }, step=global_step)
                    if writer:
                        writer.add_scalar("Charts/episodic_return", avg_return, global_step)
                        writer.add_scalar("Charts/episodic_length", avg_length, global_step)
                    logger.info(f"global_step={global_step}, episodic_return={avg_return}, episodic_length={avg_length}, entropy={entropy.mean()}")
                    # if parameter_sweeps and OPTIMAL_TEST_REWARD[seed] - avg_return < 10 and entropy.mean() < 0.15: # FIX: just works for ComboGrid
                    #     logger.info("Trying deterministically ...")
                    #     if try_agent_deterministicly(agent, options, args, seed):
                    #         logger.info(f"Optimal trajectory found on step {global_step}")
                    #         envs.close()
                    #         # writer.close()
                    #         os.makedirs(os.path.dirname(model_file_name), exist_ok=True)
                    #         torch.save(agent.state_dict(), model_file_name) # overrides the file if already exists
                    #         logger.info(f"Saved on {model_file_name}")
                    #         return
        
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(bootstrap_next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds], deterministic=deterministic)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                l1_reg = torch.tensor(0.).to(device)
                for param in agent.actor.parameters():
                    l1_reg += torch.norm(param, 1)

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + l1_lambda * l1_reg

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if writer:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("Charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/l1_reg", l1_reg.item(), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("Charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.track:
            wandb.log({
                "Charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/l1_reg": l1_reg.item(),
                "losses/explained_variance": explained_var,
                "Charts/SPS": int(global_step / (time.time() - start_time))
            }, step=global_step)
        
        if iteration % 1000 == 0:
            logger.info(f"Global steps: {global_step}")
            logger.info(f"SPS: {int(global_step / (time.time() - start_time))}")
            utils.logger_flush(logger)
        if global_step > args.total_timesteps:
            logger.info(f"Global steps: {global_step}")
            logger.info(f"SPS: {int(global_step / (time.time() - start_time))}")
            logger.info("Training finished.")
            break

    envs.close()
    # writer.close()
    if not parameter_sweeps:
        os.makedirs(os.path.dirname(model_file_name), exist_ok=True)
        torch.save(agent.state_dict(), model_file_name) # overrides the file if already exists
        logger.info(f"Saved on {model_file_name}")
