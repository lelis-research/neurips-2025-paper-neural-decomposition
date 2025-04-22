import os
import numpy as np
import imageio
import torch

def agent_environment_step_loop(env, agent, total_steps, training=True, writer=None, save_frame_freq=1):
    """
    Runs the training loop for the agent in the given environment over a specified number of timesteps.
    Returns a list of dictionaries containing the episodic return and episode length.
    """
    greedy = False if training else True
    timestep = 0
    episode_counter = 0
    results = []
    while timestep < total_steps:
        observation, info = env.reset()
        episode_return = 0.0
        episode_length = 0

        while True:
            action = agent.act(observation, greedy=greedy)
            next_observation, reward, terminated, truncated, info = env.step(action)
            if training:
                agent.update(next_observation, reward, terminated, truncated)

            episode_return += info["actual_reward"] if "actual_reward" in info else reward
            episode_length += 1
            timestep += 1

            if timestep >= total_steps:
                truncated = True

            if terminated or truncated:
                break

            observation = next_observation

        episode_counter += 1
        # log to TensorBoard
        if writer is not None:
            writer.add_scalar(f"Training/EpisodeReturn",
                              episode_return,
                              global_step=timestep)
            if timestep >= total_steps or episode_counter % save_frame_freq == 0:
                frames = env.render()
                arr = np.stack(frames)                      # (T,H,W,3)
                arr = arr.transpose(0,3,1,2)                 # (T,3,H,W)
                vid = torch.from_numpy(arr).unsqueeze(0)     # (1,T,3,H,W)
                writer.add_video(
                    f"Video/Episode_{episode_counter}", vid, timestep, fps=24
                )   
            
        results.append({
            "episode_return": episode_return,
            "episode_length": episode_length,
        })
        print(f"Episode: {episode_counter} Timestep: {timestep} Return: {episode_return:.2f}")

    return results

def agent_environment_episode_loop(env, agent, total_episodes, training=True, writer=None, save_frame_freq=1):
    """
    Runs the training loop for the agent in the given environment over a specified number of timesteps.
    Returns a list of dictionaries containing the episodic return and episode length.
    """
    greedy = False if training else True
    timestep = 0
    results = []
    for episode_counter in range(1, total_episodes+1):
        observation, info = env.reset()
        episode_return = 0.0
        episode_length = 0

        while True:
            action = agent.act(observation, greedy=greedy)
            next_observation, reward, terminated, truncated, info = env.step(action)
            if training:
                agent.update(next_observation, reward, terminated, truncated)
            episode_return += info["actual_reward"] if "actual_reward" in info else reward
            episode_length += 1
            timestep += 1
            
            if terminated or truncated:
                break

            observation = next_observation

        # log to TensorBoard
        if writer is not None:
            writer.add_scalar(f"Training/EpisodeReturn",
                              episode_return,
                              global_step=timestep)
            
            if episode_counter >= total_episodes or episode_counter % save_frame_freq == 0:
                frames = env.render()
                arr = np.stack(frames)                      # (T,H,W,3)
                arr = arr.transpose(0,3,1,2)                 # (T,3,H,W)
                vid = torch.from_numpy(arr).unsqueeze(0)     # (1,T,3,H,W)
                writer.add_video(
                    f"Video/Episode_{episode_counter}", vid, timestep, fps=24
                )   
            
        results.append({
            "episode_return": episode_return,
            "episode_length": episode_length,
        })
        print(f"Episode: {episode_counter} Timestep: {timestep} Return: {episode_return:.2f}")

    return results