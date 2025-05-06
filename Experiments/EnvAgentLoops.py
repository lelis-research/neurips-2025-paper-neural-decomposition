import os
import numpy as np
import imageio
import torch
import copy


def agent_environment_step_loop(env, agent, total_steps, training=True, writer=None, save_frame_freq=1):
    """
    Runs the training loop for the agent in the given environment over a specified number of timesteps.
    Returns a list of dictionaries containing the episodic return and episode length.
    """
    greedy = False if training else True
    timestep = 0
    episode_counter = 1
    results = []
    best_ep, best_agent = -np.inf, None
    while timestep < total_steps:
        observation, info = env.reset()
        episode_return_org = 0.0
        episode_return_wrapped = 0.0
        episode_length = 0
        frames = []
        while True:
            action = agent.act(observation, greedy=greedy)
            next_observation, reward, terminated, truncated, info = env.step(action)
            if training:
                agent.update(next_observation, reward, terminated, truncated)

            episode_return_org += info["actual_reward"] if "actual_reward" in info else reward
            episode_return_wrapped += reward
            episode_length += 1
            timestep += 1

            if timestep >= total_steps:
                truncated = True
 
            if env.render_mode =="rgb_array" and (timestep >= total_steps or episode_counter % save_frame_freq == 0): 
                frames.append(env.render())

            if terminated or truncated:
                break

            observation = next_observation

        
        # log to TensorBoard
        if writer is not None:
            writer.add_scalar(f"Training/EpisodeReturnOrg",
                              episode_return_org,
                              global_step=timestep)
            writer.add_scalar(f"Training/EpisodeReturnWrapped",
                              episode_return_wrapped,
                              global_step=timestep)
            
            if timestep >= total_steps or episode_counter % save_frame_freq == 0:
                if env.render_mode == "rgb_array_list":
                    frames = env.render()
                if len(frames) > 0:
                    # arr = np.stack(frames)                      # (T,H,W,3)
                    # arr = arr.transpose(0,3,1,2)                 # (T,3,H,W)
                    # vid = torch.from_numpy(arr).unsqueeze(0)     # (1,T,3,H,W)
                    # writer.add_video(
                    #     f"Video/Episode_{episode_counter}_Terminated_{terminated}", vid, timestep, fps=24
                    # )  
                    
                    # write a gif
                    gif_path = os.path.join(writer.log_dir, f"episode_{episode_counter}.gif")
                    imageio.mimsave(gif_path, frames, fps=24)
                    print(f"Saved GIF to {gif_path}")
        
        if episode_return_org >= best_ep:
            best_ep = episode_return_org
            best_agent = copy.deepcopy(agent)

        results.append({
            "episode_return": episode_return_org,
            "episode_length": episode_length,
        })
        print(f"Episode: {episode_counter} Timestep: {timestep} Return Org: {episode_return_org:.2f} Return Wrapped: {episode_return_wrapped:.2f}")
        episode_counter += 1  
    
    print("Best Episode Return Org: ", best_ep)
    return results, best_agent

def agent_environment_episode_loop(env, agent, total_episodes, training=True, writer=None, save_frame_freq=1):
    """
    Runs the training loop for the agent in the given environment over a specified number of timesteps.
    Returns a list of dictionaries containing the episodic return and episode length.
    """
    greedy = False if training else True
    greedy = False
    timestep = 0
    results = []
    best_ep, best_agent = -np.inf, None

    for episode_counter in range(1, total_episodes+1):
        observation, info = env.reset()
        episode_return_org = 0.0
        episode_return_wrapped = 0.0
        episode_length = 0

        while True:
            action = agent.act(observation, greedy=greedy)
            next_observation, reward, terminated, truncated, info = env.step(action)
            if training:
                agent.update(next_observation, reward, terminated, truncated)
            episode_return_org += info["actual_reward"] if "actual_reward" in info else reward
            episode_return_wrapped += reward

            episode_length += 1
            timestep += 1
            
            if env.render_mode =="rgb_array" and (episode_counter >= total_episodes or episode_counter % save_frame_freq == 0): 
                frames.append(env.render())
            
            if terminated or truncated:
                break

            observation = next_observation

        # log to TensorBoard
        if writer is not None:
            writer.add_scalar(f"Training/EpisodeReturnOrg",
                              episode_return_org,
                              global_step=timestep)
            writer.add_scalar(f"Training/EpisodeReturnWrapped",
                              episode_return_wrapped,
                              global_step=timestep)

            if episode_counter >= total_episodes or episode_counter % save_frame_freq == 0:
                if env.render_mode =="rgb_array_list":
                    frames = env.render()
                if len(frames) > 0:
                    # arr = np.stack(frames)                      # (T,H,W,3)
                    # arr = arr.transpose(0,3,1,2)                 # (T,3,H,W)
                    # vid = torch.from_numpy(arr).unsqueeze(0)     # (1,T,3,H,W)
                    # writer.add_video(
                    #     f"Video/Episode_{episode_counter}_Terminated_{terminated}", vid, timestep, fps=24
                    # )  
                    
                    # write a gif
                    gif_path = os.path.join(writer.log_dir, f"episode_{episode_counter}.gif")
                    imageio.mimsave(gif_path, frames, fps=24)
                    print(f"Saved GIF to {gif_path}")

        if episode_return_org >= best_ep:
            best_ep = episode_return_org
            best_agent = copy.deepcopy(agent)

        results.append({
            "episode_return": episode_return_org,
            "episode_length": episode_length,
        })
        print(f"Episode: {episode_counter} Timestep: {timestep} Return Org: {episode_return_org:.2f} Return Wrapped: {episode_return_wrapped:.2f}")

    print("Best Episode Return Org: ", best_ep)
    return results, best_agent