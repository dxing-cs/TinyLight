from env import TSCEnv
from typing import List
import itertools
import torch


def run_a_step(env: TSCEnv, n_obs: List[torch.Tensor], on_training: bool, store_experience: bool, learn: bool):
    n_action = []
    for agent in env.n_agent:
        action = agent.pick_action(n_obs, on_training)
        n_action.append(action)
    n_next_obs, n_rew, n_done, info = env.step(n_action)

    # optional: store experience
    if store_experience:
        for idx in range(env.n):
            env.n_agent[idx].store_experience(n_obs[idx], n_action[idx], n_rew[idx], n_next_obs[idx], n_done[idx])

    if learn:
        for agent in env.n_agent:
            agent.learn()

    return n_next_obs, n_rew, n_done, info


def run_an_episode(env: TSCEnv, config: dict, on_training: bool, store_experience: bool, learn: bool):
    n_obs = env.reset()
    n_done = [False]
    info = {}

    for config['current_episode_step_idx'] in itertools.count(start=0, step=config['action_interval']):
        if config['current_episode_step_idx'] >= config['num_step'] or all(n_done):
            break
        if config['current_episode_step_idx'] % 100 == 0:
            print('Episode {}, Step {}/{}'.format(config['current_episode_idx'], config['current_episode_step_idx'], config['num_step']), end='\r')

        n_next_obs, n_rew, n_done, info = run_a_step(env, n_obs, on_training, store_experience, learn)
        n_obs = n_next_obs
    return info


def quant_run_a_step(env: TSCEnv, n_obs: List[torch.Tensor], on_training: bool, store_experience: bool, learn: bool):
    n_action = []
    for agent in env.n_agent:
        action = agent.quan_pick_action(n_obs, on_training)
        n_action.append(action)
    n_next_obs, n_rew, n_done, info = env.step(n_action)
    n_obs = n_next_obs
    # optional: store experience
    if store_experience:
        for idx in range(env.n):
            env.n_agent[idx].store_experience(n_obs[idx], n_action[idx], n_rew[idx], n_next_obs[idx], n_done[idx])

    if learn:
        for agent in env.n_agent:
            agent.learn()

    return n_next_obs, n_rew, n_done, info


def quant_run_an_episode(env: TSCEnv, config: dict, on_training: bool, store_experience: bool, learn: bool):
    n_obs = env.reset()
    n_done = [False]
    info = {}

    for config['current_episode_step_idx'] in itertools.count(start=0, step=config['action_interval']):
        if config['current_episode_step_idx'] >= config['num_step'] or all(n_done):
            break
        if config['current_episode_step_idx'] % 100 == 0:
            print('Episode {}, Step {}/{}'.format(config['current_episode_idx'], config['current_episode_step_idx'], config['num_step']), end='\r')

        n_next_obs, n_rew, n_done, info = quant_run_a_step(env, n_obs, on_training, store_experience, learn)
        n_obs = n_next_obs
    return info
