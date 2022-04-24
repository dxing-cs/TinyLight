from env import TSCEnv
from utilities.utils import set_seed, get_agent, get_config, set_thread, set_logger, release_logger, make_dir
from utilities.snippets import run_an_episode
import numpy as np
import os
import multiprocessing
import argparse


parser = argparse.ArgumentParser(description='Training TinyLight')
parser.add_argument('--dataset', type=str, default='Hangzhou1', help='one of [Hangzhou1, Hangzhou2, Hangzhou3, Atlanta, Jinan, LosAngeles]')
args = parser.parse_args()
DEBUG = False
i_name = args.dataset

cur_agent = 'TinyLight'
regular_training = 64
freeze_time = 128


def run_an_experiment(inter_name, flow_idx, seed):
    dic_num_step = {'Atlanta': 900, 'Hangzhou1': 3600, 'Hangzhou2': 3600, 'Hangzhou3': 3600, 'Jinan': 3600, 'LosAngeles': 1800}
    config = get_config()
    config['inter_name'] = inter_name
    config.update({
        'seed': seed,
        'flow_idx': flow_idx,
        'save_result': not DEBUG,
        'dir': 'data/{}/'.format(inter_name),
        'flowFile': 'flow_{}.json'.format(flow_idx),
        'cur_agent': cur_agent,
        'render': False,
        'num_step': dic_num_step[inter_name] if inter_name in dic_num_step.keys() else 3600,
    })
    set_seed(config['seed'])
    set_thread()
    set_logger(config)

    env = TSCEnv(config)
    env.n_agent = []
    for idx in range(env.n):
        agent = get_agent(config['cur_agent'])(config, env, idx)
        env.n_agent.append(agent)

    ################################
    # Regular training
    ################################
    epsilon = config[config['cur_agent']]['epsilon']
    total_train_episode = regular_training
    for config['current_episode_idx'] in range(1, total_train_episode + 1):
        config[config['cur_agent']]['epsilon'] = max(0, epsilon - epsilon / (total_train_episode - 5) * config['current_episode_idx'])
        info = run_an_episode(env, config, on_training=True, store_experience=True, learn=True)
        config['logger'].info('[{} On Training] Inter: {}; Flow: {}; Episode: {}; ATT: {:.2f}; AQL: {:.2f}; AD: {:.2f}; Throughput: {:.2f}'.format(
            config['cur_agent'], inter_name, flow_idx, config['current_episode_idx'],
            info['world_2_average_travel_time'][0],
            info['world_2_average_queue_length'][0],
            info['world_2_average_delay'][0],
            info['world_2_average_throughput'][0],
        ))

        for agent in env.n_agent:
            config['logger'].info(agent.get_alpha_desc())

    ######################################
    # Hard thresholding and freezing alpha
    ######################################
    for agent in env.n_agent:
        agent.hard_threshold_and_freeze_alpha()

    ################################
    # Training after freezing
    ################################
    config[config['cur_agent']]['epsilon'] = 0.0
    total_train_episode_after_freezing = freeze_time
    for config['current_episode_idx'] in range(1, total_train_episode_after_freezing + 1):
        info = run_an_episode(env, config, on_training=True, store_experience=True, learn=True)
        config['logger'].info('[{} On Freezing] Inter: {}; Flow: {}; Episode: {}; ATT: {:.2f}; AQL: {:.2f}; AD: {:.2f}; Throughput: {:.2f}'.format(
            config['cur_agent'], inter_name, flow_idx, config['current_episode_idx'],
            info['world_2_average_travel_time'][0],
            info['world_2_average_queue_length'][0],
            info['world_2_average_delay'][0],
            info['world_2_average_throughput'][0],
        ))

        for agent in env.n_agent:
            config['logger'].info(agent.get_alpha_desc())

        if config['current_episode_idx'] > 1:
            for agent in env.n_agent:
                agent.network_lr_scheduler.step()

        config['logger'].info('[Learning Rate] {}'.format(env.n_agent[0].network_optim.param_groups[0]['lr']))

    ################################
    # Evaluation
    ################################
    info = run_an_episode(env, config, on_training=False, store_experience=False, learn=False)
    config['logger'].info('[{} On Evaluation] Inter: {}; Flow: {}; Episode: {}; ATT: {:.2f}; AQL: {:.2f}; AD: {:.2f}; Throughput: {:.2f}'.format(
        config['cur_agent'], inter_name, flow_idx, config['current_episode_idx'],
        info['world_2_average_travel_time'][0],
        info['world_2_average_queue_length'][0],
        info['world_2_average_delay'][0],
        info['world_2_average_throughput'][0],
    ))

    if config['save_result']:
        model_path = os.path.join(config['log_path'], 'model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for agent_idx, agent in enumerate(env.n_agent):
            agent.save(os.path.join(model_path, 'flow_{}_agent_{}.pth'.format(flow_idx, agent_idx)))

    release_logger(config)
    return info['world_2_average_travel_time'][0], \
           info['world_2_average_queue_length'][0], \
           info['world_2_average_delay'][0], \
           info['world_2_average_throughput'][0]


if __name__ == '__main__':
    if not DEBUG:
        make_dir('log/{}/{}/'.format(i_name, cur_agent))
    parallel = not DEBUG
    total_run = 10
    metrics = {
        'travel_time': [None for _ in range(total_run)],
        'queue_length': [None for _ in range(total_run)],
        'delay': [None for _ in range(total_run)],
        'throughput': [None for _ in range(total_run)]
    }
    seed_list = [992832, 284765, 905873, 776383, 198876, 192223, 223341, 182228, 885746, 992817]

    if parallel:
        with multiprocessing.Pool(processes=total_run) as pool:
            n_return_value = pool.starmap(run_an_experiment, [(i_name, f_idx, seed_list[f_idx]) for f_idx in range(10)])
            for f_idx, return_value in enumerate(n_return_value):
                metrics['travel_time'][f_idx] = return_value[0]
                metrics['queue_length'][f_idx] = return_value[1]
                metrics['delay'][f_idx] = return_value[2]
                metrics['throughput'][f_idx] = return_value[3]
    else:
        for f_idx in range(0, 10):
            return_value = run_an_experiment(inter_name=i_name, flow_idx=f_idx, seed=seed_list[f_idx])
            metrics['travel_time'][f_idx] = return_value[0]
            metrics['queue_length'][f_idx] = return_value[1]
            metrics['delay'][f_idx] = return_value[2]
            metrics['throughput'][f_idx] = return_value[3]

    print('att: {:.2f}±{:.2f}; aql: {:.2f}±{:.2f}; ad: {:.2f}±{:.2f}; ath: {:.2f}±{:.2f}'.format(
        np.mean(metrics['travel_time']), np.std(metrics['travel_time']),
        np.mean(metrics['queue_length']), np.std(metrics['queue_length']),
        np.mean(metrics['delay']), np.std(metrics['delay']),
        np.mean(metrics['throughput']), np.std(metrics['throughput'])
    ))

    print('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
        np.mean(metrics['travel_time']), np.std(metrics['travel_time']),
        np.mean(metrics['queue_length']), np.std(metrics['queue_length']),
        np.mean(metrics['delay']), np.std(metrics['delay']),
        np.mean(metrics['throughput']), np.std(metrics['throughput'])
    ))

    if not DEBUG:
        with open('log/{}/{}/summary.txt'.format(i_name, cur_agent), 'a') as fout:
            fout.write('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
                np.mean(metrics['travel_time']), np.std(metrics['travel_time']),
                np.mean(metrics['queue_length']), np.std(metrics['queue_length']),
                np.mean(metrics['delay']), np.std(metrics['delay']),
                np.mean(metrics['throughput']), np.std(metrics['throughput'])
            ))
