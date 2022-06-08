import argparse
import os
import numpy as np
import pandas as pd
import warnings

from algorithms import THLassoBandit
from collections import defaultdict
from envs.uniform import Uniform

warnings.filterwarnings(action='ignore')


def run_exp(num_trial, K, T, d, s0, algs):
    trajectories = [defaultdict(list) for _ in algs]

    for n in range(num_trial):
        print('==== Run trial {} ===='.format(n))
        rng = np.random.RandomState(np.random.randint(0, 2 ** 32))
        alg_ins = [algs[i][0](rng, **algs[i][1]) for i in range(len(algs))]
        cum_rewards = [0 for _ in range(len(algs))]
        cum_regrets = [0 for _ in range(len(algs))]
        log = [defaultdict(list) for _ in range(len(algs))]

        # run each trial
        env = Uniform(K, d, s0, rng)
        for t in range(T):
            if t % 100 == 0:
                cum_regrets_str = ', '.join(['{} regret: {}'.format(algs[i][0].__name__, cum_regrets[i]) for i in range(len(alg_ins))])
                print('trial: {}, round: {}, {}'.format(n, t, cum_regrets_str))
            # receive context set
            x = env.context()
            for i in range(len(alg_ins)):
                # pull arm and observe reward
                action = alg_ins[i].choose_action(x, t + 1)
                reward, regret = env.pull(action)
                # update policy
                alg_ins[i].update_beta(reward, t + 1)
                # record log
                cum_rewards[i] += reward
                cum_regrets[i] += regret
                log[i]['rewards'].append(cum_rewards[i])
                log[i]['regrets'].append(cum_regrets[i])
                log[i]['false_negative'].append(env.false_negative(alg_ins[i].beta))
                log[i]['false_positive'].append(env.false_positive(alg_ins[i].beta))
                log[i]['error_l1'].append(env.error_l1(alg_ins[i].beta))
                log[i]['error_l2'].append(env.error_l2(alg_ins[i].beta))
        for i in range(len(algs)):
            for k, v in log[i].items():
                trajectories[i][k].append(v)
    return trajectories


def main():
    parser = argparse.ArgumentParser(description='Main script for experiments with an uniform distribution')
    parser.add_argument('--K', type=int, default=2, help='number of arms')
    parser.add_argument('--T', type=int, default=1000, help='number of rounds')
    parser.add_argument('--d', type=int, default=1000,
                        help='dimension of feature vectors')
    parser.add_argument('--s0', type=int, default=20, help='sparsity index')
    parser.add_argument('--num_trial', type=int, default=20,
                        help='number of trials to run experiments.')
    args = parser.parse_args()

    K = args.K
    T = args.T
    d = args.d
    s0 = args.s0

    # define algorithms
    algs = [
        (THLassoBandit, {'K': K, 'd': d, 'lam0': 0.02})
    ]

    # run experiments
    print('===== Run experiments over {} trials ====='.format(args.num_trial))
    trajectories = run_exp(args.num_trial, K, T, d, s0, algs)

    # save log
    for i in range(len(algs)):
        save_path = 'log/uniform/K{}_d{}_s{}/{}/'.format(K, d, s0, algs[i][0].__name__)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        for k, v in trajectories[i].items():
            df = pd.DataFrame(np.array(v).T)
            df.index.name = '#index'
            df.to_csv(save_path + k + '.csv')


if __name__ == '__main__':
    main()
