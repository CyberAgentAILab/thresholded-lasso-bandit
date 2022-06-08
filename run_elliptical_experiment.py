import argparse
import os
import numpy as np
import pandas as pd
import warnings

from algorithms import THLassoBandit
from collections import defaultdict
from envs.elliptical import Elliptical

warnings.filterwarnings(action='ignore')


def run_exp(num_trial, K, T, d, s0, l, alg):
    trajectory = defaultdict(list)

    for n in range(num_trial):
        print('==== Run trial {} ===='.format(n))
        rng = np.random.RandomState(np.random.randint(0, 2 ** 32))
        alg_ins = alg[0](rng, **alg[1])
        cum_reward = 0
        cum_regret = 0
        log = defaultdict(list)

        # run each trial
        env = Elliptical(K, d, s0, l, rng)
        for t in range(T):
            if t % 100 == 0:
                print('trial: {}, round: {}, regret: {}'.format(n, t, cum_regret))
            # receive context set
            x = env.context()
            # pull arm and observe reward
            action = alg_ins.choose_action(x, t + 1)
            reward, regret = env.pull(action)
            # update policy
            alg_ins.update_beta(reward, t + 1)
            # record log
            cum_reward += reward
            cum_regret += regret
            log['rewards'].append(cum_reward)
            log['regrets'].append(cum_regret)
            log['false_negative'].append(env.false_negative(alg_ins.beta))
            log['false_positive'].append(env.false_positive(alg_ins.beta))
            log['error_l1'].append(env.error_l1(alg_ins.beta))
            log['error_l2'].append(env.error_l2(alg_ins.beta))
        for k, v in log.items():
            trajectory[k].append(v)
    return trajectory


def main():
    parser = argparse.ArgumentParser(description='Main script for experiments with an elliptical distribution')
    parser.add_argument('--K', type=int, default=2, help='number of arms')
    parser.add_argument('--T', type=int, default=1000, help='number of rounds')
    parser.add_argument('--d', type=int, default=1000,
                        help='dimension of feature vectors')
    parser.add_argument('--s0', type=int, default=5, help='sparsity index')
    parser.add_argument('--l', type=int, default=200,
                        help='parameter of an elliptical distribution')
    parser.add_argument('--num_trial', type=int, default=20,
                        help='number of trials to run experiments.')
    args = parser.parse_args()

    K = args.K
    T = args.T
    d = args.d
    s0 = args.s0
    l = args.l

    # define algorithm
    alg = (THLassoBandit, {'K': K, 'd': d, 'lam0': 0.02})

    # run experiments
    print('===== Run experiments over {} trials ====='.format(args.num_trial))
    trajectory = run_exp(args.num_trial, K, T, d, s0, l, alg)

    # save log
    save_path = 'log/elliptical/K{}_d{}_s{}_l{}/'.format(K, d, s0, l)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for k, v in trajectory.items():
        df = pd.DataFrame(np.array(v).T)
        df.index.name = '#index'
        df.to_csv(save_path + k + '.csv')


if __name__ == '__main__':
    main()
