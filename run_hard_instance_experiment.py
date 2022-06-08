import argparse
import os
import numpy as np
import pandas as pd
import warnings

from algorithms import THLassoBandit
from collections import defaultdict
from envs.hard_instance import HardInstance

warnings.filterwarnings(action='ignore')


def run_exp(num_trial, T, d, rho_sq, alg):
    trajectory = defaultdict(list)

    for n in range(num_trial):
        print('==== Run trial {} ===='.format(n))
        rng = np.random.RandomState(np.random.randint(0, 2 ** 32))
        alg_ins = alg[0](rng, **alg[1])
        cum_reward = 0
        cum_regret = 0
        log = defaultdict(list)

        # run each trial
        env = HardInstance(d, rho_sq, rng)
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
    parser = argparse.ArgumentParser(description='Main script for experiments with a hard instance')
    parser.add_argument('--T', type=int, default=1000, help='number of rounds')
    parser.add_argument('--d', type=int, default=1000,
                        help='dimension of feature vectors')
    parser.add_argument('--rho_sq', type=float, default=0.7,
                        help='correlation level between feature vectors of arms')
    parser.add_argument('--num_trial', type=int, default=20,
                        help='number of trials to run experiments.')
    args = parser.parse_args()

    T = args.T
    d = args.d
    rho_sq = args.rho_sq

    # define algorithm
    alg = (THLassoBandit, {'K': 3, 'd': d, 'lam0': 0.02})

    # run experiments
    print('===== Run experiments over {} trials ====='.format(args.num_trial))
    trajectory = run_exp(args.num_trial, T, d, rho_sq, alg)

    # save log
    save_path = 'log/hard_instance/d{}_rho{}/'.format(d, rho_sq)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for k, v in trajectory.items():
        df = pd.DataFrame(np.array(v).T)
        df.index.name = '#index'
        df.to_csv(save_path + k + '.csv')


if __name__ == '__main__':
    main()
