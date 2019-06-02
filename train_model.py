#!/usr/bin/env python3
import gym
import ptan
import argparse

import torch
import torch.optim as optim
import torch.multiprocessing as mp

import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

from lib import dqn_model, common

import csv
import numpy as np
import os

PLAY_STEPS = 4


def play_func(params, net, cuda, exp_queue, device_id):
    env_name = params['env_name']
    run_name = params['run_name']
    if 'max_games' not in params:
        max_games = 16000
    else:
        max_games = params['max_games']
    env = gym.make(env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    device = torch.device("cuda:{}".format(device_id) if cuda else "cpu")

    writer = SummaryWriter(comment="-" + params['run_name'] + "-03_parallel")

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    exp_source_iter = iter(exp_source)

    fh = open('models/{}_metadata.csv'.format(run_name), 'w')
    out_csv = csv.writer(fh)

    frame_idx = 0
    game_idx = 1
    model_count = 0
    model_stats = []
    mean_rewards = []
    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            epsilon_tracker.frame(frame_idx)
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                status, num_games, mean_reward, epsilon_str = reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon)
                mean_rewards.append(mean_reward)
                if status:
                    break
                if game_idx and (game_idx % 500 == 0):
                    # write to disk
                    print("Saving model...")
                    model_name = 'models/{}_{}.pth'.format(run_name, game_idx)
                    net.to(torch.device('cpu'))
                    torch.save(net, model_name)
                    net.to(device)
                    new_row = [model_name, num_games, mean_reward, epsilon_str]
                    out_csv.writerow(new_row)
                    np.savetxt('models/{}_reward.txt'.format(run_name), np.array(mean_rewards))
                if game_idx == max_games:
                    break
                game_idx += 1

    print("Saving final model...")
    model_name = 'models/{}_{}.pth'.format(run_name, game_idx)
    net.to(torch.device('cpu'))
    torch.save(net, model_name)
    net.to(device)
    new_row = [model_name, num_games, mean_reward, epsilon_str]
    out_csv.writerow(new_row)
    np.savetxt('models/{}_reward.txt'.format(run_name), np.array(mean_rewards))
    # plt.figure(figsize=(16, 9))
    # plt.tight_layout()
    # plt.title('Reward vs time, {}'.format(run_name))
    # plt.xlabel('Iteration')
    # plt.ylabel('Reward')
    # ys = np.array(mean_rewards)
    # plt.plot(ys, c='r')
    # plt.savefig('models/{}_reward.png'.format(run_name))
    # plt.close()
    fh.close()

    exp_queue.put(None)


if __name__ == "__main__":
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--cuda_id", default=0, help="CUDA ID of device")
    parser.add_argument("--env", default='pong', help="Environment of either ['pong', 'invaders', 'demon-attack', 'assault']")
    args = parser.parse_args()
    cuda_id = args.cuda_id
    assert args.env in ['pong', 'invaders', 'demon-attack', 'assault']
    params = common.HYPERPARAMS[args.env]
    params['batch_size'] *= PLAY_STEPS
    device_str = "cuda:{}".format(cuda_id) if args.cuda else "cpu"
    print("Using device: {}".format(device_str))
    device = torch.device(device_str)

    if not os.path.exists('models'):
        os.makedirs('models')

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    print("Loaded Environment: {}".format(params['env_name']))

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)

    buffer = ptan.experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    exp_queue = mp.Queue(maxsize=PLAY_STEPS * 2)
    play_proc = mp.Process(target=play_func, args=(params, net, args.cuda, exp_queue, cuda_id))
    play_proc.start()

    frame_idx = 0

    while play_proc.is_alive():
        frame_idx += PLAY_STEPS
        for _ in range(PLAY_STEPS):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)

        if len(buffer) < params['replay_initial']:
            continue
        # print("UPDATING GRAD")
        optimizer.zero_grad()
        batch = buffer.sample(params['batch_size'])
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], cuda=args.cuda, cuda_async=True, cuda_id=cuda_id)
        loss_v.backward()
        optimizer.step()

        if frame_idx % params['target_net_sync'] < PLAY_STEPS:
            tgt_net.sync()
