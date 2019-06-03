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


def play_func(envs, params, net, cuda, exp_queue, device_id):
    """
    envs - either a single env or a list of envs. With multiple envs, the exp_source class will return experiences
    (defined as a tuple of (state_framestack, action, reward, last_state_framestack) alternating between
    the two environments. Otherwise it returns just experinces from a single env. Even if the games have different
    frame shapes, they will by reduced to 84x84
    """
    run_name = 'demon_invaders'
    if 'max_games' not in params:
        max_games = 16000
    else:
        max_games = params['max_games']

    device = torch.device("cuda:{}".format(device_id) if cuda else "cpu")

    writer = SummaryWriter(comment="-" + run_name + "-03_parallel")

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=params['gamma'], steps_count=1)
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
    args = parser.parse_args()
    cuda_id = args.cuda_id

    params = common.HYPERPARAMS['invaders']

    params['batch_size'] *= PLAY_STEPS
    device_str = "cuda:{}".format(cuda_id) if args.cuda else "cpu"
    print("Using device: {}".format(device_str))
    device = torch.device(device_str)

    if not os.path.exists('models'):
        os.makedirs('models')

    envSI = gym.make('SpaceInvadersNoFrameskip-v4')
    envSI = ptan.common.wrappers.wrap_dqn(envSI)

    envDA = gym.make('DemonAttackNoFrameskip-v4')
    envDA = ptan.common.wrappers.wrap_dqn(envDA)

    assert envSI.action_space.n == envDA.action_space.n, "Different Action Space Lengths"
    assert envSI.observation_space.shape == envDA.observation_space.shape, "Different Obs. Space Shapes"

    net = dqn_model.DQN(envSI.observation_space.shape, envSI.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    buffer = ptan.experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    exp_queue = mp.Queue(maxsize=PLAY_STEPS * 2)
    play_proc = mp.Process(target=play_func, args=([envSI,envDA], params, net, args.cuda, exp_queue, cuda_id))
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

        optimizer.zero_grad()
        batch = buffer.sample(params['batch_size'])
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], cuda=args.cuda, cuda_async=True, cuda_id=cuda_id)
        loss_v.backward()
        optimizer.step()

        if frame_idx % params['target_net_sync'] < PLAY_STEPS:
            tgt_net.sync()
