#!/usr/bin/env python3
import gym
import ptan
import argparse

import torch as T
import time

from lib import dqn_model, common
from gym import wrappers

import numpy as np
from gym.envs.classic_control import rendering

PLAY_STEPS = 4

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0:
        if not err:
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", help="Path to agent model", required=True)
    parser.add_argument('--env', help='Environment to load ("invaders", "assault", "demon-attack")', required=True)
    args = parser.parse_args()

    params = common.HYPERPARAMS[args.env]

    env = gym.make(params['env_name'])
    env_to_wrap = ptan.common.wrappers.wrap_dqn(env)
    env = wrappers.Monitor(env_to_wrap, args.env + '_movie', force = True)
    state = env.reset()

    viewer = rendering.SimpleImageViewer()

    expert = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    expert.load_state_dict(T.load(args.agent, map_location='cpu').state_dict())
    expert.eval()

    num_steps = 5000

    with T.no_grad():
        for _ in range(num_steps):
            state = T.Tensor(np.array(state))

            Q_vals = expert(state.unsqueeze(0))
            action = T.argmax(Q_vals)

            next_state, reward, done, info = env.step(action)
            rgb = env.render('rgb_array')
            viewer.imshow(repeat_upsample(rgb, 3, 3))

            state = next_state
            time.sleep(0.015)
            if done:
                break

    env.close()
    env_to_wrap.close()
