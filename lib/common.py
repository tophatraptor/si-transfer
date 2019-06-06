import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HYPERPARAMS = {
    'pong': {
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      20.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    },
    'breakout-small': {
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout-small',
        'replay_size':      3*10 ** 5,
        'replay_initial':   20000,
        'target_net_sync':  1000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       64
    },
    'breakout': {
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout',
        'replay_size':      10 ** 6,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32,
    },
    'invaders': {
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'invaders',
        'replay_size': 3 * 10**5,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32, # originally 32
        'max_games': 16000
    },
    'assault': {
        'env_name':         "AssaultNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'assault',
        'replay_size':      3 * 10 ** 5,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32,
        'max_games': 16000,
        'save_iter': 1000,
    },
    'demon-attack': {
        'env_name':         "DemonAttackNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'demon-attack',
        'replay_size':      3 * 10 ** 5,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32,
        'max_games': 16000,
        'save_iter': 1000,
    },
    'invaders-am': {
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'invaders-am',
        'replay_size': 3 * 10**5,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 2 * 10 ** 6, # double
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32, # originally 32
        'max_games': 50000
    },
    'invaders-am2': {
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'invaders-am2',
        'replay_size': 3 * 10**5,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 5 * 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32, # originally 32
        'max_games': 50000
    },
}

HYPERPARAMS['assault_transfer'] = dict(HYPERPARAMS['assault'])
HYPERPARAMS['assault_transfer']['epsilon_start'] = 0.5
HYPERPARAMS['assault_transfer']['run_name'] = 'assault_transfer'

HYPERPARAMS['assault_transfer2'] = dict(HYPERPARAMS['assault_transfer'])
HYPERPARAMS['assault_transfer2']['epsilon_frames'] = 5 * 10**5
HYPERPARAMS['assault_transfer']['run_name'] = 'assault_transfer2'


def unpack_batch(batch):
    """
    Added an array for the environment names
    """
    states, actions, rewards, dones, last_states, envs = [], [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if hasattr(exp, 'env'):
            envs.append(exp.env)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))


    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
       np.array(dones, dtype=np.uint8), np.array(last_states, copy=False), np.array(envs)



def calc_loss_dqn(batch, net, tgt_net, gamma, cuda=False, cuda_async=False, cuda_id=0):
    """
    Added array for env name
    """
    states, actions, rewards, dones, next_states, envs = unpack_batch(batch)

    states_v = torch.tensor(states)
    next_states_v = torch.tensor(next_states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    done_mask = torch.ByteTensor(dones)
    if cuda:
        device = torch.device("cuda:{}".format(cuda_id))
        states_v = states_v.cuda(device=device, non_blocking=cuda_async)
        next_states_v = next_states_v.cuda(device=device, non_blocking=cuda_async)
        actions_v = actions_v.cuda(device=device, non_blocking=cuda_async)
        rewards_v = rewards_v.cuda(device=device, non_blocking=cuda_async)
        done_mask = done_mask.cuda(device=device, non_blocking=cuda_async)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def calc_loss_actormimic(batch, net, name_to_expert, beta, cuda=False, cuda_async=False, cuda_id=0):
    """
    Don't get the loss from the standard target network. Instead, get it by comparing output to two expert nets.
    Therefore we only need the state frame stack

    name_to_expert: dict mapping name of env that will be seen in the experience_AM tuples to the expert model.
    """
    states, actions, rewards, dones, next_states, envs = unpack_batch(batch)

    total_loss = 0

    for env_name in name_to_expert.keys():
        expert_model, expert_model_hidden = name_to_expert[env_name]

        states_v = torch.tensor(states[envs == env_name])
        if cuda:
            device = torch.device("cuda:{}".format(cuda_id))
            states_v = states_v.cuda(device=device, non_blocking=cuda_async)

        with torch.no_grad():
            expert_Q_softmax = F.softmax(expert_model(states_v), dim=1)
            expert_hidden = expert_model_hidden(states_v)

        learner_Q, learner_hidden = net(states_v, hidden_bool=True)
        learner_Q_softmax = F.softmax(learner_Q, dim=1)

        # Pytorch's Cross Entropy Loss takes in a class id as the target, which we don't want
        #this_loss = ( expert_Q_softmax * learner_Q_log_softmax).sum()

        this_loss = nn.MSELoss()(learner_Q_softmax, expert_Q_softmax) + beta*nn.MSELoss()(learner_hidden, expert_hidden)

        total_loss += this_loss

    return total_loss


class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        bool_status = False
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            bool_status = True
        return [bool_status, len(self.total_rewards), mean_reward, epsilon_str]


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)
