import numpy as np
import random
import gym


def score(initState=None, iterN=100, env=None, model=None, epLimit=-1):
    rewards = np.zeros(iterN)
    for e in range(iterN):
        if initState is None:
            state = env.reset()
        else:
            state = initState
            # need to set state of env, not always supported
        term = False
        i = -1
        reward = 0
        while not term and (epLimit == -1 or i < epLimit):
            i += 1
            act = model.policy(state)
            state, r, term, info = env.step(act)
            reward += r
        rewards[e] = reward
    return np.mean(rewards), rewards


def QLearning(initState=None, iterN=100000, env=None, model=None,
              eps=0.1, a=0.1, y=0.6, epLimit=-1, convN=5, convThresh=0.01):
    sims = 0
    backups = 0
    rewards = np.zeros(iterN)

    for e in range(iterN):

        if initState is None:
            state = env.reset()
        else:
            state = initState
            # need to set state of env, not always supported

        term = False
        i = -1
        reward = 0
        while not term and (epLimit == -1 or i < epLimit):
            i += 1
            if random.uniform(0, 1) < eps:
                act = env.action_space.sample()
            else:
                act = model.policy(state)

            sims += 1
            nextS, r, term, info = env.step(act)
            reward += r

            backups += 1
            model.backup(state, act, nextS, r, a, y)

            state = nextS

        rewards[e] = reward
        if e % convN == 0:
            print("episode " + str(e) + " r " + str(reward))
        # converges if average reward of convN episodes is the same as the previous convN episodes
            if reward > 0 and e >= 2 * convN:
                avg = np.mean(rewards[e-convN+1:e+1])
                prev = np.mean(rewards[e-2*convN:e-convN])
                diff = abs(avg - prev)
                if diff < convThresh and avg > 0:
                    print("reward difference " + str(diff))
                    return model, sims, backups, e, avg, rewards[:e+1]


class QTabular:
    def __init__(self, env):
        self.NS = env.observation_space.n
        self.NA = env.action_space.n
        self.q_tab = np.zeros([self.NS, self.NA])
        self.q_tab_old = np.zeros([self.NS, self.NA])

    def policy(self, state):
        return np.argmax(self.q_tab[state])

    # returns param change size
    def backup(self, state, act, nextS, r, a=0.1, y=0.6):
        q = self.q_tab[state, act]
        qmax = np.max(self.q_tab[nextS])
        nq = (1 - a) * q + a * (r + y * qmax)
        self.q_tab[state, act] = nq
