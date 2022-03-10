import numpy as np
import random
import gym


def score(initState=None, iterN=100, env=None, model=None, epLimit=-1, printErr=False):
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
        if epLimit != -1 and i >= epLimit and printErr:
            print("testing episode " + str(e) + " timed out with r " + str(reward))
        rewards[e] = reward
    return np.mean(rewards), rewards


def QLearning(initState=None, iterN=100000, env=None, model=None,
              eps=0.1, a=0.1, y=0.6, epLimit=-1, convN=5, convThresh=0.01):
    sims = 0
    backups = 0
    rewards = np.zeros(iterN)
    rewardsAvg = []
    avg = -1000

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
            # sList = []
            # for sI in env.decode(state):
            #     sList.append(sI)
            # print(np.array(sList))
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

        # convergence testing
        if convN != -1 and e % convN == 0:

            # if True:
            #     # tests without exploration or updating during the episode
            #     prev = avg
            #     avg, _ = score(iterN=20, model=model, env=env, epLimit=1000)
            #     print("episode " + str(e) + " r " + str(avg))

            if e >= 2 * convN:
                # converges if average reward of convN episodes is the same as the previous convN episodes
                avg = np.mean(rewards[e-convN:e+1])
                print("episode " + str(e) + " r " + str(avg))
                prev = np.mean(rewards[e-2*convN:e-convN])

                rewardsAvg.append(avg)
                diff = abs(avg - prev)
                if diff < convThresh and avg > 0:
                    print("reward difference " + str(diff))
                    return model, sims, backups, e, avg, rewards[:e+1], np.array(rewardsAvg)

    avg = np.mean(rewards[e - convN + 1:])
    rewardsAvg.append(avg)
    return model, sims, backups, e, avg, rewards, np.array(rewardsAvg)


class QTabular:
    def __init__(self, env):
        self.NS = env.observation_space.n
        self.NA = env.action_space.n
        self.q_tab = np.zeros([self.NS, self.NA])
        self.q_tab_old = np.zeros([self.NS, self.NA])

    def policy(self, state):
        return np.argmax(self.q_tab[state])

    def backup(self, state, act, nextS, r, a=0.1, y=0.6):
        q = self.q_tab[state, act]
        qmax = np.max(self.q_tab[nextS])
        nq = (1 - a) * q + a * (r + y * qmax)
        self.q_tab[state, act] = nq


def cartPoleFeature(state, action, NF, NA):
    out = state * (action * 2 - 1)
    return out


class QLinAprox:
    def __init__(self, env, featureEx):
        ex = env.observation_space.high

        self.NF = ex.shape[0]
        self.NA = env.action_space.n
        self.feat = featureEx

        self.w = np.zeros([self.NA, self.NF])

    def policy(self, state):
        qV = np.zeros(self.NA)
        for act in range(self.NA):
            qV = np.sum(self.feat(state, act, self.NF, self.NA) * self.w[act])
        return np.argmax(qV)

    def backup(self, state, act, nextS, r, a=0.1, y=0.6):
        self.w[act] = self.w[act] + a * r * self.feat(state, act, self.NF, self.NA)