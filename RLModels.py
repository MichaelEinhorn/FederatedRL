import numpy as np
import gym
import ctypes
import copy
from numpy.random import default_rng

from core import softmax
import core

import collections
import multiprocessing as mp
import threading

import RLAlgs


class QTabular:
    def __init__(self, env, stochasticPolicy=False, seed=None, qV=None):
        self.NS = env.observation_space.n
        self.NA = env.action_space.n
        if qV is None:
            self.q_tab = np.zeros([self.NS, self.NA])
        else:
            self.q_tab = qV

        self.stoch = stochasticPolicy

        if seed is None:
            self.rng = default_rng()
        else:
            self.rng = default_rng(seed)

    def resetSeed(self, seed=None):
        if seed is None:
            self.rng = default_rng()
        else:
            self.rng = default_rng(seed)

    # stochastic none means use default behavior
    def policy(self, state, stochastic=None):
        stoch = self.stoch
        if stochastic is not None:
            stoch = stochastic
        if stoch:
            qV = softmax(self.q_tab[state])
            return self.rng.choice(range(self.NA), 1, p=qV)[0]
        else:
            return np.argmax(self.q_tab[state])

    # q learning step
    def backup(self, state, act, nextS, reward, alpha=0.1, gamma=0.6):
        q = self.q_tab[state, act]
        qmax = np.max(self.q_tab[nextS])
        q_next = (1 - alpha) * q + alpha * (reward + gamma * qmax)
        self.q_tab[state, act] = q_next

    def update(self, qV):
        self.q_tab = qV

    # bellman update
    def bellman(self, env, gamma=0.6):
        q_tab_old = copy.deepcopy(self.q_tab)
        maxQ_a = np.amax(q_tab_old, axis=1)
        if "Action" not in env.rewardType:
            for s in range(self.NS):
                for a in range(self.NA):
                    self.q_tab[s, a] = np.sum(env.transMat[s, a] * (env.rewardMat + gamma * maxQ_a))

    def diff(self, otherModel):
        return np.linalg.norm(self.q_tab - otherModel.q_tab)

    def getWeight(self):
        return self.q_tab

    def setWeight(self, weight):
        self.q_tab[:] = weight


class QTabularFedAvg():
    def __init__(self, shape, stochasticPolicy=False, p=10, seed=None, qV=None):
        self.P = p
        self.AggWeightQ = mp.JoinableQueue()
        self.returnQ = mp.Queue()
        self.aggs = 0
        self.shape = shape
        if qV is None:
            self.q_tab = np.zeros(shape)
        else:
            self.q_tab = qV

        tsize = 1
        for s in shape:
            tsize *= s
        self.shared_q_tab_mp = mp.Array(ctypes.c_double, tsize)
        self.shared_q_tab_np = np.frombuffer(self.shared_q_tab_mp.get_obj()).reshape(shape)
        self.shared_q_tab_np[:] = qV

        self.stoch = stochasticPolicy

        if seed is None:
            self.rng = default_rng()
        else:
            self.rng = default_rng(seed)

        self.runAgg = False
        self.done = False

    def start(self, QLearnF=None, QLearningKWArgs=None, model=None, env=None):
        self.done = False
        parr = [None] * self.P
        mpEnd = mp.Value(ctypes.c_bool, False)
        kwargs = QLearningKWArgs
        kwargs.update(core.getArgs(AggWeightQ=self.AggWeightQ, returnQ=self.returnQ, WeightSharedMem=self.shared_q_tab_mp, mpEnd=mpEnd))
        for i in range(self.P):
            # gives each thread its own seed
            modelTemp = copy.deepcopy(model)
            modelTemp.resetSeed(seed=self.rng.integers(2 ** 31 - 1))
            envTemp = copy.deepcopy(env)
            envTemp.reset(seed=self.rng.integers(2 ** 31 - 1))
            kwargsTemp = core.getArgs(env=envTemp, model=modelTemp)
            kwargsTemp.update(kwargs)

            parr[i] = mp.Process(target=QLearnF, kwargs=kwargsTemp)
            parr[i].start()

        # starts aggregate thread which averages weights
        aggThread = threading.Thread(target=self.aggregate, args=(True,))
        aggThread.daemon = True
        aggThread.start()

        out = [self.returnQ.get()]
        # print(out)
        # terminates aggregate
        self.AggWeightQ.put(None)
        mpEnd.value = True

        for i in range(1, self.P):
            out.append(self.returnQ.get())
            # print(out[i])

        self.done = True

        return out, self.aggs

    # averages weights sent to queue and updates shared weights
    def aggregate(self, loop=False):
        self.aggs = 0
        while True:
            arr = []
            for i in range(self.P):
                arr.append(self.AggWeightQ.get())
                if arr[i] is None:
                    break
            if arr[i] is None:
                break
            # locked
            # print("aggregating")
            self.aggs += 1
            self.q_tab = np.zeros(self.shape)
            for model in arr:
                self.q_tab += model
            self.q_tab /= len(arr)
            self.shared_q_tab_np[:] = self.q_tab
            # end
            for i in range(self.P):
                self.AggWeightQ.task_done()
            if not loop:
                break

        # prevents process hanging on join
        while not self.done:
            try:
                temp = self.AggWeightQ.get(timeout=1)
                self.AggWeightQ.task_done()
            except:
                temp = None


def cartPoleFeature(state, action, NF, NA):
    out = state * (action * 2 - 1)
    return out


class QLinAprox:
    def __init__(self, env, featureEx, stochasticPolicy=False, seed=None):
        ex = env.observation_space.high

        self.NF = ex.shape[0]
        self.NA = env.action_space.n
        self.feat = featureEx
        self.stoch = stochasticPolicy

        self.w = np.zeros([self.NA, self.NF])

        if seed is None:
            self.rng = default_rng()
        else:
            self.rng = default_rng(seed)

    def resetSeed(self, seed=None):
        if seed is None:
            self.rng = default_rng()
        else:
            self.rng = default_rng(seed)

    def policy(self, state, stochastic=None):
        qV = np.zeros(self.NA)
        for act in range(self.NA):
            qV[act] = np.sum(self.feat(state, act, self.NF, self.NA) * self.w[act])

        stoch = self.stoch
        if stochastic is not None:
            stoch = stochastic
        if stoch:
            qV = softmax(qV)
            return self.rng.choice(range(self.NA), 1, p=qV)[0]
        else:
            return np.argmax(qV)

    def backup(self, state, act, nextS, reward, alpha=0.1, gamma=0.6):
        self.w[act] = self.w[act] + alpha * reward * self.feat(state, act, self.NF, self.NA)

    def getW(self):
        return self.w

    def setW(self, w):
        self.w[:] = w


class RandomPolicy:
    def __init__(self, env):
        self.space = env.action_space
        self.w = None

    def policy(self, state, stochastic=None):
        return self.space.sample()

    def backup(self, state, act, nextS, reward, alpha=0.1, gamma=0.6):
        return

    def getW(self):
        return None

    def setW(self, w):
        return

