import numpy as np
import gym
from numpy.random import default_rng
from utils import softmax


class RandomMDP:
    def __init__(self, NS, NA, transition="NormPowerSoftMax", reward="NormNextState", seed=None, transitionParam=15, rewardParam=1):
        self.observation_space = gym.spaces.Discrete(NS)
        self.action_space = gym.spaces.Discrete(NA)
        self.state = 0
        self.rewardType = reward
        if seed is None:
            self.rng = default_rng()
        else:
            self.rng = default_rng(seed)
        # state action next state
        self.transMat = np.zeros([NS, NA, NS])
        if transition == "NormSoftMax":
            self.transMat = self.rng.standard_normal(self.transMat.shape)
            self.transMat = softmax(self.transMat, axis=2)
            # print(self.transMat)

        if transition == "NormPowerSoftMax":
            self.transMat = self.rng.standard_normal(self.transMat.shape)
            self.transMat = np.power(self.transMat, transitionParam)
            self.transMat = softmax(self.transMat, axis=2)

        print("reward")
        print(self.transMat)

        if self.rewardType == "NormNextState":
            self.rewardMat = np.zeros([NS])
            self.rewardMat = self.rng.normal(loc=0, scale=rewardParam, size=self.rewardMat.shape)
        elif self.rewardType == "ExpNextState":
            self.rewardMat = np.zeros([NS])
            self.rewardMat = self.rng.exponential(scale=rewardParam, size=self.rewardMat.shape)
        elif self.rewardType == "NormActionNextState":
            self.rewardMat = np.zeros([NA, NS])
            self.rewardMat = self.rng.standard_normal(self.rewardMat.shape)

        print(self.rewardMat)

    def reset(self, seed=None):
        self.state = 0
        if seed is not None:
            self.rng = default_rng(seed)
        return 0

    def step(self, act):
        priorS = self.state
        self.state = self.rng.choice(range(self.observation_space.n), 1, p=self.transMat[priorS, act])[0]
        term = False
        if "Action" not in self.rewardType:
            r = self.rewardMat[self.state]
        else:
            r = self.rewardMat[act, self.state]

        return self.state, r, term, None
