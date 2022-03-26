import gym
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import RLAlgs
import RLModels
import utils
import copy

import CustomEnvs


# smaller random MDP to be solved exactly with bellman
def mdpTest():
    env = CustomEnvs.RandomMDP(6, 5, seed=42)
    envBell = CustomEnvs.RandomMDP(6, 5, seed=42)
    print('NS:' + str(env.observation_space.n) + ' NA:' + str(env.action_space.n))
    model = RLModels.QTabular(env, stochasticPolicy=True)
    modelBell = RLModels.QTabular(env, stochasticPolicy=True)

    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, y=0.6, iterN=10000)
    print("bellman")
    print(modelBell.q_tab)

    model, sims, backups, episodes, avgR, rList, diffs = RLAlgs.QLearning(env=env, model=model, eps=1, epLimit=100,
                                                                          a=0.1, y=0.6, convN=100, convThresh=0.01,
                                                                          convMethod="compare", otherModel=modelBell,
                                                                          logging=True)

    print("learning")
    print(model.q_tab)
    print("bellman")
    print(modelBell.q_tab)
    print("diff " + str(model.diff(modelBell)))

    print("simulation steps " + str(sims))
    print("episodes " + str(episodes))
    print("avg train reward " + str(avgR))

    score, sList = RLAlgs.score(model=model, env=env, epLimit=100, printErr=False)
    print("avg reward " + str(score))

    plt.plot(range(diffs.shape[0]), diffs)
    plt.gca().set(ylim=(0, 1))
    plt.savefig("trainingMDP.png")


def mdpTestFed():
    env = CustomEnvs.RandomMDP(6, 5, seed=42)
    envBell = CustomEnvs.RandomMDP(6, 5, seed=42)
    print('NS:' + str(env.observation_space.n) + ' NA:' + str(env.action_space.n))
    model = RLModels.QTabular(env, stochasticPolicy=True)
    modelBell = RLModels.QTabular(env, stochasticPolicy=True)

    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, y=0.6, iterN=10000)
    print("bellman")
    print(modelBell.q_tab)

    kwArgs = utils.getArgs(env=env, model=model, eps=1, epLimit=100,
                           a=0.1, y=0.6, convN=100, convThresh=0.01, convMethod="compare", otherModel=modelBell, logging=True,
                           syncB=-1, syncE=200)
    print(kwArgs)

    federatedModel = RLModels.QTabularFedAvg(model.getW().shape, stochasticPolicy=True, p=10)

    out, aggs = federatedModel.start(RLAlgs.QLearning, QLearningKWArgs=kwArgs)
    model, sims, backups, episodes, avgR, rList, diffs = out[0]

    print("learning")
    print(model.q_tab)
    print("bellman")
    print(modelBell.q_tab)
    print("diff " + str(model.diff(modelBell)))

    print("simulation steps " + str(sims))
    print("episodes " + str(episodes))
    print("avg train reward " + str(avgR))

    score, sList = RLAlgs.score(model=model, env=env, epLimit=100, printErr=False)
    print("avg reward " + str(score))

    plt.plot(range(diffs.shape[0]), diffs)
    plt.gca().set(ylim=(0, 1))
    plt.savefig("trainingMDPFed.png")


def mdpTestTD():
    envBell = CustomEnvs.RandomMDP(6, 5, seed=42)
    modelBell = RLModels.QTabular(envBell, stochasticPolicy=True)

    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, y=0.6, iterN=1000)
    print("bellman")
    print(modelBell.q_tab)

    envTD = CustomEnvs.RandomMDP(6, 5, seed=42)
    q_tabTD, sims, backups = RLAlgs.TDLearnNStep(env=envTD, n=10, a=0.1, y=0.6, lam=0.9, eps=0, iterN=10000,
                                                 epLimit=100, trace="accumulate",
                                                 stoch=True, logging=True, otherModel=modelBell)
    print(q_tabTD)


# small discrete space
def taxiTest():
    env = gym.make("Taxi-v3").env
    print('NS:' + str(env.observation_space.n) + ' NA:' + str(env.action_space.n))
    model = RLModels.QTabular(env, stochasticPolicy=True)
    model, sims, backups, episodes, avgR, rList, rewardsAvg = RLAlgs.QLearning(env=env, model=model, eps=0.1,
                                                                               a=0.1, y=0.6, convN=100, convThresh=0.01)
    print("simulation steps " + str(sims))
    print("episodes " + str(episodes))
    print("avg train reward " + str(avgR))

    score, sList = RLAlgs.score(model=model, env=env, epLimit=1000, printErr=True)
    print("avg reward " + str(score))

    plt.plot(range(rewardsAvg.shape[0]), rewardsAvg)
    plt.gca().set(ylim=(-20, 20))
    plt.savefig("trainingTaxi.png")


def cartTest():
    env = gym.make("CartPole-v1").env
    model = RLModels.QLinAprox(env, RLModels.cartPoleFeature, stochasticPolicy=True)
    # model = RLAlgs.RandomPolicy(env)
    model, sims, backups, episodes, avgR, rList, rewardsAvg = RLAlgs.QLearning(env=env, model=model, iterN=1000, eps=0,
                                                                               a=0.01, y=0.6, convN=-1, convThresh=0.01)

    print("train rewards")
    print(rList)
    print("weights")
    print(model.w)
    print("simulation steps " + str(sims))
    print("episodes " + str(episodes))
    print("avg train reward " + str(avgR))

    score, sList = RLAlgs.score(model=model, env=env, epLimit=1000, printErr=True)
    print("avg reward " + str(score))

    plt.plot(range(rewardsAvg.shape[0]), rewardsAvg)
    plt.gca().set(ylim=(-20, 20))
    plt.savefig("trainingCart.png")


if __name__ == '__main__':
    mdpTest()
