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

convN = 100
alpha = 0.1
discount = 0.
epsilon = 1
fedP = 10
syncBackups = 100 * convN

jsonFileName = "mdp.json"


def benchBell():
    envBell = CustomEnvs.RandomMDP(6, 5, seed=42)
    modelBell = RLModels.QTabular(envBell, stochasticPolicy=False)
    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, y=discount, iterN=10000)

    scoreBell, sList = RLAlgs.score(env=envBell, model=modelBell, epLimit=10000, iterN=1, printErr=False)

    modelRand = RLModels.RandomPolicy(envBell)
    scoreRand, sListRand = RLAlgs.score(env=envBell, model=modelRand, epLimit=10000, iterN=1, printErr=False)
    print("bellman vs random")
    print(scoreBell)
    print(scoreRand)
    print(scoreBell / scoreRand)


# smaller random MDP to be solved exactly with bellman
def mdpTest(fileExt=""):
    env = CustomEnvs.RandomMDP(6, 5, seed=42)
    envBell = CustomEnvs.RandomMDP(6, 5, seed=42)
    print('NS:' + str(env.observation_space.n) + ' NA:' + str(env.action_space.n))
    model = RLModels.QTabular(env, stochasticPolicy=True)
    modelBell = RLModels.QTabular(envBell, stochasticPolicy=True)

    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, y=discount, iterN=10000)
    print("bellman")
    print(modelBell.q_tab)

    model, sims, backups, episodes, avgR, rList, diffs = RLAlgs.QLearning(env=env, model=model, eps=epsilon, epLimit=100,
                                                                          a=alpha, halfAlpha=False, y=discount, convN=convN, convThresh=0.01,
                                                                          convMethod="compare", otherModel=modelBell,
                                                                          logging=True, noReset=True)

    # print("learning")
    # print(model.q_tab)
    # print("bellman")
    # print(modelBell.q_tab)
    print("diff " + str(model.diff(modelBell)))

    print("simulation steps " + str(sims))
    print("episodes " + str(episodes))
    print("avg train reward " + str(avgR))

    score, sList = RLAlgs.score(model=model, env=env, epLimit=10000, iterN=1, printErr=False)
    print("avg reward " + str(score))

    plt.plot(np.array(range(diffs.shape[0])) * convN, diffs, label="Single Agent")
    #plt.gca().set(ylim=(0, 1))
    plt.yscale('log')

    plt.xlabel("episodes")
    plt.ylabel("distance to bellman q values")

    plt.title(fileExt.replace("_", " "))

    plt.savefig("trainingMDP" + fileExt + ".png")


def mdpTestFed(fileExt=""):
    env = CustomEnvs.RandomMDP(6, 5, seed=42)
    envBell = CustomEnvs.RandomMDP(6, 5, seed=42)
    print('NS:' + str(env.observation_space.n) + ' NA:' + str(env.action_space.n))
    model = RLModels.QTabular(env, stochasticPolicy=True)
    modelBell = RLModels.QTabular(envBell, stochasticPolicy=True)

    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, y=discount, iterN=10000)
    print("bellman")
    print(modelBell.q_tab)

    kwArgs = utils.getArgs(eps=epsilon, epLimit=100,
                           a=alpha, halfAlpha=False, y=discount, convN=convN, convThresh=0.01, convMethod="compare", otherModel=modelBell, logging=True,
                           syncB=syncBackups, syncE=-1, noReset=True)
    print(kwArgs)

    federatedModel = RLModels.QTabularFedAvg(model.getW().shape, stochasticPolicy=True, p=fedP)

    out, aggs = federatedModel.start(RLAlgs.QLearning, QLearningKWArgs=kwArgs, env=env, model=model)
    model, sims, backups, episodes, avgR, rList, diffs = out[0]

    # print("learning")
    # print(model.q_tab)
    # print("bellman")
    # print(modelBell.q_tab)
    print("diff " + str(model.diff(modelBell)))

    print("simulation steps " + str(sims))
    print("episodes " + str(episodes))
    print("avg train reward " + str(avgR))

    score, sList = RLAlgs.score(model=model, env=env, epLimit=10000, iterN=1, printErr=False)
    print("avg reward " + str(score))

    plt.plot(np.array(range(diffs.shape[0])) * convN, diffs, label="Federated 10 Agents")
    plt.legend()
    # plt.gca().set(ylim=(0, 1))

    plt.xlabel("episodes for single agent")
    plt.ylabel("distance to bellman q values")

    plt.title(fileExt.replace("_", " "))

    plt.yscale('log')
    plt.savefig("trainingMDPFed" + fileExt + ".png")


def mdpTestTD():
    envBell = CustomEnvs.RandomMDP(6, 5, seed=42)
    modelBell = RLModels.QTabular(envBell, stochasticPolicy=True)

    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, y=discount, iterN=1000)
    print("bellman")
    print(modelBell.q_tab)

    envTD = CustomEnvs.RandomMDP(6, 5, seed=42)
    q_tabTD, sims, backups = RLAlgs.TDLearnNStep(env=envTD, n=10, a=0.1, y=discount, lam=0.9, eps=0, iterN=10000,
                                                 epLimit=100, trace="accumulate",
                                                 stoch=True, logging=True, otherModel=modelBell)
    print(q_tabTD)


# small discrete space
def taxiTest():
    env = gym.make("Taxi-v3").env
    print('NS:' + str(env.observation_space.n) + ' NA:' + str(env.action_space.n))
    model = RLModels.QTabular(env, stochasticPolicy=True)
    model, sims, backups, episodes, avgR, rList, rewardsAvg = RLAlgs.QLearning(env=env, model=model, eps=0.1,
                                                                               a=0.1, y=discount, convN=100, convThresh=0.01)
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
                                                                               a=0.01, y=discount, convN=-1, convThresh=0.01)

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
    convN = 100
    alpha = 0.1
    np.set_printoptions(suppress=True)
    benchBell()
    # for a in range(1):
    #     plt.clf()
    #     mdpTest(fileExt="alpha_" + str(alpha))
    #     mdpTestFed(fileExt="alpha_" + str(alpha))
    #     alpha /= 2
