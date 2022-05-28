import gym
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import RLAlgs
import RLModels
import utils
import copy
import os

import CustomEnvs

convN = 100
alpha = 0.1
discount = 0.6
epsilon = 1
fedP = 10
syncBackups = 100 * convN
stochasticPolicy = True
envSeed = 0
trial = 0

states = 4
actions = 3

jsonFileName = "mdpv3.json"
jsonDict = {}

if os.path.isfile(jsonFileName):
    jsonFile = open(jsonFileName)
    jsonDict = json.load(jsonFile)
    jsonFile.close()


def benchBell():
    envBell = CustomEnvs.RandomMDP(states, actions, seed=envSeed)

    print(envBell.transMat)
    print(envBell.rewardMat)

    modelBell = RLModels.QTabular(envBell, stochasticPolicy=True)
    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, y=discount, iterN=10000)

    scoreBell, sList = RLAlgs.score(env=envBell, model=modelBell, epLimit=10000, iterN=1, printErr=False)

    modelRand = RLModels.RandomPolicy(envBell)
    scoreRand, sListRand = RLAlgs.score(env=envBell, model=modelRand, epLimit=10000, iterN=1, printErr=False)
    print("bellman vs random")
    print(scoreBell)
    print(scoreRand)
    print("ratio")
    print(scoreBell / scoreRand)


# smaller random MDP to be solved exactly with bellman
def mdpTest(fileExt=""):

    jsonDictKey = str((convN, alpha, discount, epsilon, 1, -1, stochasticPolicy, envSeed, trial))
    if jsonDictKey in jsonDict:
        print("skipping " + jsonDictKey)
        return
    print("running " + jsonDictKey)

    env = CustomEnvs.RandomMDP(states, actions, seed=envSeed)
    envBell = CustomEnvs.RandomMDP(states, actions, seed=envSeed)
    print('NS:' + str(env.observation_space.n) + ' NA:' + str(env.action_space.n))
    model = RLModels.QTabular(env, stochasticPolicy=stochasticPolicy)
    modelBell = RLModels.QTabular(envBell, stochasticPolicy=stochasticPolicy)

    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, y=discount, iterN=10000)
    # print("bellman")
    # print(modelBell.q_tab)

    model, sims, backups, epsToBackup, episodes, avgR, rList, diffs = RLAlgs.QLearning(env=env, model=model, eps=epsilon, epLimit=100,
                                                                          a=alpha, halfAlpha=False, y=discount, convN=convN, convThresh=0.01,
                                                                          convMethod="compare", otherModel=modelBell,
                                                                          logging=False, noReset=True)

    # print("learning")
    # print(model.q_tab)
    # print("bellman")
    # print(modelBell.q_tab)
    print("diff " + str(model.diff(modelBell)))

    print("simulation steps " + str(sims))
    print("episodes " + str(episodes))
    print("avg train reward " + str(avgR))

    score, sList = RLAlgs.score(model=model, env=env, epLimit=10000, iterN=1, printErr=False, stochOverride=False)
    scoreStoch, sListStoch = RLAlgs.score(model=model, env=env, epLimit=10000, iterN=1, printErr=False, stochOverride=True)

    print("score " + str((score, scoreStoch)))

    scoreBell, sListBell = RLAlgs.score(model=modelBell, env=env, epLimit=10000, iterN=1, printErr=False,
                                        stochOverride=False)
    scoreStochBell, sListStochBell = RLAlgs.score(model=modelBell, env=env, epLimit=10000, iterN=1, printErr=False,
                                                  stochOverride=True)

    scoreRand, _ = RLAlgs.score(model=RLModels.RandomPolicy(env), env=env, epLimit=10000, iterN=1, printErr=False,
                             stochOverride=False)

    jsonDict[jsonDictKey] = (sims, backups, epsToBackup, episodes, avgR, rList, diffs, score, scoreStoch, scoreBell, scoreStochBell, scoreRand)
    with open(jsonFileName, 'w') as f:
        json.dump(jsonDict, f)

    # print("avg reward " + str(score))
    # diffsNp = np.array(diffs)
    # plt.plot(np.array(range(diffsNp.shape[0])) * convN, diffsNp, label="Single Agent")
    # #plt.gca().set(ylim=(0, 1))
    # plt.yscale('log')
    #
    # plt.xlabel("episodes")
    # plt.ylabel("distance to bellman q values")
    #
    # plt.title(fileExt.replace("_", " "))
    #
    # plt.savefig("trainingMDP" + fileExt + ".png")


def mdpTestFed(fileExt=""):

    jsonDictKey = str((convN, alpha, discount, epsilon, fedP, syncBackups, stochasticPolicy, envSeed, trial))
    if jsonDictKey in jsonDict:
        print("skipping " + jsonDictKey)
        return
    print("running " + jsonDictKey)

    env = CustomEnvs.RandomMDP(states, actions, seed=envSeed)
    envBell = CustomEnvs.RandomMDP(states, actions, seed=envSeed)
    print('NS:' + str(env.observation_space.n) + ' NA:' + str(env.action_space.n))
    model = RLModels.QTabular(env, stochasticPolicy=stochasticPolicy)
    modelBell = RLModels.QTabular(envBell, stochasticPolicy=stochasticPolicy)

    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, y=discount, iterN=10000)
    # print("bellman")
    # print(modelBell.q_tab)

    kwArgs = utils.getArgs(eps=epsilon, epLimit=100,
                           a=alpha, halfAlpha=False, y=discount, convN=convN, convThresh=0.01, convMethod="compare", otherModel=modelBell, logging=False,
                           syncB=syncBackups, syncE=-1, noReset=True)
    print(kwArgs)

    federatedModel = RLModels.QTabularFedAvg(model.getW().shape, stochasticPolicy=True, p=fedP)

    out, aggs = federatedModel.start(RLAlgs.QLearning, QLearningKWArgs=kwArgs, env=env, model=model)
    model, sims, backups, epsToBackup, episodes, avgR, rList, diffs = out[0]

    modelF, simsF, backupsF, epsToBackupF, episodesF, avgRF, rListF, diffsF = tuple(utils.rowsToColumnsPython(out))

    # print("learning")
    # print(model.q_tab)
    # print("bellman")
    # print(modelBell.q_tab)
    print("diff " + str(model.diff(modelBell)))

    print("simulation steps " + str(sims))
    print("episodes " + str(episodes))
    print("avg train reward " + str(avgR))

    score, sList = RLAlgs.score(model=model, env=env, epLimit=10000, iterN=1, printErr=False, stochOverride=False)
    scoreStoch, sListStoch = RLAlgs.score(model=model, env=env, epLimit=10000, iterN=1, printErr=False,
                                          stochOverride=True)

    print("score " + str((score, scoreStoch)))

    scoreBell, sListBell = RLAlgs.score(model=modelBell, env=env, epLimit=10000, iterN=1, printErr=False, stochOverride=False)
    scoreStochBell, sListStochBell = RLAlgs.score(model=modelBell, env=env, epLimit=10000, iterN=1, printErr=False,
                                          stochOverride=True)

    scoreRand, _ = RLAlgs.score(model=RLModels.RandomPolicy(env), env=env, epLimit=10000, iterN=1, printErr=False, stochOverride=False)

    print((simsF, backupsF, episodesF, avgRF, rListF, diffsF, aggs, score, scoreStoch, scoreBell, scoreStochBell, scoreRand))
    jsonDict[jsonDictKey] = (simsF, backupsF, epsToBackupF, episodesF, avgRF, rListF, diffsF, aggs, score, scoreStoch, scoreBell, scoreStochBell, scoreRand)
    with open(jsonFileName, 'w') as f:
        json.dump(jsonDict, f)

    # print("avg reward " + str(score))
    #
    # diffsNp = np.array(diffs)
    # plt.plot(np.array(range(diffsNp.shape[0])) * convN, diffsNp, label="Federated " + str(fedP) + " Agents")
    # plt.legend()
    # # plt.gca().set(ylim=(0, 1))
    #
    # plt.xlabel("episodes for single agent")
    # plt.ylabel("distance to bellman q values")
    #
    # plt.title(fileExt.replace("_", " "))
    #
    # plt.yscale('log')
    # plt.savefig("trainingMDPFed" + fileExt + ".png")


def mdpTestTD():
    envBell = CustomEnvs.RandomMDP(states, actions, seed=envSeed)
    modelBell = RLModels.QTabular(envBell, stochasticPolicy=True)

    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, y=discount, iterN=1000)
    print("bellman")
    print(modelBell.q_tab)

    envTD = CustomEnvs.RandomMDP(states, actions, seed=envSeed)
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
    np.set_printoptions(suppress=True)

    convN = 1
    syncBackups = 1
    # benchBell
    for trial in range(3):
        for epsilon in [1]:
            for syncBackups in [10000, 1000, 100, 10, 1]:
                for fedP in [2, 4, 6, 8, 10]:
                    for alpha in [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01]:
                        # plt.clf()
                        mdpTest(fileExt="alpha_" + str(alpha) + "_sync_" + str(syncBackups) + "_eps_" + str(epsilon))
                        mdpTestFed(fileExt="alpha_" + str(alpha) + "_sync_" + str(syncBackups) + "_eps_" + str(epsilon))
