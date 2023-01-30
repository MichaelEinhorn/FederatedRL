import gym
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import RLAlgs
import RLModels
import core
import copy
import os
import time

import CustomEnvs

convN = 100
alpha = 0.1
discount = 0.6
epsilon = 1
fedP = 10
syncBackups = 100 * convN
stochasticPolicy = True
envSeed = 3701
trial = 0

startTime = 0
totalTime = 3 * 60 * 60
totalTime -= 600

states = 4
actions = 3

jsonFilePrefix = "mdpv4_"
# dirPath = "~/scratch/RL"
dirPath = "/storage/home/hcoda1/2/meinhorn6/scratch/RL"
jsonDict = {}

windows = False
if os.name == 'nt':
    windows = True
    dirPath = "C:/Users/Michael Einhorn/Documents/GTML/RL"

# if os.path.isfile(jsonFileName):
#     jsonFile = open(jsonFileName)
#     jsonDict = json.load(jsonFile)
#     jsonFile.close()


def benchBell():
    envBell = CustomEnvs.RandomMDP(states, actions, seed=envSeed)

    print(envBell.transMat)
    print(envBell.rewardMat)

    modelBell = RLModels.QTabular(envBell, stochasticPolicy=True)
    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, gamma=discount, iterN=10000)

    scoreBell, sList = RLAlgs.score(env=envBell, model=modelBell, epLimit=10000, iterN=1, printErr=False)

    modelRand = RLModels.RandomPolicy(envBell)
    scoreRand, sListRand = RLAlgs.score(env=envBell, model=modelRand, epLimit=10000, iterN=1, printErr=False)
    print("bellman vs random")
    print(scoreBell)
    print(scoreRand)
    print("ratio")
    print(scoreBell / scoreRand)


# smaller random MDP to be solved exactly with bellman
def mdpTest():

    jsonDictKey = str((convN, alpha, discount, epsilon, 1, -1, stochasticPolicy, envSeed, trial))
    jsonFileSplitName = dirPath + "/" + jsonFilePrefix + ".json"

    # if jsonDictKey in jsonDict:
    if os.path.isfile(jsonFileSplitName):
        print("skipping " + jsonDictKey)
        return
    print("running " + jsonDictKey)

    env = CustomEnvs.RandomMDP(states, actions, seed=envSeed)
    envBell = CustomEnvs.RandomMDP(states, actions, seed=envSeed)
    print('NS:' + str(env.observation_space.n) + ' NA:' + str(env.action_space.n))
    model = RLModels.QTabular(env, stochasticPolicy=stochasticPolicy)
    modelBell = RLModels.QTabular(envBell, stochasticPolicy=stochasticPolicy)

    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, gamma=discount, iterN=10000)
    # print("bellman")
    # print(modelBell.q_tab)

    out_dict = RLAlgs.QLearning(env=env, model=model, epsilon=epsilon, epLimit=100,
                                                                          alpha=alpha, halfAlpha=False, gamma=discount, convN=convN, convThresh=0.01,
                                                                          convMethod="compare", otherModel=modelBell,
                                                                          logging=False, noReset=True, timeOut=(startTime, totalTime))
    model = out_dict["model"]
    del out_dict["model"]

    # print("learning")
    # print(model.q_tab)
    # print("bellman")
    # print(modelBell.q_tab)
    print("diff " + str(model.diff(modelBell)))

    print("simulation steps " + str(out_dict["sims"]))
    print("episodes " + str(out_dict["episodes"]))
    print("avg train reward " + str(out_dict["avgRew"]))

    score, sList = RLAlgs.score(model=model, env=env, epLimit=10000, iterN=1, printErr=False, stochastic=False)
    scoreStoch, sListStoch = RLAlgs.score(model=model, env=env, epLimit=10000, iterN=1, printErr=False, stochastic=True)

    out_dict["endScore"] = score
    out_dict["endScoreStoch"] = scoreStoch

    print("score " + str((score, scoreStoch)))

    scoreBell, sListBell = RLAlgs.score(model=modelBell, env=env, epLimit=10000, iterN=1, printErr=False,
                                        stochastic=False)
    scoreStochBell, sListStochBell = RLAlgs.score(model=modelBell, env=env, epLimit=10000, iterN=1, printErr=False,
                                                  stochastic=True)

    out_dict["endScoreBell"] = scoreBell
    out_dict["endScoreStochBell"] = scoreStochBell

    scoreRand, _ = RLAlgs.score(model=RLModels.RandomPolicy(env), env=env, epLimit=10000, iterN=1, printErr=False,
                             stochastic=False)

    out_dict["endScoreRand"] = scoreRand

    out_dict = core.dict_to_python(out_dict)
    jsonDict[jsonDictKey] = out_dict
    with open(jsonFileSplitName, 'w') as f:
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


def mdpTestFed():

    jsonDictKey = str((convN, alpha, discount, epsilon, fedP, syncBackups, stochasticPolicy, envSeed, trial))
    jsonFileSplitName = dirPath + "/" + jsonFilePrefix + ".json"
    
    # if jsonDictKey in jsonDict:
    if os.path.isfile(jsonFileSplitName):
        print("skipping " + jsonDictKey)
        return
    print("running " + jsonDictKey)

    env = CustomEnvs.RandomMDP(states, actions, seed=envSeed)
    envBell = CustomEnvs.RandomMDP(states, actions, seed=envSeed)
    print('NS:' + str(env.observation_space.n) + ' NA:' + str(env.action_space.n))
    model = RLModels.QTabular(env, stochasticPolicy=stochasticPolicy)
    modelBell = RLModels.QTabular(envBell, stochasticPolicy=stochasticPolicy)

    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, gamma=discount, iterN=10000)
    # print("bellman")
    # print(modelBell.q_tab)

    kwArgs = core.getArgs(epsilon=epsilon, epLimit=100,
                           alpha=alpha, halfAlpha=False, gamma=discount, convN=convN, convThresh=0.01, convMethod="compare", otherModel=modelBell, logging=False,
                           syncB=syncBackups, syncE=-1, noReset=True, timeOut=(startTime, totalTime))
    print(kwArgs)

    federatedModel = RLModels.QTabularFedAvg(model.getWeight().shape, stochasticPolicy=True, p=fedP)

    out, aggs = federatedModel.start(RLAlgs.QLearning, QLearningKWArgs=kwArgs, env=env, model=model)
    model = out[0]["model"]
    out_dict = out[0]
    del out_dict["model"]
    out_dict["aggs"] = aggs

    # modelF, simsF, backupsF, epsToBackupF, episodesF, avgRF, rListF, diffsF = tuple(core.rowsToColumnsPython(out))

    # print("learning")
    # print(model.q_tab)
    # print("bellman")
    # print(modelBell.q_tab)
    print("diff " + str(model.diff(modelBell)))

    print("simulation steps " + str(out_dict["sims"]))
    print("episodes " + str(out_dict["episodes"]))
    print("avg train reward " + str(out_dict["avgRew"]))

    score, sList = RLAlgs.score(model=model, env=env, epLimit=10000, iterN=1, printErr=False, stochastic=False)
    scoreStoch, sListStoch = RLAlgs.score(model=model, env=env, epLimit=10000, iterN=1, printErr=False,
                                          stochastic=True)

    print("score " + str((score, scoreStoch)))
    out_dict["endScore"] = score
    out_dict["endScoreStoch"] = scoreStoch

    scoreBell, sListBell = RLAlgs.score(model=modelBell, env=env, epLimit=10000, iterN=1, printErr=False, stochastic=False)
    scoreStochBell, sListStochBell = RLAlgs.score(model=modelBell, env=env, epLimit=10000, iterN=1, printErr=False,
                                          stochastic=True)

    out_dict["endScoreBell"] = scoreBell
    out_dict["endScoreStochBell"] = scoreStochBell

    scoreRand, _ = RLAlgs.score(model=RLModels.RandomPolicy(env), env=env, epLimit=10000, iterN=1, printErr=False, stochastic=False)

    out_dict["endScoreRand"] = scoreRand

    out_dict = core.dict_to_python(out_dict)
    print(out_dict)
    jsonDict[jsonDictKey] = out_dict
    with open(jsonFileSplitName, 'w') as f:
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

    modelBell = RLAlgs.QLearningTabularBellman(model=modelBell, env=envBell, gamma=discount, iterN=1000)
    print("bellman")
    print(modelBell.q_tab)

    envTD = CustomEnvs.RandomMDP(states, actions, seed=envSeed)
    q_tabTD, sims, backups = RLAlgs.TDLearnNStep(env=envTD, n=10, alpha=0.1, gamma=discount, lam=0.9, epsilon=0, iterN=10000,
                                                 epLimit=100, trace="accumulate",
                                                 stoch=True, logging=True, otherModel=modelBell)
    print(q_tabTD)


# small discrete space
def taxiTest():
    env = gym.make("Taxi-v3").env
    print('NS:' + str(env.observation_space.n) + ' NA:' + str(env.action_space.n))
    model = RLModels.QTabular(env, stochasticPolicy=True)
    model, sims, backups, episodes, avgR, rList, rewardsAvg = RLAlgs.QLearning(env=env, model=model, epsilon=0.1,
                                                                               alpha=0.1, gamma=discount, convN=100, convThresh=0.01)
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
    model, sims, backups, episodes, avgR, rList, rewardsAvg = RLAlgs.QLearning(env=env, model=model, iterN=1000, epsilon=0,
                                                                               alpha=0.01, gamma=discount, convN=-1, convThresh=0.01)

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

    # jsonDictKey = str((convN, alpha, discount, epsilon, fedP, syncBackups, stochasticPolicy, envSeed, trial))
    # # benchBell
    # for trial in range(3):
    #     for epsilon in [1]:
    #         for syncBackups in [10000, 1000, 100, 10, 1]:
    #             for fedP in [2, 4, 6, 8, 10]:
    #                 for alpha in [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01]:
    #                     # plt.clf()
    #                     mdpTest(fileExt="alpha_" + str(alpha) + "_sync_" + str(syncBackups) + "_eps_" + str(epsilon))
    #                     mdpTestFed(fileExt="alpha_" + str(alpha) + "_sync_" + str(syncBackups) + "_eps_" + str(epsilon))
    import argparse
    # add args for trial, epsilon, syncBackups, fedP, alpha, convN, discount, stochasticPolicy
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--syncBackups', type=int, default=10000)
    parser.add_argument('--fedP', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--convN', type=int, default=10)
    parser.add_argument('--discount', type=float, default=0.6)
    parser.add_argument('--stochasticPolicy', type=bool, default=True)
    
    args = parser.parse_args()
    jsonFilePrefix = args.prefix
    trial = args.trial
    epsilon = args.epsilon
    syncBackups = args.syncBackups
    fedP = args.fedP
    alpha = args.alpha
    convN = args.convN
    discount = args.discount
    stochasticPolicy = args.stochasticPolicy

    startTime = time.time()

    # envSeed = envSeed + trial

    if fedP == 1:
        mdpTest()
    else:
        mdpTestFed()


