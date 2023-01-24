import numpy as np
import gym
import copy
from numpy.random import default_rng
import time

from core import softmax, getKW
import core

import collections
import multiprocessing as mp

# evaluate a model on an environment return mean reward and list of rewards


def score(initState=None, iterN=100, env=None, model=None, epLimit=-1, printErr=False, stochastic=None):
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
            act = model.policy(state, stochastic=stochastic)
            state, r, term, info = env.step(act)
            reward += r
        if epLimit != -1 and i >= epLimit and printErr:
            print("testing episode " + str(e) +
                  " timed out with r " + str(reward))
        rewards[e] = reward
    return np.mean(rewards), rewards


def TDLearnNStep(env=None, n=1, alpha=0.1, gamma=0.6, lam=0.9, epsilon=0.1, iterN=100000, epLimit=-1, trace="replace", seed=None, stoch=False, otherModel=None, convThresh=0.01, logging=False):
    if seed is None:
        rng = default_rng()
    else:
        rng = default_rng(seed)

    sims = 0
    backups = 0
    NS = env.observation_space.n
    NA = env.action_space.n
    q_tab = np.zeros((NS, NA))
    reward = 0
    diffs = []
    lastHyperChange = -1

    for e in range(iterN):
        e_trace = np.zeros((NS, NA))
        state = 0
        act = env.action_space.sample()
        term = False
        i = -1
        tracePairs = collections.deque(maxlen=n)

        while not term and (epLimit == -1 or i < epLimit):
            i += 1
            if len(tracePairs) == n:
                tempPair = tracePairs[0]
                tracePairs.append((state, act))
                e_trace[tempPair[0], tempPair[1]] = 0
            else:
                tracePairs.append((state, act))

            sims += 1
            nextS, reward, term, info = env.step(act)

            if rng.random() < epsilon:
                actP = env.action_space.sample()
            else:
                if stoch:
                    qV = softmax(q_tab[nextS])
                    actP = rng.choice(range(NA), 1, p=qV)[0]
                else:
                    actP = np.argmax(q_tab[nextS])

            g = reward + gamma * q_tab[nextS, actP] - q_tab[state, act]
            reward += reward
            if trace == "accumulate":
                e_trace[state, act] += 1
            elif trace == "dutch":
                e_trace[state, act] = (1 - alpha) * e_trace[state, act] + 1
            elif trace == "replace":
                e_trace[state, act] = 1

            for sT, aT in tracePairs:
                backups += 1
                q_tab[sT, aT] += alpha * g * e_trace[sT, aT]
                e_trace[sT, aT] *= gamma * lam

            state = nextS
            act = actP

        if e % 100 == 0:
            # print(reward / 100)
            reward = 0

            diff = np.linalg.norm(q_tab - otherModel.q_tab)
            if logging:
                print(diff)

            diffs.append(diff)

            if diff < convThresh:
                return q_tab, sims, backups

            if len(diffs) > 21 and e % (100 * 10) == 0 and \
                    (e - lastHyperChange > 3 * (100 * 10) or lastHyperChange == -1):
                diffsNp = np.array(diffs)
                avgDiff = np.mean(diffsNp[len(diffs) - 10:])
                prev = np.mean(diffsNp[len(diffs) - 2 * 10:len(diffs) - 10])
                if prev > avgDiff:
                    if logging:
                        print("half alpha to " + str(alpha))
                    alpha = alpha / 2
                    lastHyperChange = e

    return q_tab, sims, backups


# get the bellman optimal solution. Only works for matrix MDPs
def QLearningTabularBellman(model=None, env=None, gamma=0.6, iterN=1000):
    for i in range(iterN):
        model.bellman(env=env, gamma=gamma)
    return model


# runs QLearning Algorithm
# works for both federated and single agent
def QLearning(initState=None,
              iterN=100000,  # number of episodes
              env=None,  # open AI gym obj
              model=None,  # RL Model Object
              epsilon=0.1,  # epsilon greedy exploration
              alpha=0.1,  # learning rate
              halfAlpha=True,  # if true, alpha is halved when convergence is detected
              gamma=0.6,  # discount rate
              epLimit=-1,  # end episode after this many steps
              convN=5,  # test convergence every N episodes
              convThresh=0.01,  # threshold for convergence
              logging=True,
              # score: run the model on test env and determine convergence when test reward is flat
              # trainR: check if training reward is flat
              # compare: check distance to a reference Q table such as bellman optimal
              convMethod="trainR",
              otherModel=None,
              # mp Queues
              AggWeightQ=None,
              returnQ=None,
              # only use one or the other
              syncB=-1,  # sync every syncB backups
              syncE=-1,  # sync every syncE episodes
              WeightSharedMem=None,  # shared memory for synched weights
              mpEnd=None, # flag to exit training for all processes after any process returns. Checked before a sync weights occurs
              seed=None,
              noReset=False,  # reset the environment at the start of each episode
              timeOut=None # startTime, totalTime
              ):

    if seed is None:
        rng = default_rng()
    else:
        rng = default_rng(seed)

    sims = 0
    backups = 0
    rewards = np.zeros(iterN)
    avgRewArr = []
    avgTestScoreArr = []
    avgRew = -1000
    diffs = []
    lastHyperChange = -1
    epsToBackup = []

    if WeightSharedMem is not None:
        aggW = np.frombuffer(WeightSharedMem.get_obj()
                             ).reshape(model.getWeight().shape)

    # only get state once
    if noReset:
        state = env.reset()

    # not finished training if returnComment is None
    returnComment = None
    e = -1
    # for e in range(iterN):
    while True:
        e += 1

        if not noReset:
            if initState is None:
                state = env.reset()
            else:
                state = initState
                # need to set state of env, not always supported

        epsToBackup.append(backups)

        term = False
        i = 0
        reward = 0
        while not term and (epLimit == -1 or i < epLimit):
            if rng.random() < epsilon:
                act = env.action_space.sample()
            else:
                act = model.policy(state)

            sims += 1
            nextS, r, term, info = env.step(act)
            reward += r
            backups += 1
            model.backup(state, act, nextS, r, alpha, gamma)
            state = nextS

            if syncB != -1 and backups % syncB == 0:
                # print(backups)
                if mpEnd.value:
                    returnComment = "mpEnd"
                    break
                AggWeightQ.put(model.getWeight())
                AggWeightQ.join()
                model.setWeight(aggW)

            i += 1

        rewards[e] = reward

        # sync
        if syncB != -1 and backups % syncB == 0:
            if mpEnd.value:
                returnComment = "mpEnd"
        if syncE != -1 and (e+1) % syncE == 0:
            if mpEnd.value:
                returnComment = "mpEnd"
            AggWeightQ.put(model.getWeight())
            AggWeightQ.join()
            model.setWeight(aggW)

        if timeOut is not None:
            if time.time() - timeOut[0] > timeOut[1]:
                returnComment = "timeout"

        if e == iterN - 1:
            returnComment = "max episodes"

        # convergence testing, always terminate if there is a returnComment
        if (convN != -1 and (e+1) % convN == 0) or returnComment is not None:
            if convMethod == "score":
                avgRew = np.mean(rewards[e+1 - convN:e + 1])
                avgRewArr.append(avgRew)
                # tests without exploration or updating during the episode
                prevTestScore = avgTestScore
                avgTestScore, _ = score(iterN=20, model=model, env=env, epLimit=1000)
                if logging:
                    print("episode " + str(e) + " r " + str(avgTestScore))
                avgTestScoreArr.append(avgTestScore)
                diff = abs(avgTestScore - prevTestScore)
                diffs.append(diff)

                if diff < convThresh and avgRew > 0:
                    returnComment = "constant test score"

                if returnComment is not None:
                    returnDict = getKW(model=model, sims=sims, backups=backups, epsToBackup=epsToBackup, episodes=(e+1), avgTestScore=avgTestScore, avgTestScoreArr=np.array(avgTestScoreArr),
                                       avgRew=avgRew, rewards=rewards[:e + 1], avgRewArr=np.array(avgRewArr), diffs=diffs, comment=returnComment)
                    if returnQ is not None:
                        returnQ.put(returnDict)
                    return returnDict

            elif convMethod == "trainR":
                if e >= 2 * convN:
                    # converges if average reward of convN episodes is the same as the previous convN episodes
                    avgRew = np.mean(rewards[e+1 - convN:e + 1])
                    if logging:
                        print("episode " + str(e) + " r " + str(avgRew))
                    prevAvgRew = np.mean(rewards[e-2*convN:e-convN])
                    avgRewArr.append(avgRew)
                    diff = abs(avgRew - prevAvgRew)
                    diffs.append(diff)

                    if diff < convThresh and avgRew > 0:
                        returnComment = "constant train reward"

                    if returnComment is not None:
                        # print("reward difference " + str(diff))
                        # if eps != 0:
                        #     print("training no eps")
                        #     eps = 0
                        # else:
                        returnDict = getKW(model=model, sims=sims, backups=backups, epsToBackup=epsToBackup, episodes=(e+1),
                                       avgRew=avgRew, rewards=rewards[:e + 1], avgRewArr=np.array(avgRewArr), diffs=diffs, comment=returnComment)
                        if returnQ is not None:
                            returnQ.put(returnDict)
                        return returnDict

            elif convMethod == "compare":
                diff = model.diff(otherModel)
                diffs.append(diff)
                avgRew = np.mean(rewards[e+1 - convN:e + 1])
                avgRewArr.append(avgRew)

                if logging:
                    print("diff " + str(diff))
                    # print(backups)

                if diff < convThresh:
                    returnComment = "converged distance"
                
                if returnComment is not None:
                    returnDict = getKW(model=model, sims=sims, backups=backups, epsToBackup=epsToBackup, episodes=(e+1),
                                       avgRew=avgRew, rewards=rewards[:e + 1], avgRewArr=np.array(avgRewArr), diffs=diffs, comment=returnComment)
                    if returnQ is not None:
                        returnQ.put(returnDict)
                    return returnDict

                if len(diffs) > 21 and (e+1) % (convN * 10) == 0 and \
                        (e - lastHyperChange > 3*(convN * 10) or lastHyperChange == -1):

                    diffsNp = np.array(diffs)
                    avgDiff = np.mean(diffsNp[-10:])
                    prevAvgRew = np.mean(diffsNp[-20:-10])
                    # increased error in past 20 measurements
                    if prevAvgRew < avgDiff:
                        # print(avgDiff)
                        # print(prev)
                        if halfAlpha:
                            alpha = alpha / 2
                            if logging:
                                print("half alpha to " + str(alpha))
                            lastHyperChange = e
                        else:
                            returnComment = "constant distance"
                            returnDict = getKW(model=model, sims=sims, backups=backups, epsToBackup=epsToBackup, episodes=(e+1),
                                               score=avgRew, rewards=rewards[:e + 1], scoresArr=np.array(avgRewArr), diffs=diffs, comment=returnComment)
                            if returnQ is not None:
                                returnQ.put(returnDict)
                            return returnDict
