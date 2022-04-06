import numpy as np
import gym
import copy
from numpy.random import default_rng

from utils import softmax

import collections
import multiprocessing as mp


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


def TDLearnNStep(env=None, n=1, a=0.1, y=0.6, lam=0.9, eps=0.1, iterN=100000, epLimit=-1, trace="replace", seed=None, stoch=False, otherModel=None, convThresh=0.01, logging=False):
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
            nextS, r, term, info = env.step(act)

            if rng.random() < eps:
                actP = env.action_space.sample()
            else:
                if stoch:
                    qV = softmax(q_tab[nextS])
                    actP = rng.choice(range(NA), 1, p=qV)[0]
                else:
                    actP = np.argmax(q_tab[nextS])

            g = r + y * q_tab[nextS, actP] - q_tab[state, act]
            reward += r
            if trace == "accumulate":
                e_trace[state, act] += 1
            elif trace == "dutch":
                e_trace[state, act] = (1 - a) * e_trace[state, act] + 1
            elif trace == "replace":
                e_trace[state, act] = 1

            for sT, aT in tracePairs:
                backups += 1
                q_tab[sT, aT] += a * g * e_trace[sT, aT]
                e_trace[sT, aT] *= y * lam

            state = nextS
            act = actP

        if e % 100 == 0:
            #print(reward / 100)
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
                        print("half alpha to " + str(a))
                    a = a / 2
                    lastHyperChange = e

    return q_tab, sims, backups


def QLearningTabularBellman(model=None, env=None, y=0.6, iterN=1000):
    for i in range(iterN):
        model.bellman(env=env, y=y)
    return model


def QLearning(initState=None, iterN=100000, env=None, model=None,
              eps=0.1, a=0.1, halfAlpha=True, y=0.6, epLimit=-1, convN=5, convThresh=0.01, logging=True, convMethod="trainR", otherModel=None,
              mpQueue=None, returnQ=None, syncB=-1, syncE=-1, aggWmp=None, mpEnd=None, seed=None, noReset=False):

    if seed is None:
        rng = default_rng()
    else:
        rng = default_rng(seed)

    sims = 0
    backups = 0
    rewards = np.zeros(iterN)
    rewardsAvg = [0]
    avg = -1000
    diffs = []
    lastHyperChange = -1

    if aggWmp is not None:
        aggW = np.frombuffer(aggWmp.get_obj()).reshape(model.getW().shape)

    # only get state once
    if noReset:
        state = env.reset()

    for e in range(iterN):
        if not noReset:
            if initState is None:
                state = env.reset()
            else:
                state = initState
                # need to set state of env, not always supported

        term = False
        i = 0
        reward = 0
        while not term and (epLimit == -1 or i < epLimit):
            if rng.random() < eps:
                act = env.action_space.sample()
            else:
                act = model.policy(state)

            sims += 1
            nextS, r, term, info = env.step(act)
            reward += r
            backups += 1
            model.backup(state, act, nextS, r, a, y)
            state = nextS

            if syncB != -1 and backups % syncB == 0:
                # print(backups)
                if mpEnd.value:
                    break
                mpQueue.put(model.getW())
                mpQueue.join()
                model.setW(aggW)

            i += 1

        rewards[e] = reward

        # sync
        if syncB != -1 and backups % syncB == 0:
            if mpEnd.value:
                break
        if syncE != -1 and (e+1) % syncE == 0:
            if mpEnd.value:
                break
            mpQueue.put(model.getW())
            mpQueue.join()
            model.setW(aggW)

        # convergence testing
        if convN != -1 and (e+1) % convN == 0:
            if convMethod == "score":
                # tests without exploration or updating during the episode
                prev = avg
                avg, _ = score(iterN=20, model=model, env=env, epLimit=1000)
                if logging:
                    print("episode " + str(e) + " r " + str(avg))
                rewardsAvg.append(avg)
                diff = abs(avg - prev)
                if diff < convThresh and avg > 0:
                    if returnQ is not None:
                        returnQ.put((model, sims, backups, (e+1), avg, rewards[:e + 1], np.array(rewardsAvg)))
                    return model, sims, backups, (e+1), avg, rewards[:e + 1], np.array(rewardsAvg)

            elif convMethod == "trainR":
                if e >= 2 * convN:
                    # converges if average reward of convN episodes is the same as the previous convN episodes
                    avg = np.mean(rewards[e-convN:e+1])
                    if logging:
                        print("episode " + str(e) + " r " + str(avg))
                    prev = np.mean(rewards[e-2*convN:e-convN])
                    rewardsAvg.append(avg)
                    diff = abs(avg - prev)
                    if diff < convThresh and avg > 0:
                        # print("reward difference " + str(diff))
                        # if eps != 0:
                        #     print("training no eps")
                        #     eps = 0
                        # else:
                        if returnQ is not None:
                            returnQ.put((model, sims, backups, (e+1), avg, rewards[:e+1], np.array(rewardsAvg)))
                        return model, sims, backups, (e+1), avg, rewards[:e+1], np.array(rewardsAvg)

            elif convMethod == "compare":
                diff = model.diff(otherModel)
                diffs.append(diff)

                if logging:
                    print("diff " + str(diff))
                    # print(backups)
                if diff < convThresh:
                    if returnQ is not None:
                        returnQ.put((model, sims, backups, (e+1), avg, rewards[:e + 1], np.array(diffs)))
                    return model, sims, backups, (e+1), avg, rewards[:e + 1], np.array(diffs)

                if len(diffs) > 21 and (e+1) % (convN * 10) == 0 and \
                        (e - lastHyperChange > 3*(convN * 10) or lastHyperChange == -1):

                    avgR = np.mean(rewards[e - convN:e + 1])
                    diffsNp = np.array(diffs)
                    avgDiff = np.mean(diffsNp[-10:])
                    prev = np.mean(diffsNp[-20:-10])
                    # increased error in past 20 measurements
                    if prev < avgDiff:
                        # print(avgDiff)
                        # print(prev)
                        if halfAlpha:
                            a = a / 2
                            if logging:
                                print("half alpha to " + str(a))
                            lastHyperChange = e
                        else:
                            if returnQ is not None:
                                returnQ.put((model, sims, backups, (e+1), avgR, rewards[:e + 1], np.array(diffs)))
                            return model, sims, backups, (e+1), avgR, rewards[:e + 1], np.array(diffs)

    if convN != -1:
        avg = np.mean(rewards[e - convN + 1:])
    rewardsAvg.append(avg)

    if convMethod == "compare":
        if returnQ is not None:
            returnQ.put((model, sims, backups, (e+1), avg, rewards[:e + 1], np.array(diffs)))
        return model, sims, backups, (e+1), avg, rewards[:e + 1], np.array(diffs)
    else:
        if returnQ is not None:
            returnQ.put((model, sims, backups, (e+1), avg, rewards[:e + 1], np.array(rewardsAvg)))
        return model, sims, backups, (e+1), avg, rewards, np.array(rewardsAvg)