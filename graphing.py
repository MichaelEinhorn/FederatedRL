import json
import numpy as np
import matplotlib.pyplot as plt
import os

jsonFileName = "mdpv2.json"
jsonDict = {}

convN = 10
discount = 0.6
epsilon = 1
syncBackups = 1
# syncBackups = 1
stochasticPolicy = True
envSeed = 42

alpha = 1
fedP = 2

if os.path.isfile(jsonFileName):
    jsonFile = open(jsonFileName)
    jsonDict = json.load(jsonFile)
    jsonFile.close()

print(jsonDict.keys())

# for syncBackups in [10000, 1000, 100, 10, 1]:

fedPs = [2, 4, 6, 8, 10]

for epsilon in [1, 0]:
    alpha = 1

    epsSin = []
    epsFed = {}
    scoresSin = []
    scoresFed = {}
    scoresBell = []
    scoresStochSin = []
    scoresStochFed = {}
    scoresStochBell = []
    scoresRand = []
    diffsSin = []
    diffsFed = {}
    alphas = []

    for a in range(10):

        for fedP in fedPs:

            if not fedP in epsFed:
                epsFed[fedP] = []
                scoresFed[fedP] = []
                diffsFed[fedP] = []
                scoresStochFed[fedP] = []

            jsonDictKeyFed = str((convN, alpha, discount, epsilon, fedP, syncBackups, stochasticPolicy, envSeed))

            simsF, backupsF, episodesF, avgRF, rListF, diffsF, aggs, scoreF, scoreStochF, scoreBellF, scoreStochBellF, scoreRandF = jsonDict[jsonDictKeyFed]
            epsFed[fedP].append(episodesF[0])
            scoresFed[fedP].append(scoreF/100)
            diffsFed[fedP].append(np.mean(diffsF[0][-10:]))
            scoresStochFed[fedP].append(scoreStochF/100)

        jsonDictKeySin = str((convN, alpha, discount, epsilon, 1, -1, stochasticPolicy, envSeed))
        # print(jsonDictKeySin)
        sims, backups, episodes, avgR, rList, diffs, score, scoreStoch, scoreBell, scoreStochBell, scoreRand = jsonDict[jsonDictKeySin]

        epsSin.append(episodes)
        scoresSin.append(score/100)
        scoresBell.append(scoreBell/100)
        scoresRand.append(scoreRand/100)
        scoresStochSin.append(scoreStoch / 100)
        scoresStochBell.append(scoreStochBell / 100)
        diffsSin.append(np.mean(diffs[-10:]))

        alphas.append(alpha)

        alpha /= 2

    plt.clf()
    plotName = "Q learning Difference from Bellman with Eps " + str(epsilon)
    plt.plot(alphas, diffsSin, marker="x", label="Single Agent")
    for fedP in fedPs:
        plt.plot(alphas, diffsFed[fedP], marker="x", label=str(fedP) + " Agent")
    plt.legend()
    plt.xlabel("alpha - step size")
    plt.ylabel("final distance to bellman q values")

    plt.title(plotName.replace("_", " "))
    # plt.ylim(0, 4)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(plotName.replace(" ", "_") + ".png")

    plt.clf()
    plotName = "Q learning Score with Eps " + str(epsilon)
    plt.plot(alphas, scoresSin, marker="x", label="Single Agent")
    for fedP in fedPs:
        plt.plot(alphas, scoresFed[fedP], marker="x", label=str(fedP) + " Agent")
    plt.hlines(np.mean(scoresBell), 0, 1, label="bellman score")
    plt.hlines(np.mean(scoresRand), 0, 1, label="random policy score")
    plt.legend()
    plt.xlabel("alpha - step size")
    plt.ylabel("Test Reward")

    plt.title(plotName.replace("_", " "))
    # plt.ylim(0, 4)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.savefig(plotName.replace(" ", "_") + ".png")

    plt.clf()
    plotName = "Q learning Score Stochastic Policy with Eps " + str(epsilon)
    plt.plot(alphas, scoresStochSin, marker="x", label="Single Agent")
    for fedP in fedPs:
        plt.plot(alphas, scoresStochFed[fedP], marker="x", label=str(fedP) + " Agent")
    plt.hlines(np.mean(scoresStochBell), 0, 1, label="bellman score")
    plt.hlines(np.mean(scoresRand), 0, 1, label="random policy score")
    plt.legend()
    plt.xlabel("alpha - step size")
    plt.ylabel("Test Reward")

    plt.title(plotName.replace("_", " "))
    # plt.ylim(0, 4)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.savefig(plotName.replace(" ", "_") + ".png")

    plt.clf()
    plotName = "Q learning Convergence Time with Eps " + str(epsilon)
    plt.plot(alphas, np.array(epsSin) * 100, marker="x", label="Single Agent")
    for fedP in fedPs:
        plt.plot(alphas, np.array(epsFed[fedP]) * 100, marker="x", label=str(fedP) + " Agent")
    plt.legend()
    plt.xlabel("alpha - step size")
    plt.ylabel("Number of backups for a single agent")

    plt.title(plotName.replace("_", " "))
    # plt.ylim(0, 4)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.savefig(plotName.replace(" ", "_") + ".png")
