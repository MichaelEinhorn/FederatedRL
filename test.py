import gym
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import RLAlgs


# small discrete space
def taxiTest():
    env = gym.make("Taxi-v3").env
    print('NS:' + str(env.observation_space.n) + ' NA:' + str(env.action_space.n))
    model = RLAlgs.QTabular(env)
    model, sims, backups, episodes, avgR, rList, rewardsAvg = RLAlgs.QLearning(env=env, model=model, eps=0.1,
                                                            a=0.1, y=0.6, convN=100, convThresh=0.01)
    print("simulation steps " + str(sims))
    print("episodes " + str(episodes))
    print("avg train reward " + str(avgR))

    score, sList = RLAlgs.score(model=model, env=env, epLimit=1000, printErr=True)
    print("avg reward " + str(score))

    plt.plot(range(rewardsAvg.shape[0]), rewardsAvg)
    plt.gca().set(ylim=(-20, 20))
    plt.savefig("training.png")


taxiTest()
