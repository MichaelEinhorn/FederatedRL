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
    plt.savefig("trainingTaxi.png")

def cartTest():
    env = gym.make("CartPole-v1").env
    model = RLAlgs.QLinAprox(env, RLAlgs.cartPoleFeature)
    model, sims, backups, episodes, avgR, rList, rewardsAvg = RLAlgs.QLearning(env=env, model=model, iterN=1000, eps=0.1,
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

taxiTest()
