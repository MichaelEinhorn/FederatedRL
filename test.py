import gym
import numpy as np
import matplotlib
import json
import argparse
import RLAlgs


# small discrete space
def taxiTest():
    env = gym.make("Taxi-v3").env
    print('NS:' + str(env.observation_space.n) + ' NA:' + str(env.action_space.n))
    model = RLAlgs.QTabular(env)
    model, sims, backups, episodes, avgR, rList = RLAlgs.QLearning(env=env, model=model, eps=0.1,
                                                            a=0.1, y=0.6, convN=100, convThresh=0.01)
    print("simulation steps " + str(sims))
    print("episodes " + str(episodes))
    print("avg train reward " + str(avgR))

    score, sList = RLAlgs.score(model=model, env=env)
    print("avg reward " + str(score))


taxiTest()
