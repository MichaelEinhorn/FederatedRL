import numpy as np
import torch
from torch.nn import functional as F
import time
import core

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Player:
    def __init__(self, env, num_agents=1, transitionBuffer=None, rewardScale=10, epsilon=0.01):
        self.env = env
        self.transitions = []
        self.rng = torch.Generator(device=device).manual_seed(3701)
        self.num_agents = num_agents
        self.transitionBuffer = transitionBuffer
        self.rewardScale = rewardScale
        self.epsilon = epsilon

        # stats
        self.meanEpsisodeLength = 0
        self.meanNonZeroRewards = 0

        self.startSteps = [0 for i in range(self.num_agents)]
        self.zeroRew = [0 for i in range(self.num_agents)]

        self.timeStep = 0
        self.episodes = 0

        self.timing = {}

    @torch.no_grad()
    def runGame(self, model, steps=100):
        # used to calculate mean stats
        episodeCount = 0
        self.meanEpsisodeLength = 0
        self.meanNonZeroRewards = 0
        self.timing = {"time/game/observe": 0, "time/game/act": 0, "time/game/forward": 0, "time/game/stats": 0, "time/game/transition": 0}

        end_step = self.timeStep + steps
        while self.timeStep < end_step:
            t = time.time()
            rew, obs, first = self.env.observe()
            first = torch.tensor(first)
            rew = torch.tensor(rew) / self.rewardScale
            # info = env.get_info()
            # print(obs['rgb'].shape, rew, first, info)
            obs = torch.tensor(obs['rgb']).permute(0, 3, 1, 2).float().to(device)
            obs /= 255.0 # turn 0-255 to 0-1

            self.timing["time/game/observe"] += time.time() - t

            t = time.time()
            logits, values = model(obs)
            actions = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1, generator=self.rng).squeeze()
            # epsilon greedy
            if self.epsilon != 0.0:
                rand = torch.rand(self.num_agents, generator=self.rng ,device=device) < self.epsilon
                actions = torch.where(rand, torch.randint(0, 15, (self.num_agents,), generator=self.rng, device=device), actions)

            logp = core.logprobs_from_logits(logits, actions)
            self.timing["time/game/forward"] += time.time() - t

            actions = actions.to("cpu")
            values = values.to("cpu")
            
            t = time.time()
            self.env.act(actions.numpy())
            self.timing["time/game/act"] += time.time() - t

            # print(logits.shape)
            # print(actions.shape)
            
            t = time.time()
            for i in range(self.num_agents):
                # if len(self.transitions) != 0 and not first[i]:
                #     self.transitions[-1]["next_val"][i] = values[i]

                if rew[i] != 0 and self.zeroRew[i] == 0:
                    self.zeroRew[i] = 1

                if first[i] and self.timeStep != 0:
                    self.meanEpsisodeLength += self.timeStep - self.startSteps[i]
                    self.meanNonZeroRewards += self.zeroRew[i]

                    self.episodes += 1
                    episodeCount += 1
                    # reset
                    self.startSteps[i] = self.timeStep
                    self.zeroRew[i] = 0
            self.timing["time/game/stats"] += time.time() - t
            
            t = time.time()
            # self.transitions[-1]["done"] = first
            if len(self.transitions) != 0:
                self.transitions[-1]["next_val"] = torch.where(first, torch.zeros_like(values), values)
            self.addTransition(rew, obs, actions, first, values, logp)
            self.timing["time/game/transition"] += time.time() - t

            self.timeStep += 1

        if episodeCount != 0:
            self.meanEpsisodeLength /= episodeCount
            self.meanNonZeroRewards /= episodeCount

    @torch.no_grad()
    def computeAdvantages(self, gamma=0.99, lam=0.95, whiten=True):
        lastgaelam = np.zeros(self.num_agents)
        advantages_reversed = []
        returns_reversed = []
        for t in reversed(range(len(self.transitions))):
            trans_dict = self.transitions[t]
            reward = trans_dict["reward"]
            first = trans_dict["first"]
            values = trans_dict["val"]

            nextvalues = trans_dict["next_val"]
            delta = reward + gamma * nextvalues - values
            lastgaelam = delta + gamma * lam * lastgaelam

            advantages_reversed.append(lastgaelam)
            returns_reversed.append(lastgaelam + values)

            for i in range(self.num_agents):
                if first[i]:
                    lastgaelam[i] = 0

        returns_reversed.reverse()
        advantages_reversed.reverse()

        advantages_mean = 0
        advantages_std = 1
        if whiten:
            advantageTens = torch.stack(advantages_reversed)
            advantages_mean = torch.mean(advantageTens)
            advantages_std = torch.std(advantageTens)
        
        return returns_reversed, advantages_reversed, advantages_mean, advantages_std

    @torch.no_grad()
    def fillBuffer(self, gamma=0.99, lam=0.95, whiten=True):
        returns, advantages, advantages_mean, advantages_std = self.computeAdvantages(gamma, lam, whiten)
        for i in range(len(self.transitions)):
            trans_dict = self.transitions[i]
            # trans_dict["return"] = 
            # trans_dict["advantage"] 
            if whiten:
                adv = (advantages[i] - advantages_mean) / advantages_std
            else:
                adv = advantages[i]
            lineItem = (trans_dict["reward"], 
                        trans_dict["obs"],
                        trans_dict["action"],
                        trans_dict["first"], 
                        trans_dict["val"], 
                        trans_dict["logp"], 
                        trans_dict["next_val"],
                        returns[i], 
                        adv)
            self.transitionBuffer.append(lineItem)
        self.transitions = []

    @torch.no_grad()
    def addTransition(self, reward, obs, action, first, val, logp):
        trans_dict = core.dict_to_cpu({"reward": reward, 
                                       "obs": obs, 
                                       "action": action, 
                                       "first": first, 
                                       "val": val, 
                                       "logp": logp, 
                                       "next_val": torch.zeros_like(val)})
        self.transitions.append(trans_dict)
    