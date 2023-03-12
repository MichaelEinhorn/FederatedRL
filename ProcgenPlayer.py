import numpy as np
import torch
from torch.nn import functional as F
import time
import core

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Player:
    default_params = {
        "alg_name": "ppo",
        # "lr": 1.41e-5,
        "epsilon": 0.0, # epsilon greedy
        "epsilon_decay": 1, # every fill buffer
        # Reward Transforms
        "rewardScale": 10.0, # appied after scale
        "livingReward": 0.0, # divides env rew
        "terminateReward": 0.0, # adds to end of all episodes sucsessful or not
        # misc
        "finishedOnly": True, # keeps running until every game started before endStep is finished
    }
    def __init__(self, env, num_agents=1, transitionBuffer=None, **params):
        self.params = self.default_params
        self.params.update(params)
        self.alg_name = self.params["alg_name"]
        self.epsilon = self.params["epsilon"]
        
        self.env = env
        self.transitions = []
        self.rng = torch.Generator(device=device).manual_seed(3701)
        self.num_agents = num_agents
        self.transitionBuffer = transitionBuffer

        # stats
        self.meanEpsisodeLength = 0
        self.meanNonZeroRewards = 0
        self.meanEpisodeRewards = 0

        self.advantageMean = -1
        self.advantageStd = -1
        self.stepsPerGameLoop = 0 # steps param in last runGame
        self.staleSteps = 0 # extra transitions from previous loop

        self.timeStep = 0
        self.episodes = 0
        self.gameLoops = 0

        self.startSteps = [self.timeStep for i in range(self.num_agents)]
        self.zeroRew = [0 for i in range(self.num_agents)]
        self.sumRew = [0 for i in range(self.num_agents)]

        self.timing = {}

    def state_dict(self):
        return {
            "params": self.params,
            "timeStep": self.timeStep,
            "episodes": self.episodes,
            "gameLoops": self.gameLoops,
            "epsilon": self.epsilon,
            "transitions": self.transitions,
        }
    
    def load_state_dict(self, state_dict):
        self.params = state_dict["params"]
        self.timeStep = state_dict["timeStep"]
        self.episodes = state_dict["episodes"]
        self.gameLoops = state_dict["gameLoops"]
        self.epsilon = state_dict["epsilon"]
        self.transitions = state_dict["transitions"]

        self.startSteps = [self.timeStep for i in range(self.num_agents)]
        self.zeroRew = [0 for i in range(self.num_agents)]
        self.sumRew = [0 for i in range(self.num_agents)]

    def reset(self, rEnv=None, **params):
        if rEnv is not None:
            self.env = rEnv

        self.params.update(params)
        self.epsilon = self.params["epsilon"]

        self.transitions = []
        self.timeStep = 0
        self.episodes = 0
        self.gameLoops = 0
        self.startSteps = [self.timeStep for i in range(self.num_agents)]
        self.zeroRew = [0 for i in range(self.num_agents)]
        self.sumRew = [0 for i in range(self.num_agents)]


    @torch.no_grad()
    def runGame(self, model, steps=100):
        # used to calculate mean stats
        episodeCount = 0
        self.meanEpsisodeLength = 0
        self.meanNonZeroRewards = 0
        self.meanEpisodeRewards = 0
        self.timing = {"time/game/observe": 0, "time/game/act": 0, "time/game/forward": 0, "time/game/stats": 0, "time/game/transition": 0}

        self.stepsPerGameLoop = steps
        end_step = self.timeStep + steps
        if self.params["finishedOnly"]:
            end_step -= len(self.transitions)
            self.staleSteps = len(self.transitions)

        while (not self.params["finishedOnly"] and self.timeStep < end_step) or (self.params["finishedOnly"] and min(self.startSteps) < end_step):
            t = time.time()
            rew, obs, first = self.env.observe()
            
            first = torch.tensor(first)
            rew = torch.tensor(rew) / self.params["rewardScale"]
            rew += self.params["livingReward"]
            # info = env.get_info()
            # print(obs['rgb'].shape, rew, first, info)
            obs = torch.tensor(obs['rgb']).permute(0, 3, 1, 2).float().to(device)
            obs /= 255.0 # turn 0-255 to 0-1

            if self.num_agents == 1:
                rew = rew.unsqueeze(0)
                obs = obs.unsqueeze(0)
                first = first.unsqueeze(0)

            self.timing["time/game/observe"] += time.time() - t

            t = time.time()
            logits, values = model(obs)
            actions = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1, generator=self.rng).squeeze()
            # epsilon greedy
            if self.params["epsilon"] != 0.0:
                rand = torch.rand(self.num_agents, generator=self.rng ,device=device) < self.params["epsilon"]
                actions = torch.where(rand, torch.randint(0, 15, (self.num_agents,), generator=self.rng, device=device), actions)

            logp = core.logprobs_from_logits(logits, actions)
            self.timing["time/game/forward"] += time.time() - t

            actions = actions.to("cpu")
            values = values.to("cpu")
            
            t = time.time()
            if self.num_agents == 1:
                self.env.act(actions.squeeze().numpy())
            else:
                self.env.act(actions.numpy())
            self.timing["time/game/act"] += time.time() - t

            # print(logits.shape)
            # print(actions.shape)
            # print(rew.shape)
            t = time.time()
            for i in range(self.num_agents):
                if first[i]:
                    if len(self.transitions) != 0:
                        # reward transforms end of episode
                        if self.params["terminateReward"] != 0:
                            self.transitions[-1]["reward"][i] += self.params["terminateReward"]

                            # fix stats
                            self.sumRew[i] += self.params["terminateReward"]
                            if self.params["terminateReward"] > 0 and self.zeroRew[i] == 0:
                                self.zeroRew[i] = 1
                
                    if self.timeStep != 0:
                        # end of episode stats
                        self.meanEpsisodeLength += self.timeStep - self.startSteps[i]
                        self.meanNonZeroRewards += self.zeroRew[i]
                        self.meanEpisodeRewards += self.sumRew[i]

                        self.episodes += 1
                        episodeCount += 1
                        # reset
                        self.startSteps[i] = self.timeStep
                        self.zeroRew[i] = 0
                        self.sumRew[i] = 0

                # every step stats
                self.sumRew[i] += rew[i].item()
                if rew[i] > 0 and self.zeroRew[i] == 0:
                    self.zeroRew[i] = 1

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
            self.meanEpisodeRewards /= episodeCount

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
            advantageTens = torch.stack(advantages_reversed) # dim 0
            if self.params["finishedOnly"]:
                advantageTens = advantageTens[:self.stepsPerGameLoop] # only whiten advantages which will be returned
            advantages_mean = torch.mean(advantageTens)
            advantages_std = torch.std(advantageTens)

            self.advantageMean = advantages_mean.item()
            self.advantageStd = advantages_std.item()
        
        return returns_reversed, advantages_reversed, advantages_mean, advantages_std

    @torch.no_grad()
    def fillBuffer(self, gamma=0.99, lam=0.95, whiten=True):
        self.gameLoops += 1
        self.epsilon *= self.params["epsilon_decay"]

        returns, advantages, advantages_mean, advantages_std = self.computeAdvantages(gamma, lam, whiten)
        for i in range(len(self.transitions)):
            if self.params["finishedOnly"] and i >= self.stepsPerGameLoop:
                break

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

        if self.params["finishedOnly"]:
            self.transitions = self.transitions[self.stepsPerGameLoop:] # keep the last transitions for the next game loop
        else:
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
    
    @torch.no_grad()
    def getStats(self):
        stats = {
            "episodeLength": self.meanEpsisodeLength, # of completed episodes
            "nonZeroReward": self.meanNonZeroRewards,  # completed episodes with positive reward
            "episodeReward": self.meanEpisodeRewards, # of completed episodes
            "advantageMean_PreWhiten": self.advantageMean, # of first stepsPerGameLoop steps
            "advantageStd_PreWhiten": self.advantageStd, # of first stepsPerGameLoop steps
            "staleSteps": self.staleSteps # extra steps from previous run game loop
            }
        stats = core.dict_to_cpu(core.flatten_dict(stats, sep="/", prefix="game/"))
        return stats
    
class VectorPlayer:
    default_params = {
        "alg_name": "ppo",
        # "lr": 1.41e-5,
        "epsilon": 0.0, # epsilon greedy
        "epsilon_decay": 1, # every fill buffer
        # Reward Transforms
        "rewardScale": 10.0, # appied after scale
        "livingReward": 0.0, # divides env rew
        "terminateReward": 0.0, # adds to end of all episodes sucsessful or not
        # misc
        "finishedOnly": True, # keeps running until every game started before endStep is finished
    }
    def __init__(self, env, num_agents=1, num_models=1, transitionBuffer=None, **params):
        self.params = self.default_params
        self.params.update(params)
        self.alg_name = self.params["alg_name"]
        self.epsilon = self.params["epsilon"]
        
        self.env = env
        self.transitions = []
        self.rng = torch.Generator(device=device).manual_seed(3701)
        self.num_agents = num_agents
        self.num_models = num_models
        self.transitionBuffer = transitionBuffer

        # stats
        self.meanEpsisodeLength = 0
        self.meanNonZeroRewards = 0
        self.meanEpisodeRewards = 0

        self.advantageMean = -1
        self.advantageStd = -1
        self.stepsPerGameLoop = 0 # steps param in last runGame
        self.staleSteps = 0 # extra transitions from previous loop

        self.timeStep = 0
        self.episodes = 0
        self.gameLoops = 0

        self.startSteps = [self.timeStep for i in range(self.num_agents)]
        self.zeroRew = [0 for i in range(self.num_agents)]
        self.sumRew = [0 for i in range(self.num_agents)]

        self.timing = {}

    def state_dict(self):
        return {
            "params": self.params,
            "timeStep": self.timeStep,
            "episodes": self.episodes,
            "gameLoops": self.gameLoops,
            "epsilon": self.epsilon,
            "transitions": self.transitions,
        }
    
    def load_state_dict(self, state_dict):
        self.params = state_dict["params"]
        self.timeStep = state_dict["timeStep"]
        self.episodes = state_dict["episodes"]
        self.gameLoops = state_dict["gameLoops"]
        self.epsilon = state_dict["epsilon"]
        self.transitions = state_dict["transitions"]

        self.startSteps = [self.timeStep for i in range(self.num_agents)]
        self.zeroRew = [0 for i in range(self.num_agents)]
        self.sumRew = [0 for i in range(self.num_agents)]

    def reset(self, rEnv=None, **params):
        if rEnv is not None:
            self.env = rEnv

        self.params.update(params)
        self.epsilon = self.params["epsilon"]

        self.transitions = []
        self.timeStep = 0
        self.episodes = 0
        self.gameLoops = 0
        self.startSteps = [self.timeStep for i in range(self.num_agents)]
        self.zeroRew = [0 for i in range(self.num_agents)]
        self.sumRew = [0 for i in range(self.num_agents)]


    @torch.no_grad()
    def runGame(self, model, steps=100):
        # used to calculate mean stats
        episodeCount = 0
        self.meanEpsisodeLength = 0
        self.meanNonZeroRewards = 0
        self.meanEpisodeRewards = 0
        self.timing = {"time/game/observe": 0, "time/game/act": 0, "time/game/forward": 0, "time/game/stats": 0, "time/game/transition": 0}

        self.stepsPerGameLoop = steps
        end_step = self.timeStep + steps
        if self.params["finishedOnly"]:
            end_step -= len(self.transitions)
            self.staleSteps = len(self.transitions)

        while (not self.params["finishedOnly"] and self.timeStep < end_step) or (self.params["finishedOnly"] and min(self.startSteps) < end_step):
            t = time.time()
            rew, obs, first = self.env.observe()
            
            first = torch.tensor(first)
            rew = torch.tensor(rew) / self.params["rewardScale"]
            rew += self.params["livingReward"]
            # info = env.get_info()
            # print(obs['rgb'].shape, rew, first, info)
            obs = torch.tensor(obs['rgb']).permute(0, 3, 1, 2).float().to(device)
            obs /= 255.0 # turn 0-255 to 0-1

            if self.num_agents == 1:
                rew = rew.unsqueeze(0)
                obs = obs.unsqueeze(0)
                first = first.unsqueeze(0)

            self.timing["time/game/observe"] += time.time() - t

            t = time.time()
            logits, values = model(obs)
            actions = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1, generator=self.rng).squeeze()
            # epsilon greedy
            if self.params["epsilon"] != 0.0:
                rand = torch.rand(self.num_agents, generator=self.rng ,device=device) < self.params["epsilon"]
                actions = torch.where(rand, torch.randint(0, 15, (self.num_agents,), generator=self.rng, device=device), actions)

            logp = core.logprobs_from_logits(logits, actions)
            self.timing["time/game/forward"] += time.time() - t

            actions = actions.to("cpu")
            values = values.to("cpu")
            
            t = time.time()
            if self.num_agents == 1:
                self.env.act(actions.squeeze().numpy())
            else:
                self.env.act(actions.numpy())
            self.timing["time/game/act"] += time.time() - t

            # print(logits.shape)
            # print(actions.shape)
            # print(rew.shape)
            t = time.time()
            for i in range(self.num_agents):
                if first[i]:
                    if len(self.transitions) != 0:
                        # reward transforms end of episode
                        if self.params["terminateReward"] != 0:
                            self.transitions[-1]["reward"][i] += self.params["terminateReward"]

                            # fix stats
                            self.sumRew[i] += self.params["terminateReward"]
                            if self.params["terminateReward"] > 0 and self.zeroRew[i] == 0:
                                self.zeroRew[i] = 1
                
                    if self.timeStep != 0:
                        # end of episode stats
                        self.meanEpsisodeLength += self.timeStep - self.startSteps[i]
                        self.meanNonZeroRewards += self.zeroRew[i]
                        self.meanEpisodeRewards += self.sumRew[i]

                        self.episodes += 1
                        episodeCount += 1
                        # reset
                        self.startSteps[i] = self.timeStep
                        self.zeroRew[i] = 0
                        self.sumRew[i] = 0

                # every step stats
                self.sumRew[i] += rew[i].item()
                if rew[i] > 0 and self.zeroRew[i] == 0:
                    self.zeroRew[i] = 1

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
            self.meanEpisodeRewards /= episodeCount

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
            advantageTens = torch.stack(advantages_reversed) # dim 0
            if self.params["finishedOnly"]:
                advantageTens = advantageTens[:self.stepsPerGameLoop] # only whiten advantages which will be returned
            advantages_mean = torch.mean(advantageTens)
            advantages_std = torch.std(advantageTens)

            self.advantageMean = advantages_mean.item()
            self.advantageStd = advantages_std.item()
        
        return returns_reversed, advantages_reversed, advantages_mean, advantages_std

    @torch.no_grad()
    def fillBuffer(self, gamma=0.99, lam=0.95, whiten=True):
        self.gameLoops += 1
        self.epsilon *= self.params["epsilon_decay"]

        returns, advantages, advantages_mean, advantages_std = self.computeAdvantages(gamma, lam, whiten)
        for i in range(len(self.transitions)):
            if self.params["finishedOnly"] and i >= self.stepsPerGameLoop:
                break

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

        if self.params["finishedOnly"]:
            self.transitions = self.transitions[self.stepsPerGameLoop:] # keep the last transitions for the next game loop
        else:
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
    
    @torch.no_grad()
    def getStats(self):
        stats = {
            "episodeLength": self.meanEpsisodeLength, # of completed episodes
            "nonZeroReward": self.meanNonZeroRewards,  # completed episodes with positive reward
            "episodeReward": self.meanEpisodeRewards, # of completed episodes
            "advantageMean_PreWhiten": self.advantageMean, # of first stepsPerGameLoop steps
            "advantageStd_PreWhiten": self.advantageStd, # of first stepsPerGameLoop steps
            "staleSteps": self.staleSteps # extra steps from previous run game loop
            }
        stats = core.dict_to_cpu(core.flatten_dict(stats, sep="/", prefix="game/"))
        return stats