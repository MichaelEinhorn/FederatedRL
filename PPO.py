import torch
import time

import datastructures
import core

import ProcgenPlayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO:
    default_params = {
        "alg_name": "ppo",
        # "lr": 1.41e-5,
        "lr": 1e-3,
        "gamma": 0.99,
        "lam": 0.95,
        "whiten": True,
        "cliprange": .2,
        "cliprange_value": .2,
        "vf_coef": 0.5,
        "epoch_steps": 256,
        "epochs_per_game": 1,
        # Entropy coefficient
        "ent_coef" : 0.0,
    }
    def __init__(self, model, env, num_agents, **params):
        self.params = self.default_params
        self.params.update(params)
        self.alg_name = self.params["alg_name"]
        
        self.env = env
        self.num_agents = num_agents

        self.model = model

        self.transitionBuffer = datastructures.LineBuffer(self.params["epoch_steps"])
        self.player = ProcgenPlayer.Player(env, num_agents=num_agents, transitionBuffer=self.transitionBuffer)

        self.dataset = datastructures.LineDataset(self.transitionBuffer, self.params["epoch_steps"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])

        self.all_stats = []

        self.epoch = 0
        self.steps = 0

        self.timing = {}
        

    @torch.no_grad()
    def runGame(self):
        self.transitionBuffer.clear()
        
        t = time.time()
        self.player.runGame(self.model, steps=self.params["epoch_steps"])
        self.timing["time/runGame"] = time.time() - t
        
        t = time.time()
        self.player.fillBuffer(gamma=self.params["gamma"], lam=self.params["lam"], whiten=self.params["whiten"])
        self.timing["time/computeAdvantages"] = time.time() - t

        self.steps += self.params["epoch_steps"] * self.num_agents

    def train(self, debug=False):
        end_epoch = self.epoch + self.params["epochs_per_game"]
        while self.epoch < end_epoch:
            step_stats = {}
            batchCount = 0
            epochTime = time.time()
            self.timing.update({f"time/{self.alg_name}/forward": 0, f"time/{self.alg_name}/backward": 0, f"time/{self.alg_name}/optim": 0, f"time/{self.alg_name}/stats": 0})
            

            for batch in self.dataset:
                reward, obs, action, first, old_values, old_logprobs, next_val, returns, advantages = batch
                batchCount += 1
                
                reward = reward.to(device)
                obs = obs.to(device)
                action = action.to(device)
                first = first.to(device)
                old_values = old_values.to(device)
                old_logprobs = old_logprobs.to(device)
                next_val = next_val.to(device)
                returns = returns.to(device).detach()
                advantages = advantages.to(device).detach()
                
                t = time.time()
                self.optimizer.zero_grad()
                self.timing[f"time/{self.alg_name}/optim"] += time.time() - t

                t = time.time()
                logits, vpred = self.model(obs)
                logprob = core.logprobs_from_logits(logits, action)
                self.timing[f"time/{self.alg_name}/forward"] += time.time() - t

                vpredclipped = core.clip_by_value(vpred, 
                                                  old_values - self.params["cliprange_value"], 
                                                  old_values + self.params["cliprange_value"])
                
                vf_losses1 = (vpred - returns) ** 2
                vf_losses2 = (vpredclipped - returns) ** 2
                vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
                vf_clipfrac = torch.mean(torch.gt(vf_losses2, vf_losses1).double())

                ratio = torch.exp(logprob - old_logprobs)
                pg_losses = -advantages * ratio
                pg_losses2 = -advantages * torch.clamp(ratio, 
                                                       1.0 - self.params["cliprange"], 
                                                       1.0 + self.params["cliprange"])
                
                pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
                pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

                loss = pg_loss + self.params['vf_coef'] * vf_loss

                entropy = torch.mean(core.entropy_from_logits(logits))
                ent_loss = -torch.mean(entropy)
                if self.params["ent_coef"] != 0.0:
                    loss += self.params["ent_coef"] * ent_loss

                if debug:
                    return loss
                else:
                    t = time.time()
                    loss.backward()
                    self.timing[f"time/{self.alg_name}/backward"] += time.time() - t
                    t = time.time()
                    self.optimizer.step()
                    self.timing[f"time/{self.alg_name}/optim"] += time.time() - t

                t = time.time()
                approxkl = .5 * torch.mean((logprob - old_logprobs) ** 2)
                policykl = torch.mean(logprob - old_logprobs)
                return_mean, return_var = torch.mean(returns), torch.var(returns)
                value_mean, value_var = torch.mean(old_values), torch.var(old_values)

                stats = dict(
                    reward = torch.mean(reward),
                    loss=dict(policy=pg_loss, 
                            value=self.params['vf_coef'] * vf_loss, 
                            ent=self.params["ent_coef"] * ent_loss, 
                            total=loss),
                    policy=dict(entropy=entropy, approxkl=approxkl, policykl=policykl, clipfrac=pg_clipfrac,
                                advantages_mean=torch.mean(advantages), ratio_mean=torch.mean(ratio)),
                    returns=dict(mean=return_mean, var=return_var),
                    val=dict(vpred=torch.mean(vpred), 
                            clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
                )
                stats = core.dict_to_cpu(core.flatten_dict(stats))
                # step_stats.append(stats)
                core.update_dict_add(step_stats, stats)
                self.timing[f"time/{self.alg_name}/stats"] += time.time() - t

            self.timing["time/epoch"] = time.time() - epochTime

            t = time.time()
            # step_stats = core.stack_stat_dicts(step_stats)
            
            # measure score
            train_stats = {"epoch": self.epoch, 
                          "steps": self.steps,
                          "episodeLength": self.player.meanEpsisodeLength,
                          "nonZeroReward": self.player.meanNonZeroRewards, 
                          'objective/vf_coef': self.params['vf_coef'],
                          'objective/ent_coef': self.params["ent_coef"],
                          'objective/lr': self.params["lr"]}
            for k, v in step_stats.items():
            # print(k, v)
                # train_stats[f'{self.alg_name}/{k}'] = torch.mean(v, axis=0)
                train_stats[f'{self.alg_name}/{k}'] = v / batchCount

            self.timing[f"time/{self.alg_name}/stats"] += time.time() - t

            train_stats.update(self.timing)
            train_stats.update(self.player.timing)
            self.timing = {}
            self.all_stats.append(train_stats)
            self.epoch += 1