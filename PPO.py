import torch
import time
import transformers
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
        # optimizer
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "train_steps": 1000,
    }
    def __init__(self, model, env, num_agents=1, player=None, **params):
        self.params = self.default_params
        self.params.update(params)
        self.alg_name = self.params["alg_name"]
        
        self.env = env
        self.num_agents = num_agents

        self.model = model

        self.transitionBuffer = datastructures.LineBuffer(self.params["epoch_steps"])
        self.player = player
        self.player.transitionBuffer = self.transitionBuffer

        self.dataset = datastructures.LineDataset(self.transitionBuffer, self.params["epoch_steps"])
        self.optimizer = transformers.AdamW(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"])
        self.scheduler = transformers.get_cosine_schedule_with_warmup(self.optimizer, self.params["warmup_steps"], self.params["train_steps"])
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])

        self.all_stats = []

        self.epoch = 0
        self.steps = 0

        self.timing = {}

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "params": self.params,
            "epoch": self.epoch,
            "steps": self.steps,
            "stats": self.all_stats,
        }
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.params = state_dict["params"]
        self.epoch = state_dict["epoch"]
        self.steps = state_dict["steps"]
        self.all_stats = state_dict["stats"]
        

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
                          'objective/vf_coef': self.params['vf_coef'],
                          'objective/ent_coef': self.params["ent_coef"],
                          'objective/lr': self.scheduler.get_last_lr() if self.scheduler is not None else self.params['lr'],
                          }
            train_stats.update(self.player.getStats())
            
            # learning rate schedule
            if self.scheduler is not None:
                self.scheduler.step()

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

class VectorPPO:
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
        # optimizer
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "train_steps": 1000,
    }
    def __init__(self, model, env, num_agents=1, num_models=1, player=None, **params):
        self.params = self.default_params
        self.params.update(params)
        self.alg_name = self.params["alg_name"]
        
        self.env = env
        self.num_agents = num_agents
        self.num_models = num_models

        self.model = model

        self.transitionBuffer = datastructures.LineBuffer(self.params["epoch_steps"])
        self.player = player
        self.player.transitionBuffer = self.transitionBuffer

        self.dataset = datastructures.LineDataset(self.transitionBuffer, self.params["epoch_steps"])

        # separate optimizer for each model. Not sure if a single optimizer for all params would actually be the same as independent training
        self.optimzers = []
        self.schedulers = []
        for i in range(self.num_models):
            self.optimzers.append(transformers.AdamW(self.modelList[i].parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"]))
            self.schedulers.append(transformers.get_cosine_schedule_with_warmup(self.optimzers[-1], self.params["warmup_steps"], self.params["train_steps"]))
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])

        self.all_stats = []

        self.epoch = 0
        self.steps = 0

        self.timing = {}

    def state_dict(self):
        return {
            "optimizer": [self.optimizers[i].state_dict() for i in range(self.num_models)],
            "scheduler": [self.schedulers[i].state_dict() for i in range(self.num_models)],
            "params": self.params,
            "epoch": self.epoch,
            "steps": self.steps,
            "stats": self.all_stats,
        }
    
    def load_state_dict(self, state_dict):
        for i in range(self.num_models):
            self.optimizers[i].load_state_dict(state_dict["optimizer"][i])
            self.schedulers[i].load_state_dict(state_dict["scheduler"][i])
        self.params = state_dict["params"]
        self.epoch = state_dict["epoch"]
        self.steps = state_dict["steps"]
        self.all_stats = state_dict["stats"]
        

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
                for i in range(self.num_models):
                    self.optimzers[i].zero_grad()
                self.timing[f"time/{self.alg_name}/optim"] += time.time() - t

                t = time.time()
                logits, vpred = self.model(obs)
                logprob = core.logprobs_from_logits(logits, action) # gathers on last dimension keeps Model x Batch
                self.timing[f"time/{self.alg_name}/forward"] += time.time() - t

                vpredclipped = core.clip_by_value(vpred, 
                                                  old_values - self.params["cliprange_value"], 
                                                  old_values + self.params["cliprange_value"])
                
                vf_losses1 = (vpred - returns) ** 2
                vf_losses2 = (vpredclipped - returns) ** 2
                # mean over all dimensions except model dim
                vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2), dim=(i for i in range(1, len(vf_losses1.shape))))
                vf_clipfrac = torch.mean(torch.gt(vf_losses2, vf_losses1).double(), dim=(i for i in range(1, len(vf_losses1.shape))))

                ratio = torch.exp(logprob - old_logprobs)
                pg_losses = -advantages * ratio
                pg_losses2 = -advantages * torch.clamp(ratio, 
                                                       1.0 - self.params["cliprange"], 
                                                       1.0 + self.params["cliprange"])
                
                # mean over all dimensions except model dim
                pg_loss = torch.mean(torch.max(pg_losses, pg_losses2), dim=(i for i in range(1, len(pg_losses.shape))))
                pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double(), dim=(i for i in range(1, len(pg_losses.shape))))

                loss = pg_loss + self.params['vf_coef'] * vf_loss

                # mean over all dimensions except model dim
                entropy = torch.mean(core.entropy_from_logits(logits), dim=(i for i in range(1, len(logits.shape))))
                ent_loss = -entropy
                if self.params["ent_coef"] != 0.0:
                    loss += self.params["ent_coef"] * ent_loss

                if debug:
                    return loss
                else:
                    t = time.time()
                    loss.backward()
                    self.timing[f"time/{self.alg_name}/backward"] += time.time() - t
                    t = time.time()
                    for i in range(self.num_models):
                        self.optimzers[i].step()
                    self.timing[f"time/{self.alg_name}/optim"] += time.time() - t

                t = time.time()
                approxkl = .5 * torch.mean((logprob - old_logprobs) ** 2, dim=(i for i in range(1, len(logprob.shape))))
                policykl = torch.mean(logprob - old_logprobs, dim=(i for i in range(1, len(logprob.shape))))
                return_mean, return_var = torch.mean(returns, dim=(i for i in range(1, len(returns.shape)))), torch.var(returns, dim=(i for i in range(1, len(returns.shape))))
                value_mean, value_var = torch.mean(old_values, dim=(i for i in range(1, len(old_values.shape)))), torch.var(old_values, dim=(i for i in range(1, len(old_values.shape))))

                stats = dict(
                    reward = torch.mean(reward, dim=(i for i in range(1, len(reward.shape)))),
                    loss=dict(policy=pg_loss, 
                            value=self.params['vf_coef'] * vf_loss, 
                            ent=self.params["ent_coef"] * ent_loss, 
                            total=loss),
                    policy=dict(entropy=entropy, approxkl=approxkl, policykl=policykl, clipfrac=pg_clipfrac,
                                advantages_mean=torch.mean(advantages, dim=(i for i in range(1, len(advantages.shape)))),
                                ratio_mean=torch.mean(ratio, dim=(i for i in range(1, len(ratio.shape))))),
                    returns=dict(mean=return_mean, var=return_var),
                    val=dict(vpred=torch.mean(vpred, dim=(i for i in range(1, len(vpred.shape)))), 
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
                          'objective/vf_coef': self.params['vf_coef'],
                          'objective/ent_coef': self.params["ent_coef"],
                          'objective/lr': self.schedulers[0].get_last_lr() if self.schedulers is not None else self.params['lr'],
                          }
            train_stats.update(self.player.getStats())
            
            # learning rate schedule
            if self.schedulers is not None:
                for i in range(self.num_models):
                    self.schedulers[i].step()

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