import os
import argparse

import torch
from procgen import ProcgenGym3Env
from torchinfo import summary

import CVModels
from PPO import VectorPPO
from ProcgenPlayer import VectorPlayer
import core

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet", help="Path to model file")
parser.add_argument("--load", type=str, default=None, help="Path to model file")
parser.add_argument("--epoch", type=int, default=1000, help="num epochs")
parser.add_argument("--num_models", type=int, default=1, help="num epochs")
parser.add_argument("--num_agents", type=int, default=64, help="num epochs")
# sync
parser.add_argument("--syncFunc", type=str, default="avg", help="sync function")
parser.add_argument("--syncFreq", type=int, default=1, help="sync frequency in epochs")

parser.add_argument("--living_reward", type=float, default=0, help="living reward")
parser.add_argument("--game_name", type=str, default="coinrun", help="game name")

args = parser.parse_args()

# check for windows
windows = False
if os.name == 'nt':
    windows = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = None
player = None
ppo = None
env= None
envKW = {}

if windows:
    modelPath = "models/"
else:
    modelPath = "/storage/home/hcoda1/2/meinhorn6/scratch/RL/models/"

def loadAll(fname, loadEnv=True):
    global env, envKW
    model.load_state_dict(torch.load(modelPath + fname + "/model.pth"))
    player.load_state_dict(torch.load(modelPath + fname + "/player.pth"))
    ppo.load_state_dict(torch.load(modelPath + fname + "/ppo.pth"))
    if loadEnv:
        envKW = torch.load(modelPath + fname + "/envKW.pth")
        env = ProcgenGym3Env(**envKW)
        env.callmethod("set_state", torch.load(modelPath + fname + "/env_states.pth"))
    else:
        player.reset()
    ppo.all_stats = torch.load(modelPath + fname + "/stats.pth")
    model.stats["sync/comms"] = ppo.all_stats[-1]["sync/comms"]
    model.stats["sync/data"] = ppo.all_stats[-1]["sync/data"]

def saveAll(fname):
    os.makedirs(modelPath + fname, exist_ok=True)
    torch.save(model.state_dict(), modelPath + fname + "/model.pth")
    torch.save(player.state_dict(), modelPath + fname + "/player.pth")
    torch.save(ppo.state_dict(), modelPath + fname + "/ppo.pth")
    torch.save(envKW, modelPath + fname + "/envKW.pth")
    torch.save(env.callmethod("get_state"), modelPath + fname + "/env_states.pth")
    torch.save(ppo.all_stats, modelPath + fname + "/stats.pth")
    torch.save(player.trainEpisodeStats, modelPath + fname + "/trainEpisodeStats.pth")

def initModel(modelName, logging=False):
    # individual model
    if "impala" in modelName.lower(): # 626k
        model = CVModels.ImpalaValue()
    elif "vit620k" in modelName.lower(): # deep ViT
        model = CVModels.ViTValue(depth=12, num_heads=8, embed_dim=64, mlp_ratio=4, valueHeadLayers=1)
    elif "vit700k" in modelName.lower(): # wide ViT
        model = CVModels.ViTValue(depth=6, num_heads=8, embed_dim=96, mlp_ratio=4, valueHeadLayers=1)
    elif "vit60k" in modelName.lower():
        model = CVModels.ViTValue(depth=4, num_heads=4, embed_dim=32, mlp_ratio=4, valueHeadLayers=1)
    elif "vit15k" in modelName.lower():
        model = CVModels.ViTValue(depth=3, num_heads=4, embed_dim=16, mlp_ratio=4, valueHeadLayers=1)
    elif "resnet" in modelName.lower():
        model = CVModels.CNNAgent([64, 64, 3], 15, channels=16, layers=[1,1,1,1], scale=[1,1,1,1], vheadLayers=1)
    # vectorized model
    if "vector" in modelName.lower():
        model = CVModels.VectorModelValue(model, num_models, syncFunc=syncFunc)
        if logging:
            print(summary(model, input_shape=(2, 2, 3, 64, 64)))
    elif logging:
        print(summary(model, input_shape=(2, 3, 64, 64)))
    return model

modelName = args.model
num_epoch = args.epoch
num_models = args.num_models
num_agents = args.num_agents
syncFreq = args.syncFreq
livingReward = args.living_reward

gamma = 0.99
rewardScale = 10
terminateReward = 1 - 10.0 / rewardScale
lr = 2.5e-4
ent_coef = 1e-2
eps = 0.0

if args.syncFunc == "avg":
    syncFunc = CVModels.avgSync
elif args.syncFunc == "sum":
    syncFunc = CVModels.sumSync()

print("init model ", modelName)
model = initModel(modelName, logging=False).to(device)
model.train()

envKW = core.getKW(num=num_models*num_agents, env_name=args.game_name, distribution_mode="easy", paint_vel_info=True, use_backgrounds=False, restrict_themes=True, use_monochrome_assets=True)
print("init env", envKW)
env = ProcgenGym3Env(**envKW)
print(env.ob_space)
print(env.ac_space)

print("init player and ppo")
print("terminateReward", terminateReward, "livingReward", livingReward, "discountedSumLiving", livingReward / (1 - gamma)) # if terminate reward > discountedSumLiving the agent will perfer to run into obstacles.
player = VectorPlayer(env, num_agents=num_agents, num_models=num_models, 
                      epsilon=eps, epsilon_decay=1, 
                      rewardScale=rewardScale, livingReward=livingReward, terminateReward=terminateReward, 
                      finishedOnly=True, maxStaleSteps=64)
ppo = VectorPPO(model, env, num_agents=num_agents, num_models=num_models, player=player, 
                lr=lr, gamma=gamma, ent_coef=ent_coef, weight_decay=0.0, 
                warmup_steps=10, train_steps=1000, sync_epochs=syncFreq,
                batch_size=1, epochs_per_game=3)

print(ppo.params)
print(player.params)

lastFileName = None
if args.load is not None:
    if not os.path.isdir(modelPath + args.load):
        raise FileNotFoundError(f"load path {modelPath + args.load} is not a directory")
    print("loading existing state")
    lastFileName = args.load
    loadAll(lastFileName)
    
model.train()
for i in range(num_epoch // ppo.params['epochs_per_game']):
    if ppo.all_stats[-1]['epoch'] > num_epoch:
        break

    ppo.runGame()
    ppo.train()
    if i % 10 == 0:
        # print("episodeLength", ppo.all_stats[-1]["game/episodeLength"], "episodeReward", ppo.all_stats[-1]["game/episodeReward"],
        #       "epoch", ppo.all_stats[-1]["epoch"], "steps", ppo.all_stats[-1]["steps"], 
        #       "\nloss", ppo.all_stats[-1]["ppo/loss/total"].item(), "policy", ppo.all_stats[-1]["ppo/loss/policy"].item(), 
        #       "value", ppo.all_stats[-1]["ppo/loss/value"].item(),
        #       "entropy", ppo.all_stats[-1]["ppo/policy/entropy"].item())
        print(f"episodeLength {ppo.all_stats[-1]['game/episodeLength']} episodeReward {ppo.all_stats[-1]['game/episodeReward']} " + 
              f"\nepoch {ppo.all_stats[-1]['epoch']} steps {ppo.all_stats[-1]['steps']} " +
              f"\nloss {ppo.all_stats[-1]['ppo/loss/total']} policy {ppo.all_stats[-1]['ppo/loss/policy']} " +
              f"\nvalue {ppo.all_stats[-1]['ppo/loss/value']} entropy {ppo.all_stats[-1]['ppo/policy/entropy']} " +
              f"\ncomms {ppo.all_stats[-1]['sync/comms']} data {ppo.all_stats[-1]['sync/data']} " +
              f"\nstale {ppo.all_stats[-1]['game/staleSteps']}")
        if i % 100 == 0:
            print(ppo.all_stats[-1])
    else:
        print(f"epoch {ppo.all_stats[-1]['epoch']} episodeLength {ppo.all_stats[-1]['game/episodeLength']} episodeReward {ppo.all_stats[-1]['game/episodeReward']}               ", end="\r")
        
    if i % (50 // ppo.params['epochs_per_game']) == 0:
        print(f"saving E{ppo.all_stats[-1]['epoch']}")
        if lastFileName is not None: # move to backup folder
            os.rename(modelPath + lastFileName, modelPath + "backup/" + lastFileName)
        lastFileName = f"{modelName}-E{ppo.all_stats[-1]['epoch']}-RS{rewardScale}-G{gamma}-Lv{livingReward!=0}_4-16"
        saveAll(lastFileName)

print("final save")
if lastFileName is not None: # move to backup folder
    os.rename(modelPath + lastFileName, modelPath + "backup/" + lastFileName)
lastFileName = f"{modelName}-E{ppo.all_stats[-1]['epoch']}-RS{rewardScale}-G{gamma}-Lv{livingReward!=0}_4-16"
saveAll(lastFileName)

