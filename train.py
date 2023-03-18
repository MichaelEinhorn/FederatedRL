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
parser.add_argument("--epoch", type=int, default=100, help="num epochs")
parser.add_argument("--num_models", type=int, default=1, help="num epochs")
parser.add_argument("--num_agents", type=int, default=16, help="num epochs")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = None
player = None
ppo = None
env= None
envKW = {}

modelPath = "models/"
def loadAll(fname, loadEnv=True):
    global model, player, ppo, env, envKW
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

def saveAll(fname):
    global model, player, ppo, env, envKW
    os.makedirs(modelPath + fname, exist_ok=True)
    torch.save(model.state_dict(), modelPath + fname + "/model.pth")
    torch.save(player.state_dict(), modelPath + fname + "/player.pth")
    torch.save(ppo.state_dict(), modelPath + fname + "/ppo.pth")
    torch.save(envKW, modelPath + fname + "/envKW.pth")
    torch.save(env.callmethod("get_state"), modelPath + fname + "/env_states.pth")
    torch.save(ppo.all_stats, modelPath + fname + "/stats.pth")


num_models = args.num_models
num_agents = args.num_agents

print("init model")
if "vector" in args.model.lowercase():
    if args.model == "resnet":
        submodel = CVModels.CNNAgent([64, 64, 3], 15, channels=16, layers=[1,1,1,1], scale=[1,1,1,1], vheadLayers=1)
        model = CVModels.VectorModelValue(submodel, num_models)
    elif "vit" in args.model.lowercase():
        if "big" in args.model.lowercase():
            submodel = CVModels.ViTValue(depth=4, num_heads=4, embed_dim=32, mlp_ratio=4, valueHeadLayers=1).to(device)
        else:
            submodel = CVModels.ViTValue().to(device)
        model = CVModels.VectorModelValue(submodel, num_models, syncFunc=CVModels.avgSync)
    model.to(device)
else:
    if args.model == "resnet":
        model = CVModels.CNNAgent([64, 64, 3], 15, channels=16, layers=[1,1,1,1], scale=[1,1,1,1], vheadLayers=1)
        model.load_state_dict(torch.load("resnet.pth"))
    elif "vit" in args.model.lowercase():
        if "big" in args.model.lowercase():
            model = CVModels.ViTValue(depth=4, num_heads=4, embed_dim=32, mlp_ratio=4, valueHeadLayers=1).to(device)
        else:
            model = CVModels.ViTValue().to(device)
    model.to(device)

print("init env")
envKW = core.getKW(num=num_models*num_agents, env_name="coinrun", distribution_mode="easy", paint_vel_info=True, use_backgrounds=False, restrict_themes=True)
env = ProcgenGym3Env(**envKW)
print(env.ob_space)
print(env.ac_space)

print("init player and ppo")
rewardScale = 8.0
terminateReward = 1 - 10.0 / rewardScale
livingReward = 0
print("terminateReward", terminateReward, "livingReward", livingReward, "discountedSumLiving", livingReward / (1 - 0.99)) # if terminate reward > discountedSumLiving the agent will perfer to run into obstacles.
player = VectorPlayer(env, num_agents=num_agents, num_models=num_models, epsilon=0.01, epsilon_decay=0.99, rewardScale=rewardScale, livingReward=0, terminateReward=terminateReward)
ppo = VectorPPO(model, env, num_agents=num_agents, num_models=num_models, player=player, gamma=0.99, weight_decay=0.0, warmup_steps=10, train_steps=1000, sync_epochs=1)

if os.isdir(modelPath + args.model):
    print("loading existing state")
    loadAll(args.model)
    
model.train()
print(summary(model, input_size=(2, 2, 3, 64, 64)))

for i in range(args.epoch):
    ppo.runGame()
    ppo.train()
    print(f"train epoch {i}/{args.epoch} episodeLength {ppo.all_stats[-1]['game/episodeLength']} episodeReward {ppo.all_stats[-1]['game/episodeReward']} stale {ppo.all_stats[-1]['game/staleSteps']}                    ", end="\n" if i % 10 == 0 else "\r")

# match regex with epoch{number}
# new model name substitute epoch{number + args.epoch}
saveAll(args.model)

