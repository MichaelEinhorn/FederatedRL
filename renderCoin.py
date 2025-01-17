import argparse
import torch
from torch.nn import functional as F
import gym
import gym3
from procgen import ProcgenGym3Env
from torchinfo import summary
import CVModels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet", help="Path to model file")
args = parser.parse_args()

if "vector" in args.model.lower():
    if args.model == "resnet":
        m = CVModels.CNNAgent([64, 64, 3], 15, channels=16, layers=[1,1,1,1], scale=[1,1,1,1], vheadLayers=1)
        model = CVModels.VectorModelValue(m, n)
        model.load_state_dict(torch.load("resnet.pth"))
    elif "vit" in args.model.lower():
        if "big" in args.model.lower():
            m = CVModels.ViTValue(depth=4, num_heads=4, embed_dim=32, mlp_ratio=4, valueHeadLayers=1).to(device)
        else:
            m = CVModels.ViTValue().to(device)
        model = CVModels.VectorModelValue(m, n)
        model.load_state_dict(torch.load(f"models/{args.model}/model.pth"))
    model.to(device)
else:
    if "impala" in args.model.lower():
        model = CVModels.ImpalaValue().to(device)
    elif args.model == "resnet":
        model = CVModels.CNNAgent([64, 64, 3], 15, channels=16, layers=[1,1,1,1], scale=[1,1,1,1], vheadLayers=1)
    elif "vit" in args.model.lower():
        if "big" in args.model.lower():
            model = CVModels.ViTValue(depth=4, num_heads=4, embed_dim=32, mlp_ratio=4, valueHeadLayers=1).to(device)
        else:
            model = CVModels.ViTValue().to(device)
    model.load_state_dict(torch.load(f"models/{args.model}/model.pth"))
    model.to(device)

model.eval()
envVideo = ProcgenGym3Env(num=1, env_name="coinrun", distribution_mode="easy", paint_vel_info=True, render_mode="rgb_array", use_backgrounds=False, restrict_themes=True)
envVideo = gym3.ViewerWrapper(envVideo, info_key="rgb")
while True:
    rew, obs, first = envVideo.observe()
    obs = torch.tensor(obs['rgb']).permute(0, 3, 1, 2).float().to(device)
    obs /= 255.0 # turn 0-255 to 0-1
    if "vector" in args.model.lower():
        logits, values = model.modelList[0](obs)
    else:
        logits, values = model(obs)
    actions = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
    envVideo.act(actions.to("cpu").numpy())
    print("action", actions[0].item(), "value", values[0, 0].item())