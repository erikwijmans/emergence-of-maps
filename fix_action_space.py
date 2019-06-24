import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--output", type=str, default=None)

args = parser.parse_args()

ckpt = torch.load(args.path, map_location="cpu")
state = ckpt["state_dict"]

reordering = torch.tensor([3, 0, 1, 2], dtype=torch.long)
for k in [
    "actor_critic.action_distribution.linear.weight",
    "actor_critic.action_distribution.linear.bias",
]:
    state[k] = state[k][reordering]


ckpt["state_dict"] = state
torch.save(ckpt, args.output if args.output is not None else args.path)
