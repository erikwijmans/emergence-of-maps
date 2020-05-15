import collections

import torch

from nav_analysis.rl.ppo.policy import Policy


class RNNMemoryBuffer(object):
    def __init__(
        self, actor_critic: Policy, num_processes: int, memory_length: int = None
    ):
        self.actor_critic = actor_critic
        self.memory_length = memory_length
        self.active = memory_length is not None
        self.num_processes = num_processes

        if self.active:
            self.input_buffers = [
                collections.deque(maxlen=memory_length) for _ in range(num_processes)
            ]

    def add(self, observations, prev_actions, masks, gt_hidden):
        self.gt_hidden = gt_hidden.clone()
        if not self.active:
            return

        for i in range(self.num_processes):
            if masks[i] == 0.0:
                self.input_buffers[i] = collections.deque(maxlen=self.memory_length)

            self.input_buffers[i].append(
                ({k: v[i] for k, v in observations.items()}, prev_actions[i], masks[i])
            )

    def get_hidden_states(self):
        if not self.active:
            return self.gt_hidden.clone()

        needs_hidden = [
            i
            for i in range(self.num_processes)
            if len(self.input_buffers[i]) == self.memory_length
        ]
        if len(needs_hidden) == 0:
            return self.gt_hidden

        obs = collections.defaultdict(list)
        prev_acts = []
        masks = []
        for i in range(self.memory_length):
            for j in needs_hidden:
                inp = self.input_buffers[j][i]
                for k, v in inp[0].items():
                    obs[k].append(v)

                prev_acts.append(inp[1])
                masks.append(inp[2])

        for k, v in obs.items():
            obs[k] = torch.stack(v, 0)

        prev_acts = torch.stack(prev_acts, 0)
        masks = torch.stack(masks, 0)

        new_hiddens = torch.zeros_like(self.gt_hidden)[:, needs_hidden]
        _, _, _, new_hiddens, _ = self.actor_critic.evaluate_actions(
            obs, new_hiddens, prev_acts, masks, prev_acts
        )

        ret_hiddens = self.gt_hidden.clone()
        ret_hiddens[:, needs_hidden] = new_hiddens
        return ret_hiddens

    def pause_at(self, idx):
        state_inds = list(range(self.num_processes))
        state_inds.pop(idx)
        self.gt_hidden = self.gt_hidden[:, state_inds]

        if self.active:
            self.input_buffers.pop(idx)

        self.num_processes -= 1
