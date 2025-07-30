import torch
import torch.nn as nn


# Input : Joint Observation (n * obs_size)
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        x = self.build_state(state)

        return self.network(x)

    def build_state(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)

        return state_tensor.unsqueeze(0)
