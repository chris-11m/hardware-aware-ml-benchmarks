
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def count_parameters(model):

    total = 0 

    for p in model.parameters():
        if p.requires_grad:
            number_of_values = p.numel()
            total += number_of_values

    return total


if __name__ == "__main__":
    input_dim = 105

    shallow = MLP(input_dim, [128])
    deep = MLP(input_dim, [64, 64, 64])

    print("Shallow parameters:", count_parameters(shallow))
    print("Deep parameters:", count_parameters(deep))
