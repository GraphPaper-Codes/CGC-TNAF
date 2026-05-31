import torch
import torch.nn as nn
import torch.nn.functional as F
from opt import args

class DualProjectionModel(nn.Module):
    def __init__(self, dims):
        super().__init__()

        input_dim, hidden_dim = dims[0], dims[1]

        self.projection_a = nn.Linear(input_dim, hidden_dim)
        self.projection_b = nn.Linear(input_dim, hidden_dim)

        self.alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))

        self._initialize_mixing_parameters()

    def _initialize_mixing_parameters(self):
        if args.dataset == "citeseer":
            initial_alpha = torch.sigmoid(torch.tensor(0.5))
            initial_beta = torch.sigmoid(torch.tensor(0.99999))
        else:
            initial_alpha = torch.tensor(0.5)
            initial_beta = torch.tensor(0.99999)

        self.alpha.data.copy_(initial_alpha.to(args.device))
        self.beta.data.copy_(initial_beta.to(args.device))

    def _normalized_projection(self, layer, x):
        return F.normalize(layer(x), p=2, dim=1)

    def _add_training_noise(self, embedding, enabled):
        if not enabled:
            return embedding

        noise = torch.randn_like(embedding) * args.sigma
        return embedding + noise.to(embedding.device)

    def forward(self, x_lmh, is_train=True):
        x_lmh = x_lmh.float()

        z_left = self._normalized_projection(self.projection_a, x_lmh)
        z_right = self._normalized_projection(self.projection_b, x_lmh)

        z_right_noisy = self._add_training_noise(
            z_right,
            enabled=is_train
        )

        return z_left, z_right, z_right_noisy


class ContrastiveObjective(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def _scaled_exp_similarity(self, x, y):
        similarity = F.cosine_similarity(x, y)
        return torch.exp(similarity / self.temperature)

    def forward(self, z1, z2, z2_s):
        positive_score = self._scaled_exp_similarity(z1, z2)

        negative_score_1 = self._scaled_exp_similarity(z1, z2_s)
        negative_score_2 = self._scaled_exp_similarity(z2, z2_s)

        denominator = positive_score + negative_score_1 + negative_score_2

        contrastive_loss = -torch.log(positive_score / denominator)

        return contrastive_loss.mean()