from __future__ import print_function
import torch
from torch import nn


class motion_ae(nn.Module):

    def __init__(self, hidden_state):

        super(motion_ae, self).__init__()
        self.hidden_size = hidden_state
        self.encoder_fc = nn.Linear(4, self.hidden_size // 2)
        self.encoder = nn.GRU(self.hidden_size // 2, self.hidden_size, num_layers=1, batch_first=True)
        self.decoder = nn.GRU(4, self.hidden_size, num_layers=1, batch_first=True)
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 4)
        )

    def forward(self, observation, tf):
        observation_enc = nn.ReLU()(self.encoder_fc(observation))
        _, encoder_h = self.encoder(observation_enc)

        T = observation.shape[1]
        mask = np.random.uniform(size=T - 1) < tf

        reconstructed = []
        init_motion = observation[:, 0:1, :]

        x, h = self.decoder(init_motion, encoder_h)
        x = self.decoder_fc(x)
        x = x + init_motion

        reconstructed.append(x)

        for t in range(1, T):
            if mask[t - 1]:
                x_t, h = self.decoder(observation[:, t:t + 1, :], h)
            else:
                x_t, h = self.decoder(x, h)

            x_t = self.decoder_fc(x_t)
            x = x_t + x

            reconstructed.append(x)

        return torch.cat(reconstructed, dim=1)

    def inference(self, observation):
        observation = self.encoder_fc(observation)
        hiddens = []
        h = None
        for t in range(observation.shape[1]):
            _, h = self.encoder(observation[:, t:t+1, :], h)
            hiddens.append(h)
        hiddens = torch.cat(hiddens, dim=0)
        hiddens = hiddens.permute(1, 0, 2)
        return hiddens
