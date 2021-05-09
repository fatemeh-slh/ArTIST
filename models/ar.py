from __future__ import print_function
import torch
from torch import nn

from .residual_block import ResidualBlock


class motion_ar(nn.Module):

    def __init__(self, hidden_state=512, num_clusters=1024):

        super(motion_ar, self).__init__()
        self.hidden_size = hidden_state
        self.num_clusters = num_clusters
        self.fc_embedding = ResidualBlock(4, self.hidden_size)
        self.embedding = nn.Sequential(nn.Linear(self.hidden_size + 256, self.hidden_size), nn.ReLU())
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)

        self.fc_x = nn.Linear(self.hidden_size, num_clusters)
        self.fc_y = nn.Linear(self.hidden_size, num_clusters)
        self.fc_w = nn.Linear(self.hidden_size, num_clusters)
        self.fc_h = nn.Linear(self.hidden_size, num_clusters)

    def init_hidden(self, bsz):
        return (nn.Parameter(torch.zeros(self.rnn.num_layers, bsz, self.hidden_size).normal_(0, 0.01), requires_grad=True).cuda(),
                nn.Parameter(torch.zeros(self.rnn.num_layers, bsz, self.hidden_size).normal_(0, 0.01),
                             requires_grad=True).cuda())

    def sample_cluster(self, dist, centroid):
        indices = (torch.topk(nn.Softmax(dim=-1)(dist), k=1)[1]).squeeze(1)
        return centroid[0, list(indices.squeeze(1))].unsqueeze(1)

    def sample_cluster_multinomial(self, dist, centroid):
        m = torch.distributions.multinomial.Multinomial(total_count=self.num_clusters, probs=nn.Softmax(dim=-1)(dist))
        indices = torch.multinomial(nn.Softmax(dim=-1)(dist)[:, 0], 1)

        return centroid[0, list(indices)].unsqueeze(1)

    def single_forward(self, x, s, h):

        x = self.fc_embedding(x)
        x_s = self.embedding(torch.cat([x, s], dim=2))
        x_orig = x.clone()

        x, h = self.rnn(x_s, h)
        x += x_orig

        featureX = self.fc_x(x)
        featureY = self.fc_y(x)
        featureW = self.fc_w(x)
        featureH = self.fc_h(x)

        return h, featureX, featureY, featureW, featureH

    def forward(self, observation, social, mask_index, centroid_x, centroid_y, centroid_w, centroid_h):

        # observation and last box are bboxes not delta!
        B, T = observation.shape[0], observation.shape[1]
        sampled_delta = torch.zeros(observation.shape).cuda()

        # h = init_h
        h = self.init_hidden(B)

        reconstructed_x = []
        reconstructed_y = []
        reconstructed_w = []
        reconstructed_h = []

        init_motion = observation[:, 0:1, :]
        init_social = social[:, 0:1, :]
        sampled_delta[:, 0:1, :] = observation[:, 0:1, :].clone()
        pos = torch.zeros(B, 1, 1).cuda()

        h, featureX, featureY, featureW, featureH = self.single_forward(init_motion, init_social, h)

        featureX = featureX.view(B, 1, -1)
        featureY = featureY.view(B, 1, -1)
        featureW = featureW.view(B, 1, -1)
        featureH = featureH.view(B, 1, -1)

        reconstructed_x.append(featureX)
        reconstructed_y.append(featureY)
        reconstructed_w.append(featureW)
        reconstructed_h.append(featureH)

        for t in range(1, T - 1):
            if t < mask_index:
                sampled_delta[:, t:t + 1, :] = observation[:, t:t + 1, :].clone()
                h, featureX, featureY, featureW, featureH = self.single_forward(observation[:, t:t+1, :], social[:, t:t+1, :], h)
            else:
                sampled_x = torch.zeros(B, 1, 4).cuda()

                sampled_x[:, :, 0] = self.sample_cluster(featureX, centroid_x)
                sampled_x[:, :, 1] = self.sample_cluster(featureY, centroid_y)
                sampled_x[:, :, 2] = self.sample_cluster(featureW, centroid_w)
                sampled_x[:, :, 3] = self.sample_cluster(featureH, centroid_h)

                sampled_delta[:, t:t+1, :] = sampled_x.clone()
                h, featureX, featureY, featureW, featureH = self.single_forward(sampled_x, social[:, t:t+1, :], h)

            featureX = featureX.view(B, 1, -1)
            featureY = featureY.view(B, 1, -1)
            featureW = featureW.view(B, 1, -1)
            featureH = featureH.view(B, 1, -1)

            reconstructed_x.append(featureX)
            reconstructed_y.append(featureY)
            reconstructed_w.append(featureW)
            reconstructed_h.append(featureH)

        sampled_delta[:, -1:, :] = observation[:, -1:, :].clone()

        return sampled_delta, torch.cat(reconstructed_x, dim=1), torch.cat(reconstructed_y, dim=1),\
               torch.cat(reconstructed_w, dim=1), torch.cat(reconstructed_h, dim=1)

    def inference(self, observation, social, gap, centroid_x, centroid_y, centroid_w, centroid_h):
        # observation and last box are bboxes not delta!
        B, mask_index = observation.shape[0], observation.shape[1]

        T = mask_index + gap

        # h = init_h
        h = self.init_hidden(B)

        reconstructed_x = []
        reconstructed_y = []
        reconstructed_w = []
        reconstructed_h = []

        generated_bbox = []

        last_bbox = [observation[0, -1, 0].item(),
                     observation[0, -1, 1].item(),
                     observation[0, -1, 2].item(),
                     observation[0, -1, 3].item()]

        generated_delta = torch.zeros(1, gap, 4)
        delta_cnt = 0

        # new alternative
        if observation.shape[1] == 1:
            current_obs = observation[:, 0:1, :] * 0.0
            h, featureX_obs, featureY_obs, featureW_obs, featureH_obs = self.single_forward(current_obs, social[:, :observation.shape[1], :], h)
            featureX, featureY, featureW, featureH = featureX_obs[:, -1, :].unsqueeze(1), featureY_obs[:, -1, :].unsqueeze(1), featureW_obs[:, -1, :].unsqueeze(1), featureH_obs[:, -1, :].unsqueeze(1)
        else:
            current_obs = observation[:, 1:, :] - observation[:, :-1, :]
            # current_obs = torch.cat([torch.zeros(current_obs.shape[0], 1, current_obs.shape[2]).cuda(), current_obs], dim=1)
            h, featureX_obs, featureY_obs, featureW_obs, featureH_obs = self.single_forward(current_obs, social[:, :current_obs.shape[1], :], h)
            featureX, featureY, featureW, featureH = featureX_obs[:, -1, :].unsqueeze(1), featureY_obs[:, -1, :].unsqueeze(1), featureW_obs[:, -1, :].unsqueeze(1), featureH_obs[:, -1, :].unsqueeze(1)


        # new alternative
        for t in range(observation.shape[1], observation.shape[1] + gap - 1):
            sampled_x = torch.zeros(B, 1, 4).cuda()
            sampled_x[:, :, 0] = self.sample_cluster_multinomial(featureX, centroid_x)
            sampled_x[:, :, 1] = self.sample_cluster_multinomial(featureY, centroid_y)
            sampled_x[:, :, 2] = self.sample_cluster_multinomial(featureW, centroid_w)
            sampled_x[:, :, 3] = self.sample_cluster_multinomial(featureH, centroid_h)

            # sampled_x[:, :, 0] = self.sample_cluster(featureX, centroid_x)
            # sampled_x[:, :, 1] = self.sample_cluster(featureY, centroid_y)
            # sampled_x[:, :, 2] = self.sample_cluster(featureW, centroid_w)
            # sampled_x[:, :, 3] = self.sample_cluster(featureH, centroid_h)

            generated_delta[0, delta_cnt] = sampled_x[0, 0]
            delta_cnt += 1

            last_bbox = [last_bbox[0] + sampled_x[0, 0, 0].item(),
                         last_bbox[1] + sampled_x[0, 0, 1].item(),
                         last_bbox[2] + sampled_x[0, 0, 2].item(),
                         last_bbox[3] + sampled_x[0, 0, 3].item()]

            generated_bbox.append(last_bbox)
            current_social = social[:, t:t + 1, :]
            h, featureX, featureY, featureW, featureH = self.single_forward(sampled_x, current_social, h)

            featureX = featureX.view(B, 1, -1)
            featureY = featureY.view(B, 1, -1)
            featureW = featureW.view(B, 1, -1)
            featureH = featureH.view(B, 1, -1)

            reconstructed_x.append(featureX)
            reconstructed_y.append(featureY)
            reconstructed_w.append(featureW)
            reconstructed_h.append(featureH)

        sampled_detection = torch.zeros(B, 1, 4).cuda()
        sampled_detection[:, :, 0] = self.sample_cluster_multinomial(featureX, centroid_x)
        sampled_detection[:, :, 1] = self.sample_cluster_multinomial(featureY, centroid_y)
        sampled_detection[:, :, 2] = self.sample_cluster_multinomial(featureW, centroid_w)
        sampled_detection[:, :, 3] = self.sample_cluster_multinomial(featureH, centroid_h)
        # sampled_detection[:, :, 0] = self.sample_cluster(featureX, centroid_x)
        # sampled_detection[:, :, 1] = self.sample_cluster(featureY, centroid_y)
        # sampled_detection[:, :, 2] = self.sample_cluster(featureW, centroid_w)
        # sampled_detection[:, :, 3] = self.sample_cluster(featureH, centroid_h)

        last_detection = [last_bbox[0] + sampled_detection[0, 0, 0].item(),
                          last_bbox[1] + sampled_detection[0, 0, 1].item(),
                          last_bbox[2] + sampled_detection[0, 0, 2].item(),
                          last_bbox[3] + sampled_detection[0, 0, 3].item()]

        if len(reconstructed_x) > 0:
            reconstructed_x, reconstructed_y, reconstructed_w, reconstructed_h = torch.cat(reconstructed_x, dim=1),\
            torch.cat(reconstructed_y, dim=1), torch.cat(reconstructed_w, dim=1), torch.cat(reconstructed_h, dim=1)

            reconstructed_x, reconstructed_y, reconstructed_w, reconstructed_h = torch.cat([featureX_obs, reconstructed_x], dim=1), \
                                                                                 torch.cat([featureY_obs, reconstructed_y], dim=1), \
                                                                                 torch.cat([featureW_obs, reconstructed_w], dim=1), \
                                                                                 torch.cat([featureH_obs, reconstructed_h], dim=1)
        else:
            reconstructed_x, reconstructed_y, reconstructed_w, reconstructed_h = featureX_obs, featureY_obs, featureW_obs, featureH_obs
        return reconstructed_x, reconstructed_y, reconstructed_w, reconstructed_h, \
               generated_bbox, generated_delta, last_detection

    def batch_inference(self, observation, social, gap, centroid_x, centroid_y, centroid_w, centroid_h):
        # observation and last box are bboxes not delta!
        B, mask_index = observation.shape[0], observation.shape[1]

        T = mask_index + gap

        # h = init_h
        h = self.init_hidden(B)

        reconstructed_x = []
        reconstructed_y = []
        reconstructed_w = []
        reconstructed_h = []

        generated_bbox = torch.zeros(B, gap - 1, 4).cuda()

        last_bbox = torch.zeros(B, 1, 4).cuda()
        last_bbox[:, 0, 0] = observation[:, -1, 0]
        last_bbox[:, 0, 1] = observation[:, -1, 1]
        last_bbox[:, 0, 2] = observation[:, -1, 2]
        last_bbox[:, 0, 3] = observation[:, -1, 3]

        generated_delta = torch.zeros(B, gap - 1, 4)
        delta_cnt = 0

        # new alternative
        if observation.shape[1] == 1:
            current_obs = observation[:, 0:1, :] * 0.0
            h, featureX_obs, featureY_obs, featureW_obs, featureH_obs = self.single_forward(current_obs, social[:, :observation.shape[1], :], h)
            featureX, featureY, featureW, featureH = featureX_obs[:, -1, :].unsqueeze(1), featureY_obs[:, -1, :].unsqueeze(1), featureW_obs[:, -1, :].unsqueeze(1), featureH_obs[:, -1, :].unsqueeze(1)
        else:
            current_obs = observation[:, 1:, :] - observation[:, :-1, :]
            # current_obs = torch.cat([torch.zeros(current_obs.shape[0], 1, current_obs.shape[2]).cuda(), current_obs], dim=1)
            h, featureX_obs, featureY_obs, featureW_obs, featureH_obs = self.single_forward(current_obs, social[:, :current_obs.shape[1], :], h)
            featureX, featureY, featureW, featureH = featureX_obs[:, -1, :].unsqueeze(1), featureY_obs[:, -1, :].unsqueeze(1), featureW_obs[:, -1, :].unsqueeze(1), featureH_obs[:, -1, :].unsqueeze(1)


        # new alternative
        for t in range(observation.shape[1], observation.shape[1] + gap - 1):
            sampled_x = torch.zeros(B, 1, 4).cuda()
            sampled_x[:, :, 0] = self.sample_cluster_multinomial(featureX, centroid_x)
            sampled_x[:, :, 1] = self.sample_cluster_multinomial(featureY, centroid_y)
            sampled_x[:, :, 2] = self.sample_cluster_multinomial(featureW, centroid_w)
            sampled_x[:, :, 3] = self.sample_cluster_multinomial(featureH, centroid_h)

            generated_delta[:, delta_cnt] = sampled_x[:, 0]

            last_bbox[:, 0, 0] += sampled_x[:, 0, 0]
            last_bbox[:, 0, 1] += sampled_x[:, 0, 1]
            last_bbox[:, 0, 2] += sampled_x[:, 0, 2]
            last_bbox[:, 0, 3] += sampled_x[:, 0, 3]

            generated_bbox[:, delta_cnt] = last_bbox[:, 0]
            delta_cnt += 1

            current_social = social[:, t:t + 1, :]
            h, featureX, featureY, featureW, featureH = self.single_forward(sampled_x, current_social, h)

            featureX = featureX.view(B, 1, -1)
            featureY = featureY.view(B, 1, -1)
            featureW = featureW.view(B, 1, -1)
            featureH = featureH.view(B, 1, -1)

            reconstructed_x.append(featureX)
            reconstructed_y.append(featureY)
            reconstructed_w.append(featureW)
            reconstructed_h.append(featureH)

        sampled_detection = torch.zeros(B, 1, 4).cuda()
        sampled_detection[:, :, 0] = self.sample_cluster_multinomial(featureX, centroid_x)
        sampled_detection[:, :, 1] = self.sample_cluster_multinomial(featureY, centroid_y)
        sampled_detection[:, :, 2] = self.sample_cluster_multinomial(featureW, centroid_w)
        sampled_detection[:, :, 3] = self.sample_cluster_multinomial(featureH, centroid_h)
        # sampled_detection[:, :, 0] = self.sample_cluster(featureX, centroid_x)
        # sampled_detection[:, :, 1] = self.sample_cluster(featureY, centroid_y)
        # sampled_detection[:, :, 2] = self.sample_cluster(featureW, centroid_w)
        # sampled_detection[:, :, 3] = self.sample_cluster(featureH, centroid_h)

        last_detection = torch.zeros(B, 1, 4).cuda()
        last_detection[:, 0, 0] = last_bbox[:, 0, 0] + sampled_detection[:, 0, 0]
        last_detection[:, 0, 1] = last_bbox[:, 0, 1] + sampled_detection[:, 0, 1]
        last_detection[:, 0, 2] = last_bbox[:, 0, 2] + sampled_detection[:, 0, 2]
        last_detection[:, 0, 3] = last_bbox[:, 0, 3] + sampled_detection[:, 0, 3]

        if len(reconstructed_x) > 0:
            reconstructed_x, reconstructed_y, reconstructed_w, reconstructed_h = torch.cat(reconstructed_x, dim=1),\
            torch.cat(reconstructed_y, dim=1), torch.cat(reconstructed_w, dim=1), torch.cat(reconstructed_h, dim=1)

            reconstructed_x, reconstructed_y, reconstructed_w, reconstructed_h = torch.cat([featureX_obs, reconstructed_x], dim=1), \
                                                                                 torch.cat([featureY_obs, reconstructed_y], dim=1), \
                                                                                 torch.cat([featureW_obs, reconstructed_w], dim=1), \
                                                                                 torch.cat([featureH_obs, reconstructed_h], dim=1)
        else:
            reconstructed_x, reconstructed_y, reconstructed_w, reconstructed_h = featureX_obs, featureY_obs, featureW_obs, featureH_obs
        return reconstructed_x, reconstructed_y, reconstructed_w, reconstructed_h, \
               generated_bbox, generated_delta, last_detection