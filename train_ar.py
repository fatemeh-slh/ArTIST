from torch import nn, optim
from utils.utils_ar import *
from utils.dataloader_ar import dataloader as data_loader
from utils.clustering import load_clusters
from models.ar import motion_ar
from models.ae import motion_ae

torch.backends.cudnn.enabled = False


def jitter_seq(seq, jitter):
    new_seq = seq.clone()
    for t in range(new_seq.shape[1]):
        new_seq[:, t, 0] += np.random.randint(-1 * jitter, jitter)
        new_seq[:, t, 1] += np.random.randint(-1 * jitter, jitter)
        new_seq[:, t, 2] += np.random.randint(-1 * jitter, jitter)
        new_seq[:, t, 3] += np.random.randint(-1 * jitter, jitter)

    return new_seq


def train(model, model_ae, centroid_x, centroid_y, centroid_w, centroid_h, num_cluster):

    train_loader = data_loader(model_ae, split="train", batch_size=256)
    model.train()
    model_optimizer = optim.Adam(model.parameters(), lr=1e-3)

    video_len = 100
    gradient_clip = 0.1
    MAX_ITER = 200000
    criterion = nn.NLLLoss().cuda()

    for batch_idx in range(MAX_ITER):

        data, image_wh, social = next(train_loader.generate(video_len))
        self_data = data.cuda()
        image_wh = image_wh.cuda()
        social = social.cuda()

        if_jitter = np.random.randint(1, 100) > 50

        if if_jitter:
            self_data = jitter_seq(self_data, 10)

        # computing delta
        self_delta_tmp = self_data[:, 1:, :] - self_data[:, 0:-1, :]
        self_delta = torch.zeros(self_delta_tmp.shape[0], self_delta_tmp.shape[1] + 1, self_delta_tmp.shape[2]).cuda()
        self_delta[:, 1:, :] = self_delta_tmp

        rand_len = np.random.randint(5, video_len - 1)

        if_mask = np.random.randint(1, 10) < 8
        mask_index = rand_len + 1
        if if_mask:
            mask_index = int(rand_len * 0.7)


        # 2. Train model
        model_optimizer.zero_grad()

        sampled_deltas, dist_x, dist_y, dist_w, dist_h = model(self_delta[:, :rand_len, :], social[:, :rand_len, :], mask_index, centroid_x,
                                               centroid_y, centroid_w, centroid_h)

        nll_loss = 0
        dist_x = nn.LogSoftmax(dim=-1)(dist_x)
        dist_y = nn.LogSoftmax(dim=-1)(dist_y)
        dist_w = nn.LogSoftmax(dim=-1)(dist_w)
        dist_h = nn.LogSoftmax(dim=-1)(dist_h)
        for i in range(1, rand_len):
            GT_delta = self_data[:, i: i + 1, :] - self_data[:, i - 1: i, :]
            GT_delta_norm = torch.zeros(GT_delta.shape).cuda()
            GT_delta_norm[:, :, 0] = GT_delta[:, :, 0]
            GT_delta_norm[:, :, 1] = GT_delta[:, :, 1]
            GT_delta_norm[:, :, 2] = GT_delta[:, :, 2]
            GT_delta_norm[:, :, 3] = GT_delta[:, :, 3]
            delta_x = GT_delta_norm[:, :, 0:1]
            delta_y = GT_delta_norm[:, :, 1:2]
            delta_w = GT_delta_norm[:, :, 2:3]
            delta_h = GT_delta_norm[:, :, 3:4]
            min_idx_x, min_idx_y, min_idx_w, min_idx_h = quantize_cluster(delta_x, delta_y, delta_w, delta_h,
                                                                          centroid_x, centroid_y, centroid_w, centroid_h, num_cluster)

            nll_loss += criterion(dist_x[:, i - 1:i, :].view(dist_x.shape[0], -1), min_idx_x.view(-1).long())
            nll_loss += criterion(dist_y[:, i - 1:i, :].view(dist_y.shape[0], -1), min_idx_y.view(-1).long())
            nll_loss += criterion(dist_w[:, i - 1:i, :].view(dist_w.shape[0], -1), min_idx_w.view(-1).long())
            nll_loss += criterion(dist_h[:, i - 1:i, :].view(dist_h.shape[0], -1), min_idx_h.view(-1).long())

        loss = nll_loss #+ euc_dist_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        model_optimizer.step()

        if batch_idx % 250 == 0:
            for param_group in model_optimizer.param_groups:
                param_group['lr'] *= 0.999

        print(batch_idx,
                  "\t nll:", round(nll_loss.item()/rand_len, 2))

        if batch_idx % 1000 == 0:
            torch.save(model.state_dict(), "checkpoint/ar/ar_" + str(batch_idx // 1000) + ".pth")


if __name__ == '__main__':
    centroid_x, centroid_y, centroid_w, centroid_h = load_clusters()
    num_cluster = 1024
    model_ae = motion_ae(256).cuda()
    model_ae.load_state_dict(torch.load("checkpoint/ae/ae_8.pth"))
    model_ae.eval()
    model_social = motion_ar(num_clusters=num_cluster).cuda()
    train(model_social, motion_ae, centroid_x, centroid_y, centroid_w, centroid_h, num_cluster)
