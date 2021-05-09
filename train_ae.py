from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from .dataloader_ae import dataloader
from models.ae import motion_ae


torch.backends.cudnn.benchmark = True


def train(model):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    tf = 1
    gradient_clip = 0.1
    max_iter = 80001
    model.train()

    for batch_idx in range(max_iter):
        seq_len = np.random.randint(3, 100)
        data, wh = next(iter(train_loader.generate(seq_len)))

        data = data.cuda()
        data_vel = data[:, 1:, :] - data[:, :-1, :]
        optimizer.zero_grad()

        reconstructed = model(data_vel, tf)
        loss = F.mse_loss(reconstructed, data_vel, reduction="sum")
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        if batch_idx % 50 == 0:
            tf *= 0.99
            model.eval()

            with torch.no_grad():
                test_sample, wh = next(iter(val_loader.generate(100)))
                test_sample = test_sample.cuda()
                test_sample_vel = test_sample[:, 1:, :] - test_sample[:, :-1, :]
                reconstructed = model(test_sample_vel, 0)

                np.save("temp/ae/train/reconstructed.npy", reconstructed.cpu().detach().numpy()[0])
                np.save("temp/ae/train/original.npy", test_sample.cpu().detach().numpy()[0])
                np.save("temp/ae/train/wh.npy", wh.numpy()[0])

                print("[", batch_idx, "]\tLoss: ", round(loss.item() / seq_len, 4), "\ttf: ", round(tf, 4))
            model.train()
        if batch_idx % 10000 == 0:
            torch.save(model.state_dict(), "checkpoint/ae/ae_" + str(batch_idx//10000) + ".pth")


def evaluate(model):
    with torch.no_grad():
        test_sample, wh = next(iter(val_loader.generate(100)))
        test_sample = test_sample.cuda()
        test_sample_vel = test_sample[:, 1:, :] - test_sample[:, :-1, :]
        reconstructed = model(test_sample_vel, 0)

        np.save("temp/ae/eval_reconstructed.npy", reconstructed.cpu().detach().numpy()[0])
        np.save("temp/ae/eval_original.npy", test_sample.cpu().detach().numpy()[0])
        np.save("temp/ae/eval_wh.npy", wh.numpy()[0])


if __name__ == "__main__":

    train_loader = dataloader(split="train", batch_size=512)
    val_loader = dataloader(split="val", batch_size=1)

    model = motion_ae(256).cuda()
    train(model)

    #model.load_state_dict(torch.load("checkpoints/ae_weights/ae_combined_path_mot_8.pth"))
    #model.eval()
    #evaluate(model)

