from __future__ import print_function
import numpy as np
import torch
import random


class dataloader:
    def __init__(self, split="train", batch_size=64):
        self.batch_size = batch_size
        if split == "train":
            self.dataset = np.load('data/postp_combined_path_mot_train.npy', allow_pickle=True).item()
        else:
            self.dataset = np.load('data/postp_mot_val.npy', allow_pickle=True).item()

    def generate(self, seq_len=180):
        while True:
            current_batch_motion = np.zeros((self.batch_size, seq_len, 4))
            current_image_wh = torch.zeros((self.batch_size, 2))

            for batch in range(self.batch_size):
                while True:
                    dir = random.choice(list(self.dataset.keys()))
                    selected_seq = random.choice(list(self.dataset[dir]['tracklets'].keys()))
                    len_selected_seq = len(self.dataset[dir]['tracklets'][selected_seq]['sequence'])
                    if len_selected_seq > seq_len + 1:
                        break

                # select a random time
                offset = np.random.randint(0, len_selected_seq - seq_len)

                for t in range(offset, offset + seq_len):
                    items = np.array(self.dataset[dir]['tracklets'][selected_seq]['sequence'][t])
                    current_batch_motion[batch, t - offset] = items[2:6]
                    current_image_wh[batch, 0] = int(self.dataset[dir]['imWidth'])
                current_image_wh[batch, 1] = int(self.dataset[dir]['imHeight'])

            yield torch.from_numpy(current_batch_motion).float(), current_image_wh


if __name__ == '__main__':
    loader = dataloader()
    data, wh = next(iter(loader.generate(20)))
