from utils.dataloader_ar import dataloader
from models import motion_ae
import torch
import numpy as np

if __name__ == "__main__":

    model_ae = motion_ae(256).cuda()
    model_ae.load_state_dict(torch.load("checkpoint/ae/ae_8.pth"))
    model_ae.eval()
    val_loader = dataloader(model_ae, split="val", batch_size=1)
    test_set = []
    for i in range(1000):
        seq_len = np.random.randint(5, 100)
        obs_len = np.random.randint(1, seq_len - 1)
        data, wh, social = next(iter(val_loader.generate(seq_len)))
        test_set.append({'data': data, 'wh': wh, 'social': social, 'seq_len': seq_len, 'obs_len': obs_len})

    np.save('demo_test_subset.npy', test_set)
