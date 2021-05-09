from __future__ import print_function
import numpy as np
import torch
import random


def find_frame_idx(frame_num, sequences):
    for i in range(len(sequences['sequence'])):
        if int(sequences['sequence'][i][0]) == frame_num:
            return i
    return -1


class dataloader:
    def __init__(self, model_ae, split="train", batch_size=64):
        self.batch_size = batch_size
        if split == "train":
            self.dataset = np.load('data/postp_combined_path_mot_train.npy', allow_pickle=True).item()
        else:
            self.dataset = np.load('data/postp_mot_val.npy', allow_pickle=True).item()
        self.model_ae = model_ae

    def generate(self, seq_len):
        while True:
            current_batch_motion = np.zeros((self.batch_size, seq_len, 4))
            current_image_wh = torch.zeros((self.batch_size, 2))
            social_list_h = []
            for batch in range(self.batch_size):

                while True:
                    dir = random.choice(list(self.dataset.keys()))
                    selected_seq = random.choice(list(self.dataset[dir]['tracklets'].keys()))
                    len_selected_seq = len(self.dataset[dir]['tracklets'][selected_seq]['sequence'])
                    if len_selected_seq > seq_len + 1:
                        break

                # select a random time
                offset = np.random.randint(0, len_selected_seq - seq_len)
                sequence = self.dataset[dir]['tracklets'][selected_seq]['sequence']
                current_batch_motion[batch, :] = sequence[offset: offset + seq_len]

                current_image_wh[batch, 0] = int(self.dataset[dir]['imWidth'])
                current_image_wh[batch, 1] = int(self.dataset[dir]['imHeight'])

                social_ids = self.dataset[dir]['tracklets'][selected_seq]['social_ids']
                social_ids = np.unique(social_ids)
                start_frame = int(self.dataset[dir]['tracklets'][selected_seq]['start']) + offset
                end_frame = int(self.dataset[dir]['tracklets'][selected_seq]['start']) + offset + seq_len
                social_dict = {}
                for sid in social_ids:
                    start_s = self.dataset[dir]['tracklets'][sid]['start']
                    end_s = self.dataset[dir]['tracklets'][sid]['end']
                    SEQ = np.zeros((seq_len, 4))
                    START, END = -1, -1

                    if start_s == start_frame and end_frame == end_s:
                        SEQ = self.dataset[dir]['tracklets'][sid]['sequence'][0:-1]
                        START = start_s
                        END = end_s
                    elif start_frame < end_s <= end_frame:
                        if start_s >= start_frame:
                            SEQ = self.dataset[dir]['tracklets'][sid]['sequence'][0:-1]
                            START = start_s
                            END = end_s
                        else:
                            idx = start_frame - start_s
                            SEQ = self.dataset[dir]['tracklets'][sid]['sequence'][idx:-1]
                            START = start_frame
                            END = end_s
                    elif end_s > start_frame and start_s < end_frame:
                        if start_s >= start_frame:
                            idx_end = end_frame - start_s
                            SEQ = self.dataset[dir]['tracklets'][sid]['sequence'][:idx_end]
                            START = start_s
                            END = end_frame
                        elif start_s < start_frame:
                            idx_end = end_frame - start_s
                            idx_start = start_frame - start_s
                            SEQ = self.dataset[dir]['tracklets'][sid]['sequence'][idx_start:idx_end]
                            START = start_frame
                            END = end_frame
                    if START != -1 and END != -1 and START != END:
                        social_dict[sid] = {'seq': SEQ, 'start': START, 'end': END}

                current_batch_social = torch.zeros(len(social_dict.keys()), seq_len, 4)
                idx = 0
                for key, value in social_dict.items():
                    start_social = int(value['start'])
                    end_social = int(value['end'])
                    if start_social > end_social:
                        pass
                    else:

                        if start_social == start_frame and end_social == end_frame:
                            current_batch_social[idx] = torch.from_numpy(value['seq'])

                        if start_social > start_frame and end_social < end_frame:
                            current_batch_social[idx, :(start_social - start_frame)] = torch.from_numpy(value['seq'][0, :])
                            current_batch_social[idx, (start_social - start_frame):-(end_frame - end_social)] = torch.from_numpy(value['seq'][:, :])
                            current_batch_social[idx, -(end_frame - end_social):] = torch.from_numpy(value['seq'][-1, :])
                        elif end_social < end_frame and start_social == start_frame:
                            current_batch_social[idx, -(end_frame - end_social):] = torch.from_numpy(value['seq'][-1, :])
                            current_batch_social[idx, :-(end_frame - end_social)] = torch.from_numpy(value['seq'][:, :])
                        elif end_social == end_frame and start_social > start_frame:
                            current_batch_social[idx, :(start_social - start_frame)] = torch.from_numpy(value['seq'][0, :])
                            current_batch_social[idx, (start_social - start_frame):] = torch.from_numpy(value['seq'][:, :])
                        else:
                            current_batch_social[idx] = torch.from_numpy(value['seq'])

                    idx += 1
                social_vel = torch.zeros(current_batch_social.shape)
                social_vel[:, 1:, :] = current_batch_social[:, 1:, :] - current_batch_social[:, :-1, :]
                with torch.no_grad():
                     social_vel = social_vel.float().cuda()
                     try:
                         h = self.model_ae.inference(social_vel)
                         h = torch.max(h, dim=0)[0].unsqueeze(0)
                     except:
                         h = torch.zeros(1, seq_len, 256).float().cuda()
                social_list_h.append(h)
                social_rep = torch.cat(social_list_h, dim=0)
            yield torch.from_numpy(current_batch_motion).float(), current_image_wh, social_rep.float()
