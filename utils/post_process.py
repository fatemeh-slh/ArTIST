import numpy as np


def postprocess(data='data/train_mot17.npy', output_file="data/postp_train_mot17.npy"):
    dataset = np.load(data, allow_pickle=True).item()

    for key, value in dataset.items():
        for keyt, valuet in value['tracklets'].items():
            seq = valuet['sequence']
            new_seq = np.zeros((len(seq), 4))
            for idx, item in enumerate(seq):
                new_seq[idx, 0] = float(seq[idx][2]) / int(value['imWidth'])
                new_seq[idx, 1] = float(seq[idx][3]) / int(value['imHeight'])
                new_seq[idx, 2] = float(seq[idx][4]) / int(value['imWidth'])
                new_seq[idx, 3] = float(seq[idx][5]) / int(value['imHeight'])
            valuet['sequence'] = new_seq

    np.save(output_file, dataset)