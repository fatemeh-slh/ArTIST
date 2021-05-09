import numpy as np
import glob
import configparser
import argparse


def prepare_data(data_path='data/MOT17Labels', split='train'):

    data_dict = {}
    dirs = glob.glob(data_path + split + '/*')
    dirs.sort()

    for dir in dirs:
        max_id = 500000
        tracklets = {}
        if 'FRCNN' in dir:
            config = configparser.ConfigParser()
            config.read(dir + '/seqinfo.ini')
            seq = config['Sequence']['name']
            if seq not in data_dict:
                data_dict[seq] = {}
                data_dict[seq]['frameRate'] = config['Sequence']['frameRate']
                data_dict[seq]['imWidth'] = config['Sequence']['imWidth']
                data_dict[seq]['imHeight'] = config['Sequence']['imHeight']
                data_dict[seq]['tracklets'] = {}
            # change to gt_val_half if split = val
            # change to gt_train_half if split = train and you would like to keep half of the train for validation
            gt_file = dir + '/gt/gt.txt'
            fid = open(gt_file)
            lines = str.split(fid.read(), '\n')[:-1]
            for line in lines:
                items = str.split(line, ',')
                current_id = int(items[1])
                if int(items[6]) != 1 or float(items[-1]) < 0.25:
                    continue
                else:
                    if int(items[1]) not in tracklets:
                        tracklets[int(items[1])] = {}
                        tracklets[int(items[1])]['sequence'] = []
                        tracklets[int(items[1])]['social_ids'] = []
                        tracklets[int(items[1])]['start'] = int(items[0])
                        tracklets[int(items[1])]['end'] = int(items[0])
                        current_id = int(items[1])
                    if int(items[0]) - tracklets[current_id]['end'] > 1:
                        current_id = max_id + 1
                        max_id += 1
                        if current_id not in tracklets:
                            tracklets[current_id] = {}
                            tracklets[current_id]['sequence'] = []
                            tracklets[current_id]['social_ids'] = []
                            tracklets[current_id]['start'] = int(items[0])
                            tracklets[current_id]['end'] = int(items[0])
                    else:
                        tracklets[current_id]['end'] = int(items[0])
                    tracklets[current_id]['sequence'].append(items)

            data_dict[seq]['tracklets'] = tracklets


    for key, value in data_dict.items():
        for key_t, value_t in value['tracklets'].items():
            id = key_t
            start = value_t['start']
            end = value_t['end']
            for key_s, value_s in value['tracklets'].items():
                if key_s != key_t:
                    if start<=start<=value_s['end'] <= end:
                        value_t['social_ids'].append(key_s)
                    if start<=value_s['start'] <= end:
                        value_t['social_ids'].append(key_s)
                    if value_s['start'] < start and value_s['end'] > end:
                        value_t['social_ids'].append(key_s)

    if args.dataset == "mot":
        np.save('data/train_mot17.npy', data_dict)
    if args.dataset == "pathtrack":
        if split == "train":
            np.save('data/path_track_train.npy', data_dict)
        else:
            np.save('data/path_track_test.npy', data_dict)


def merge_datasets():

    dicts = []
    mot_train = np.load('data/train_mot17.npy', allow_pickle=True).item()
    path_train = np.load('data/path_track_train.npy', allow_pickle=True).item()
    path_test = np.load('data/path_track_test.npy', allow_pickle=True).item()

    dicts.append(mot_train)
    dicts.append(path_train)
    dicts.append(path_test)

    super_dict = {}
    for d in dicts:
        for k, v in d.items():
            super_dict[k] = v

    np.save('data/train_path_mot.npy', super_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, default='prepare', choices=['prepare', 'merge'])
    parser.add_argument("-d", "--dataset", type=str, choices=['mot', 'pathtrack'], default='mot')
    parser.add_argument("-s", "--split", type=str, default='train')
    args = parser.parse_args()

    if args.task == "prepare":
        if args.dataset == 'mot':
            data_path = 'data/MOT17Labels'
            prepare_data(data_path=data_path, split=args.split)
        elif args.dataset == 'pathtrack':
            data_path = 'data/pathtrack_release_v1.0/pathtrack_release/'
            prepare_data(data_path=data_path, split=args.split)

    if args.task == "merge":
        merge_datasets()