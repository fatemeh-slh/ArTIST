from __future__ import print_function
import numpy as np
from sklearn.cluster import KMeans
from random import shuffle
import torch
import argparse


def prepare_clustering_data(data_path="train_path_mot.npy"):
    dataset = np.load(data_path, allow_pickle=True).item()
    samples_x = []
    samples_y = []
    samples_w = []
    samples_h = []
    for dir in dataset.keys():
        for tracklet_key, tracklet_value in dataset[dir]['tracklets'].items():
            for t in range(len(tracklet_value['sequence']) - 1):
                items1 = np.array(tracklet_value['sequence'][t])
                items2 = np.array(tracklet_value['sequence'][t + 1])

                x1 = float(items1[2]) / int(dataset[dir]['imWidth'])
                y1 = float(items1[3]) / int(dataset[dir]['imHeight'])
                w1 = float(items1[4]) / int(dataset[dir]['imWidth'])
                h1 = float(items1[5]) / int(dataset[dir]['imHeight'])
                x2 = float(items2[2]) / int(dataset[dir]['imWidth'])
                y2 = float(items2[3]) / int(dataset[dir]['imHeight'])
                w2 = float(items2[4]) / int(dataset[dir]['imWidth'])
                h2 = float(items2[5]) / int(dataset[dir]['imHeight'])
                samples_x.append(x2 - x1)
                samples_y.append(y2 - y1)
                samples_w.append(w2 - w1)
                samples_h.append(h2 - h1)
                print(dir, tracklet_key)

    shuffle(samples_x)
    shuffle(samples_y)
    shuffle(samples_w)
    shuffle(samples_h)

    np.save('centroid/samples_x_motpath.npy', np.array(samples_x, dtype='float32'))
    np.save('centroid/samples_y_motpath.npy', np.array(samples_y, dtype='float32'))
    np.save('centroid/samples_w_motpath.npy', np.array(samples_w, dtype='float32'))
    np.save('centroid/samples_h_motpath.npy', np.array(samples_h, dtype='float32'))


def clustering_all(num_cluster=1024, stride=4):

    samples_x = np.load('centroid/samples_x_motpath.npy')
    samples_y = np.load('centroid/samples_y_motpath.npy')
    samples_w = np.load('centroid/samples_w_motpath.npy')
    samples_h = np.load('centroid/samples_h_motpath.npy')

    samples_x = samples_x[0::stride]
    kmeans_x = KMeans(n_clusters=num_cluster, verbose=1)
    kmeans_x.fit(np.expand_dims(samples_x, 1))
    np.save('cluster/centroids_x.npy', kmeans_x.cluster_centers_)

    samples_y = samples_y[0::stride]
    kmeans_y = KMeans(n_clusters=num_cluster, verbose=1)
    kmeans_y.fit(np.expand_dims(samples_y, 1))
    np.save('cluster/centroids_y.npy', kmeans_y.cluster_centers_)

    samples_w = samples_w[0::stride]
    kmeans_w = KMeans(n_clusters=num_cluster, verbose=1)
    kmeans_w.fit(np.expand_dims(samples_w, 1))
    np.save('cluster/centroids_w.npy', kmeans_w.cluster_centers_)
    #
    samples_h = samples_h[0::stride]
    kmeans_h = KMeans(n_clusters=num_cluster, verbose=1)
    kmeans_h.fit(np.expand_dims(samples_h, 1))
    np.save('cluster/centroids_h.npy', kmeans_h.cluster_centers_)


def clustering(num_cluster=1024, stride=4, data_component='x'):
    if data_component == 'x':
        samples = np.load('centroid/samples_x_motpath.npy')
    if data_component == 'y':
        samples = np.load('centroid/samples_y_motpath.npy')
    if data_component == 'w':
        samples = np.load('centroid/samples_w_motpath.npy')
    if data_component == 'h':
        samples = np.load('centroid/samples_h_motpath.npy')

    samples = samples[0::stride]
    kmeans = KMeans(n_clusters=num_cluster, verbose=1)
    kmeans.fit(np.expand_dims(samples, 1))
    np.save('cluster/centroids_' + data_component + '.npy', kmeans.cluster_centers_)


def load_clusters():
    centroid_x = np.load("centroid/centroids_x.npy")
    centroid_x = np.sort(centroid_x, 0)
    centroid_x = torch.from_numpy(centroid_x).transpose(1, 0).cuda()

    centroid_y = np.load("centroid/centroids_y.npy")
    centroid_y = np.sort(centroid_y, 0)
    centroid_y = torch.from_numpy(centroid_y).transpose(1, 0).cuda()

    centroid_w = np.load("centroid/centroids_w.npy")
    centroid_w = np.sort(centroid_w, 0)
    centroid_w = torch.from_numpy(centroid_w).transpose(1, 0).cuda()

    centroid_h = np.load("centroid/centroids_h.npy")
    centroid_h = np.sort(centroid_h, 0)
    centroid_h = torch.from_numpy(centroid_h).transpose(1, 0).cuda()

    return centroid_x, centroid_y, centroid_w, centroid_h


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, default='do_all',
                        choices=['prepare', 'cluster_x', 'cluster_y', 'cluster_w', 'cluster_h', 'cluster_all', 'do_all'])
    parser.add_argument("-d", "--data", type=str, required=True)
    parser.add_argument("-s", "--stride", type=int, default=4)
    parser.add_argument("-c", "--cluster", type=int, default=1024)
    args = parser.parse_args()

    if args.task == "prepare":
        prepare_clustering_data(args.data)
    if args.task == "cluster_x":
        clustering(num_cluster=args.cluster, stride=args.stride, data_component='x')
    if args.task == "cluster_y":
        clustering(num_cluster=args.cluster, stride=args.stride, data_component='y')
    if args.task == "cluster_w":
        clustering(num_cluster=args.cluster, stride=args.stride, data_component='w')
    if args.task == "cluster_h":
        clustering(num_cluster=args.cluster, stride=args.stride, data_component='h')
    if args.task == "cluster_all":
        clustering_all(num_cluster=args.cluster, stride=args.stride)
    if args.task == "do_all":
        prepare_clustering_data(args.data)
        clustering_all(num_cluster=args.cluster, stride=args.stride)