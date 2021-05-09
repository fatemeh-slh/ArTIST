"""
ArTIST evaluation: Inferring the likelihood of current observations (detections)
author: Fatemeh Saleh
"""

import cv2
from utils.utils_ar import infer_log_likelihood, iou
import numpy as np
import torch
from utils.clustering import clustering, load_clusters
from models.ar import motion_ar
from models.ae import motion_ae

torch.backends.cudnn.enabled = False


def test(model, centroid_x, centroid_y, centroid_w, centroid_h, cnt, num_cluster):
    """[summary]

    Args:
        model (nn.Module): ArTIST model
        centroid_x (list): The centroids of x coordinate of the bounding boxes in the training set
        centroid_y (list): The centroids of y coordinate of the bounding boxes in the training set
        centroid_w (list): The centroids of width of the bounding boxes in the training set
        centroid_h ((list): The centroids of height of the bounding boxes in the training set
        cnt (int): The test sequence index
        num_cluster (int): number of clusters
    """

    test_set = np.load('data/demo_test_subset.npy', allow_pickle=True)
    rand_len = int(test_set[cnt]['seq_len'])

    # we consider the sequence to be observed up until mask_index
    mask_index = int(rand_len * 0.75)
    data = test_set[cnt]['data'][:, :mask_index, :]
    BBOX = []
    image_wh = test_set[cnt]['wh'][0]
    width = image_wh[0]
    height = image_wh[1]
    # social information (see section 3.3 of https://arxiv.org/pdf/2012.02337v1.pdf)
    social = test_set[cnt]['social']
    valid_box = test_set[cnt]['data'][:, mask_index, :]

    # creating some imaginary detections at current time step
    instance_options = [[820/width, 830/height, 80/width, 170/height], [400/width, 310/height, 110/width, 330/height], [1000/width, 300/height, 170/width, 300/height]]
    instance_options.append([valid_box[0, 0], valid_box[0, 1], valid_box[0, 2], valid_box[0, 3]])

    scores, ious = np.zeros(len(instance_options)), np.zeros(len(instance_options))
    gap = 1
    gaussian_kernel = np.load("data/kernel.npy")
    gaussian_kernel = torch.autograd.Variable(torch.from_numpy(gaussian_kernel).float()).cuda()
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)

    self_data = data.cuda()
    self_data = self_data.cuda().float()

    # computing the motion velocity
    self_delta_tmp = self_data[:, 1:, :] - self_data[:, 0:-1, :]
    self_delta = torch.zeros(self_delta_tmp.shape[0], self_delta_tmp.shape[1] + 1,
                             self_delta_tmp.shape[2]).cuda()
    self_delta[:, 1:, :] = self_delta_tmp

    # computing the distribution over the next plausible bounding box
    dist_x, dist_y, dist_w, dist_h, sampled_boxes, sampled_deltas, sampled_detection = model.inference(
        self_data,
        social[:, :mask_index, :],
        gap,
        centroid_x,
        centroid_y,
        centroid_w,
        centroid_h)

    # making it a probability distribution
    dist_x = torch.nn.Softmax(dim=-1)(dist_x)
    dist_y = torch.nn.Softmax(dim=-1)(dist_y)
    dist_w = torch.nn.Softmax(dim=-1)(dist_w)
    dist_h = torch.nn.Softmax(dim=-1)(dist_h)

    # smoothing the distributions using a gaussian kernel
    y_g1d_x = torch.nn.functional.conv1d(dist_x, gaussian_kernel.repeat(dist_x.shape[1], dist_x.shape[1], 1),
                                         padding=24)
    y_g1d_y = torch.nn.functional.conv1d(dist_y, gaussian_kernel.repeat(dist_y.shape[1], dist_y.shape[1], 1),
                                         padding=24)
    y_g1d_w = torch.nn.functional.conv1d(dist_w, gaussian_kernel.repeat(dist_w.shape[1], dist_w.shape[1], 1),
                                         padding=24)
    y_g1d_h = torch.nn.functional.conv1d(dist_h, gaussian_kernel.repeat(dist_h.shape[1], dist_h.shape[1], 1),
                                         padding=24)

    extended_track = torch.zeros(1, self_delta.shape[1] + len(sampled_boxes) + 1, 4).cuda()
    extended_track[0, :self_delta.shape[1], :] = self_delta[0, :, :]
    extended_track[0, self_delta.shape[1]:-1, :] = sampled_deltas[0, :-1, :]

    observation_last = [self_data[0, -1, 0].item() * width, self_data[0, -1, 1].item() * height,
                            (self_data[0, -1, 0].item() * width) + (self_data[0, -1, 2].item() * width),
                            (self_data[0, -1, 1].item() * height) + (self_data[0, -1, 3].item() * height)]

    # loop over the detections...
    for opt_idx, option in enumerate(instance_options):
        opt = [(option[0]), (option[1]), (option[2]), (option[3])]

        option_unnorm = [
            option[0] * width,
            option[1] * height,
            (option[2] + option[0]) * width,
            (option[3] + option[1]) * height
            ]
        iou_validate = iou(option_unnorm, observation_last)
        ious[opt_idx] = iou_validate

        last_delta = torch.from_numpy(np.array(opt) - self_data[0, -1].cpu().detach().numpy()).cuda()

        extended_track[0, -1, :] = last_delta

        # inferring the likelihood of each detection (considered as the last observation of the sequence)
        likelihoods_smooth = infer_log_likelihood(y_g1d_x, y_g1d_y, y_g1d_w, y_g1d_h,
                                                    extended_track[:, 1:, 0:1],
                                                    extended_track[:, 1:, 1:2],
                                                    extended_track[:, 1:, 2:3],
                                                    extended_track[:, 1:, 3:4],
                                                    centroid_x, centroid_y, centroid_w, centroid_h, num_cluster)
        all_scores = np.array(likelihoods_smooth[-1])
        likelihoods_smooth = np.sum(all_scores)
        score = likelihoods_smooth
        scores[opt_idx] = score

    # visualization
    I1 = np.ones((int(image_wh[1].item()), int(image_wh[0].item()), 3))
    I1 = I1 * 255

    for i in range(0, mask_index):
        I1 = cv2.rectangle(I1, (
                int(data[0, i, 0].item() * image_wh[0]), int(data[0, i, 1].item() * image_wh[1])),
                                (int(data[0, i, 0].item() * image_wh[0] + data[0, i, 2].item() * image_wh[0]),
                                int(data[0, i, 1].item() * image_wh[1] + data[0, i, 3].item() * image_wh[1])),
                                (255, 0, 0), 1)  #### GT

    for i in range(len(instance_options)):
        I1 = cv2.rectangle(I1, (
            int(instance_options[i][0]*width), int(instance_options[i][1]*height),
                            int(instance_options[i][2]*width), int(instance_options[i][3]*height)), (0, 0, 0), 2)
        cv2.putText(I1, 'log p: ' + str(round(scores[i], 2)), (int(instance_options[i][0]*width), int(instance_options[i][1]*height)-40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0),
                    1, lineType=cv2.LINE_AA)
        cv2.putText(I1, 'IoU w/ last bbox: ' + str(round(ious[i], 2)), (int(instance_options[i][0]*width), int(instance_options[i][1]*height)-15), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0),
                1 , lineType=cv2.LINE_AA)
    cv2.imwrite('temp/ar/log_p/likelihood.jpg', I1)



if __name__ == '__main__':
    model_ae = motion_ae(256).cuda()
    model_ae.load_state_dict(torch.load('checkpoint/ae/ae_8.pth'))
    model_ae.eval()
    model_ar = motion_ar(512, 1024).cuda()
    model_ar.load_state_dict(torch.load('checkpoint/ar/ar_110.pth'))
    model_ar.eval()
    centroid_x, centroid_y, centroid_w, centroid_h = load_clusters()
    test(model_ar, centroid_x, centroid_y, centroid_w, centroid_h, 100, 1024)