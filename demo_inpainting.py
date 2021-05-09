"""
ArTIST evaluation: Inpainting the missing observation/detections
author: Fatemeh Saleh
"""
import cv2
from utils.utils_ar import iou
import numpy as np
import torch
from utils.clustering import load_clusters
from models.ar import motion_ar
from models.ae import motion_ae

torch.backends.cudnn.enabled = False

def test(model, centroid_x, centroid_y, centroid_w, centroid_h, cnt, n_sampling):
    """ This function generates multiple plausible continuations of an observed sequence

    Args:
        model (nn.Module): ArTIST model
        centroid_x (list): The centroids of x coordinate of the bounding boxes in the training set
        centroid_y (list): The centroids of y coordinate of the bounding boxes in the training set
        centroid_w (list): The centroids of width of the bounding boxes in the training set
        centroid_h ((list): The centroids of height of the bounding boxes in the training set
        cnt (int): The test sequence index
        n_sampling (int): The preferred number of inpaintings
    """

    test_set = np.load('data/demo_test_subset.npy', allow_pickle=True)
    rand_len = int(test_set[cnt]['seq_len'])

    # we consider 75% of the sequence as observation, and aim to generate the rest 25%
    mask_index = int(rand_len * 0.75)
    data = test_set[cnt]['data'][:, :, :]
    # the observed sequence
    masked_data = test_set[cnt]['data'][:, :mask_index, :]
    image_wh = test_set[cnt]['wh'][0]
    width = image_wh[0]
    height = image_wh[1]
    # social information (see section 3.3 of https://arxiv.org/pdf/2012.02337v1.pdf)
    social = test_set[cnt]['social']
    gap = rand_len - mask_index

    self_data = masked_data.cuda()
    self_data = self_data.cuda().float()

    # computing the motion velocity
    self_delta_tmp = self_data[:, 1:, :] - self_data[:, 0:-1, :]
    self_delta = torch.zeros(self_delta_tmp.shape[0], self_delta_tmp.shape[1] + 1,
                             self_delta_tmp.shape[2]).cuda()
    self_delta[:, 1:, :] = self_delta_tmp

    inpainted_boxes = []

    # generating multiple plausible continuations via the batch_inference function
    # To generate n_sampling sequences, this function considers the data as a batch of size n_sampling 
    with torch.no_grad():
        dist_x, dist_y, dist_w, dist_h, sampled_boxes, sampled_deltas, sampled_detection = model.batch_inference(
            self_data.repeat(n_sampling, 1, 1), social.repeat(n_sampling, 1, 1), gap, centroid_x, centroid_y, centroid_w, centroid_h)

    # visualization 
    for n in range(n_sampling):
        I1 = np.ones((int(image_wh[1].item()), int(image_wh[0].item()), 3))
        I1 = I1 * 255

        # visualizing the observed groundtruth sequence (Color: Blue)
        for i in range(0, mask_index):
            I1 = cv2.rectangle(I1, (int(data[0, i, 0].item() * image_wh[0]), int(data[0, i, 1].item() * image_wh[1])),
                            (int(data[0, i, 0].item() * image_wh[0] + data[0, i, 2].item() * image_wh[0]),
                                int(data[0, i, 1].item() * image_wh[1] + data[0, i, 3].item() * image_wh[1])),
                            (255, 0, 0), 1)

        sequence_iou = []
        # visualizing the future sequence (groundtruth and inpainted)
        for i in range(gap - 1):

            # groundtruth future sequence (Color: Red)
            gt_box = [
                int(data[0, mask_index + i, 0].item() * image_wh[0]),
                int(data[0, mask_index + i, 1].item() * image_wh[1]),
                int(data[0, mask_index + i, 0].item() * image_wh[0] + data[0, mask_index + i, 2].item() * image_wh[0]),
                int(data[0, mask_index + i, 1].item() * image_wh[1] + data[0, mask_index + i, 3].item() * image_wh[1])]
            
            # inpainted future sequence (Color: Green)
            inpainted_box = [
                int(sampled_boxes[n, i, 0].item() * image_wh[0]),
                int(sampled_boxes[n, i, 1].item() * image_wh[1]),
                int(sampled_boxes[n, i, 0].item() * image_wh[0] + sampled_boxes[n, i, 2].item() * image_wh[0]),
                int(sampled_boxes[n, i, 1].item() * image_wh[1] + sampled_boxes[n, i, 3].item() * image_wh[1])]

            sequence_iou.append(iou(gt_box, inpainted_box))
            I1 = cv2.rectangle(I1, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 0, 255), 1)
            I1 = cv2.rectangle(I1, (inpainted_box[0], inpainted_box[1]), (inpainted_box[2], inpainted_box[3]), (0, 255, 0), 1)

        # additional information: last time IoU and future sequence mean IoU
        cv2.putText(I1, 'Last time IoU: ' + str(round(sequence_iou[-1], 2)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1 , lineType=cv2.LINE_AA)
        cv2.putText(I1, 'Future sequence mIoU: ' + str(round(np.mean(sequence_iou), 2)), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1 , lineType=cv2.LINE_AA)
        cv2.imwrite('temp/ar/inpainting/inpainting_' + str(n).zfill(2) + '.jpg', I1)


if __name__ == '__main__':
    model_ae = motion_ae(256).cuda()
    model_ae.load_state_dict(torch.load('checkpoint/ae/ae_8.pth'))
    model_ae.eval()
    model_ar = motion_ar(512, 1024).cuda()
    model_ar.load_state_dict(torch.load('checkpoint/ar/ar_110.pth'))
    model_ar.eval()
    centroid_x, centroid_y, centroid_w, centroid_h = load_clusters()
    test(model_ar, centroid_x, centroid_y, centroid_w, centroid_h, 100, 40)