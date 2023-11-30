"""
XView2 Metric from Generated Masks

Usage: python xview2_metric.py <root_folder_path>

Arguments: (Required) root_folder_path: root folder where the folders for images, labels, target masks and 
                                        generated ouput from the model is.
            
Note:      Dataset should be organized like this before running the script:
            (root folder)
            |
            | ----->   images/              ## contains the pre-disaster and post-disaster images.
            | ----->   labels/              ## contains the labels in json.
            | ----->   targets/             ## contains the pre-disaster building localization masks
            |                                   post-disaster localization + damage segmentation masks.
            | ----->   generated-output/    ## contains the generated masks from LISA.

Returns:
    Prints the xview2 metric. (Currently only localization as haven't generated damage classification results yet.)
"""

import cv2
import sys
import numpy as np
import glob

from tqdm import tqdm


def localization_f1_score(gt_mask, pred_mask):
    assert(gt_mask.shape == pred_mask.shape)
    for mask in [gt_mask, pred_mask]:
        assert(np.unique(mask).shape[0] == 1 or np.unique(mask).shape[0] == 2)
        assert(np.unique(mask)[0] == 0)
        if np.unique(mask).shape[0] == 2:
            assert(np.unique(mask)[1] == 1)

    gt_mask_f, pred_mask_f = gt_mask.flatten(), pred_mask.flatten()

    tp = np.sum(np.logical_and(gt_mask_f == 1, pred_mask_f == 1))
    fp = np.sum(np.logical_and(gt_mask_f == 0, pred_mask_f == 1))
    tn = np.sum(np.logical_and(gt_mask_f == 0, pred_mask_f == 0))
    fn = np.sum(np.logical_and(gt_mask_f == 1, pred_mask_f == 0))

    # precision = tp/(tp+fp)
    # recall = tp/(tp+fn)

    # f1_2 = (2*precision*recall)/(precision + recall)
    f1 = (2*tp)/(2*tp + fp + fn)
    return f1


def localization_f1_per_img(in_filenames, idx):
    # Getting filenames
    gt_localization_mask_fn = targets_root + in_filenames[idx] + "_pre_disaster_target.png"
    pred_localization_mask_fn = output_root + in_filenames[idx] + "_pre_disaster_mask_0.jpg"
    
    gt_mask = cv2.imread(gt_localization_mask_fn)
    pred_localization_mask = cv2.imread(pred_localization_mask_fn)
    pred_localization_mask_b = pred_localization_mask.copy()
    pred_localization_mask_b[pred_localization_mask > 0] = 1
    return localization_f1_score(gt_mask, pred_localization_mask_b)


def xview2_metric_complete_dataset(in_filenames):
    print("Calculating Localization F-1: ")
    localization_f1_arr = [localization_f1_per_img(in_filenames, i) for i in tqdm(range(len(in_filenames)))]
    print("Average building localization F1 predicted by LISA: ", np.sum(localization_f1_arr)/len(localization_f1_arr))


def visualization_per_idx(in_filenames, idx):
    # Getting filenames
    pre_disaster_fn = images_root + in_filenames[idx] + "_pre_disaster.png"
    gt_localization_mask_fn = targets_root + in_filenames[idx] + "_pre_disaster_target.png"
    pred_localization_mask_fn = output_root + in_filenames[idx] + "_pre_disaster_mask_0.jpg"
    print("Pre disaster image path: ", pre_disaster_fn)
    print("GT Mask path: ", gt_localization_mask_fn)
    print("Pred Mask path: ", pred_localization_mask_fn)

    # Retrieving the images
    in_img = cv2.imread(pre_disaster_fn)
    pred_localization_mask = cv2.imread(pred_localization_mask_fn)
    gt_mask = cv2.imread(gt_localization_mask_fn)
    # gt_img_norm = gt_mask.copy()

    # Normalizing the mask from [0,1,2,3,...,n] to [0,255] where n is number of classes
    # For localization n = 1 in xBD and damage assessment n = 4.
    # uniques = np.unique(gt_img)
    # for i, u in enumerate(uniques):
    #     color = int((255/len(uniques)) * (i-1))
    #     gt_img_norm[gt_img == u] = color

    # Binarizing the localization map
    # Getting the range of values that the model predicted, and converting them to binary
    # where 0 = no building, 129 = building.
    print("pred_localization_mask uniques", np.unique(pred_localization_mask))
    pred_localization_mask_b = pred_localization_mask.copy()
    pred_localization_mask_b[pred_localization_mask > 0] = 1

    # Visualizing all of the maps
    cv2.imshow("pre_disaster = " + pre_disaster_fn, in_img)
    cv2.waitKey(0)
    cv2.imshow("target = " + gt_localization_mask_fn, gt_mask * 129)
    cv2.waitKey(0)
    cv2.imshow("pred_localization_mask 2 = " + pred_localization_mask_fn, pred_localization_mask_b * 129)
    cv2.waitKey(0)
    # print(f1_per_img(in_filenames, idx))


if __name__ == '__main__':
    dataset_root = sys.argv[1]
    images_root = dataset_root + "images/"
    targets_root = dataset_root + "targets/"
    output_root = dataset_root + "generated_output/"
    # print(dataset_root, images_root, targets_root)

    in_filenames = glob.glob(images_root + "*.png")
    in_filenames = [fn[len(images_root):] for fn in in_filenames]
    in_filenames = list(set([(fn.split("_")[0] + "_" + fn.split("_")[1]) for fn in in_filenames]))

    xview2_metric_complete_dataset(in_filenames)