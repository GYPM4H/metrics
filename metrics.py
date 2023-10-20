import numpy as np
from scipy.optimize import linear_sum_assignment

# Function to calculate IoU
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Function to build the cost matrix
def build_cost_matrix(predicted_masks, ground_truth_masks):
    num_preds = len(predicted_masks)
    num_gts = len(ground_truth_masks)
    max_items = max(num_preds, num_gts)
    cost_matrix = np.ones((max_items, max_items))

    for i in range(num_preds):
        for j in range(num_gts):
            iou = calculate_iou(predicted_masks[i], ground_truth_masks[j])
            cost_matrix[i, j] = 1 - iou  # converting IoU to cost
    return cost_matrix

# Main function to calculate mAP and mAR
def calculate_map_mar(predicted_masks, ground_truth_masks, iou_thresholds=[0.5]):
    cost_matrix = build_cost_matrix(predicted_masks, ground_truth_masks)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        iou = 1 - cost_matrix[r, c]
        if r < len(predicted_masks) and c < len(ground_truth_masks) and iou > 0:  # valid match
            matches.append((r, c, iou))

    aps = []  # list to store average precisions
    ars = []  # list to store average recalls

    for iou_threshold in iou_thresholds:
        true_positives = 0
        matched_ground_truths = set()

        for match in matches:
            if match[2] >= iou_threshold:
                true_positives += 1
                matched_ground_truths.add(match[1])

        false_positives = len(predicted_masks) - true_positives
        false_negatives = len(ground_truth_masks) - len(matched_ground_truths)

        # Precision: TP / (TP + FP)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        # Recall: TP / (TP + FN)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        aps.append(precision)
        ars.append(recall)

    mAP = np.mean(aps)
    mAR = np.mean(ars)

    return mAP, mAR






