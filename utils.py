import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

def plot_masks(masks, title):
    fig, axs = plt.subplots(1, len(masks), figsize=(7, 7))
    for i, mask in enumerate(masks):
        axs[i].imshow(mask, cmap='gray')
        axs[i].axis('off')  # Hide axes

    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.tight_layout()
    plt.show()

def calculate_iou(mask1, mask2):
    # Ensure the masks are binary
    mask1_bin = mask1 > 0
    mask2_bin = mask2 > 0

    # Intersection is the logical AND between the two masks
    intersection = np.logical_and(mask1_bin, mask2_bin)

    # Union is the logical OR between the two masks
    union = np.logical_or(mask1_bin, mask2_bin)

    # Calculate IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou

def build_ious(predicted_masks, ground_truth_masks):
    ious = []
    # Calculate the IoU for each pair of predicted and ground truth masks
    for i in range(len(predicted_masks)):
        for j in range(len(ground_truth_masks)):
            iou = calculate_iou(predicted_masks[i], ground_truth_masks[j])
            ious.append((i, j, iou))  # Storing the indices along with the IoU for reference
    return ious

def hungarian_algorithm(predicted_masks, ground_truth_masks):
    # Calculate IoU values for each pair of predicted and ground truth masks
    ious = build_ious(predicted_masks, ground_truth_masks)

    # Number of predictions and ground truths
    num_preds = len(predicted_masks)
    num_gts = len(ground_truth_masks)

    # Determine the size of the square cost matrix
    max_items = max(num_preds, num_gts)
    cost_matrix = np.ones((max_items, max_items))

    # Fill in the IoU values (converted to costs) where applicable
    for i, j, iou in ious:
        cost_matrix[i, j] = 1 - iou  # converting IoU to cost
    
    # Use the Hungarian algorithm to find the optimal one-to-one matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Extract the matched pairs and corresponding IoUs, and determine false positives and false negatives
    matches = []
    false_positives = 0
    false_negatives = 0
    for r, c in zip(row_ind, col_ind):
        if r < num_preds and c < num_gts: # valid match
            matches.append((r, c, 1 - cost_matrix[r, c]))
        elif r >= num_preds: # unmatched ground truth (false negative)
            false_negatives += 1
        elif c >= num_gts: # unmatched prediction (false positive)
            false_positives += 1

    return matches, false_positives, false_negatives

def calculate_precision_recall(matches, false_positives, false_negatives):
    # True Positives are the correct matches, which are the ones we found during the matching process
    TP = len(matches)

    # We've already calculated False Positives and False Negatives during the matching process
    FP = false_positives
    FN = false_negatives

    # Calculate precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall

def calculate_f1_score(precision, recall):
    # Calculate the F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score

def categorize_mask_relative(mask, small_threshold=0.01, large_threshold=0.05):
    # Total pixels in the image
    total_pixels = mask.shape[0] * mask.shape[1]

    # Calculate the area of the mask
    area = np.sum(mask == 1.)

    # Calculate the relative area of the mask
    relative_area = area / total_pixels

    # Categorize the mask based on its relative area
    if relative_area < small_threshold:
        return "small"
    elif relative_area < large_threshold:
        return "medium"
    else:
        return "large"
    
def build_masks_by_category(predicted_masks, ground_truth_masks, small_threshold=0.01, large_threshold=0.05):
    categories_gt = [categorize_mask_relative(mask, small_threshold, large_threshold) for mask in ground_truth_masks]
    categories_pred = [categorize_mask_relative(mask, small_threshold, large_threshold) for mask in predicted_masks]
    masks_by_category_pred = {"small": [], "medium": [], "large": []}
    masks_by_category_gt = {"small": [], "medium": [], "large": []}

    # Filter masks into their respective categories
    for mask, category_ in zip(predicted_masks, categories_pred):
        masks_by_category_pred[category_].append(mask)
    for mask, category_ in zip(ground_truth_masks, categories_gt):
        masks_by_category_gt[category_].append(mask)

    return masks_by_category_pred, masks_by_category_gt

    
def build_ious_by_category(masks_by_category_pred, masks_by_category_gt):
    # Initialize a dictionary to store IoUs by category
    ious_by_category = {"small": [], "medium": [], "large": []}

    # Calculate IoUs for each category
    for category_ in ious_by_category.keys():
        pred_masks = masks_by_category_pred[category_]
        gt_masks = masks_by_category_gt[category_]

        for i, pred_mask in enumerate(pred_masks):
            for j, gt_mask in enumerate(gt_masks):
                iou = calculate_iou(pred_mask, gt_mask)
                ious_by_category[category_].append((i, j, iou))  # Store the indices and IoU

    return ious_by_category

def hungarian_matching_categorized(ious, num_preds, num_gts):
    # Create a cost matrix, initializing with a high cost (low IoU)
    cost_matrix = np.ones((num_preds, num_gts)) * 1e7

    # Update the cost matrix with the negative IoU values (since we are minimizing cost)
    for i, j, iou in ious:
        cost_matrix[i, j] = -iou

    # Apply the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter assignments with zero IoU (unmatched)
    matches = [(r, c) for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] < 0]

    return matches


if __name__ == "__main__":
    from data import generate_preds, generate_gts
    # Set random seed for reproducibility
    np.random.seed(0)

    # Creating synthetic predicted masks
    predicted_masks = generate_preds(10, 256, 256)
    ground_truth_masks = generate_gts(8, 256, 256)

    cat_pred, cat_gt = build_masks_by_category(predicted_masks, ground_truth_masks)
    print(cat_pred, cat_gt)
    ious_by_category = build_ious_by_category(cat_pred, cat_gt)
    print(ious_by_category)