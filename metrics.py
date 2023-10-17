import numpy as np

from utils import *
from data import generate_gts, generate_preds

def calculate_AP_AR_50(matches, false_positives, false_negatives):
    # Initialize lists to store precision and recall values
    precisions = []
    recalls = []

    # Set the IoU threshold to 0.5
    iou_threshold = 0.5

    FP = false_positives
    FN = false_negatives

    # For the purpose of this example, we'll calculate precision and recall for each individual match
    # Then, we'll calculate AP as the mean of these precision values, and AR as the mean of recall values
    for index, match in enumerate(matches):
        iou = match[2]  # extract the IoU for the match
        if iou >= iou_threshold:
            TP = index + 1  # true positives up to the current match
            FP = FP  # false positives remain constant in this simplified example
            FN = FN  # false negatives remain constant as well

            # Calculate precision and recall for this match
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            
            precisions.append(precision)
            recalls.append(recall)

    # AP is the average precision across all matches
    # AR is the average recall across all matches
    AP = np.mean(precisions) if precisions else 0
    AR = np.mean(recalls) if recalls else 0

    return AP, AR

def calculate_mAP_mAR(matches, false_positives, false_negatives):
    # Initialize lists to store AP and AR values for each IoU threshold
    APs = []
    ARs = []

    FP = false_positives
    FN = false_negatives

    # IoU thresholds
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    # Calculate AP and AR for each IoU threshold
    for iou_threshold in iou_thresholds:
        # Initialize lists to store precision and recall values for this IoU threshold
        precisions = []
        recalls = []

        for index, match in enumerate(matches):
            iou = match[2]  # extract the IoU for the match
            if iou >= iou_threshold:
                TP = index + 1  # true positives up to the current match
                FP = FP  # false positives remain constant in this simplified example
                FN = FN # false negatives remain constant as well

                # Calculate precision and recall for this match
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                
                precisions.append(precision)
                recalls.append(recall)

        # AP is the average precision across all matches at this IoU threshold
        # AR is the average recall across all matches at this IoU threshold
        AP = np.mean(precisions) if precisions else 0
        AR = np.mean(recalls) if recalls else 0

        APs.append(AP)
        ARs.append(AR)

    # Calculate the mean AP and AR across all IoU thresholds
    mean_AP = np.mean(APs)
    mean_AR = np.mean(ARs)

    return mean_AP, mean_AR


def calculate_AP_AR_50_categorized(predicted_masks, ground_truth_masks, small_threshold=0.01, large_threshold=0.05):
    # Categorize the masks based on their relative size
    masks_by_category_pred, masks_by_category_gt = build_masks_by_category(predicted_masks, 
                                                                           ground_truth_masks, 
                                                                           small_threshold, 
                                                                           large_threshold)
    
    # Calculate IoUs for each category
    ious_by_category = build_ious_by_category(masks_by_category_pred, masks_by_category_gt)

    # Initialize dictionaries to store matches and metrics by category
    matches_by_category = {"small": [], "medium": [], "large": []}
    metrics_by_category = {"small": None, "medium": None, "large": None}
    ap_ar_by_category = {"small": {"AP": 0, "AR": 0}, "medium": {"AP": 0, "AR": 0}, "large": {"AP": 0, "AR": 0}}


    # IoU threshold for a match
    iou_threshold = 0.5

    for category in ious_by_category.keys():
        ious = ious_by_category[category]
        num_preds = len(masks_by_category_pred[category])
        num_gts = len(masks_by_category_gt[category])

        # Hungarian matching
        matches = hungarian_matching_categorized(ious, num_preds, num_gts)

        # Calculate metrics (precision, recall, AP, AR) based on the matches
        true_positives = len([1 for _, _, iou in ious if iou > iou_threshold])
        false_positives = num_preds - true_positives
        false_negatives = num_gts - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        # For simplicity, we are considering AP as precision and AR as recall in this context
        # In a complete implementation, AP is the area under the precision-recall curve
        matches_by_category[category] = matches
        metrics_by_category[category] = {"precision": precision, "recall": recall, "AP": precision, "AR": recall}

        ap_ar_by_category[category]["AP"] = precision
        ap_ar_by_category[category]["AR"] = recall

    return ap_ar_by_category


def calculate_mAP_mAR_categorized(predicted_masks, ground_truth_masks, small_threshold=0.01, large_threshold=0.05):
    # Categorize the masks based on their relative size
    masks_by_category_pred, masks_by_category_gt = build_masks_by_category(predicted_masks, 
                                                                           ground_truth_masks, 
                                                                           small_threshold, 
                                                                           large_threshold)
    
    # Calculate IoUs for each category
    ious_by_category = build_ious_by_category(masks_by_category_pred, masks_by_category_gt)

    # Define the IoU thresholds at which to evaluate AP and AR
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    # Initialize structures to store precision and recall values at each IoU threshold
    metrics_by_category_per_iou = {iou: {"small": None, "medium": None, "large": None} for iou in iou_thresholds}
    categories = ["small", "medium", "large"]

    for iou_threshold in iou_thresholds:
        for category in ["small", "medium", "large"]:
            ious = ious_by_category[category]
            num_preds = len(masks_by_category_pred[category])
            num_gts = len(masks_by_category_gt[category])

            # Hungarian matching
            matches = hungarian_matching_categorized(ious, num_preds, num_gts)

            # Calculate metrics (precision, recall) based on the matches
            true_positives = len([1 for _, _, iou in ious if iou > iou_threshold])
            false_positives = num_preds - true_positives
            false_negatives = num_gts - true_positives

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

            # Store the metrics
            metrics_by_category_per_iou[iou_threshold][category] = {"precision": precision, "recall": recall}

    # Now, calculate mAP and mAR for each category again
    map_mar_by_category = {}
    for category in categories:
        # Retrieve precision and recall for this category at each IoU threshold
        precision_by_iou = []
        recall_by_iou = []
        for iou_threshold in iou_thresholds:
            metrics = metrics_by_category_per_iou[iou_threshold][category]
            if metrics:  # Check if there are metrics for this category at this IoU threshold
                precision_by_iou.append(metrics["precision"])
                recall_by_iou.append(metrics["recall"])

        # Calculate the mean AP and AR for this category
        mean_AP_category = np.mean(precision_by_iou) if precision_by_iou else 0
        mean_AR_category = np.mean(recall_by_iou) if recall_by_iou else 0

        # Store the results
        map_mar_by_category[category] = {"mAP": mean_AP_category, "mAR": mean_AR_category}

    return map_mar_by_category




if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(0)

    # Creating synthetic predicted masks
    predicted_masks = generate_preds(10, 256, 256)
    ground_truth_masks = generate_gts(8, 256, 256)

    matches, fp, fn = hungarian_algorithm(predicted_masks[:4], predicted_masks)
    AP, AR = calculate_AP_AR_50(matches, fp, fn)
    mAP, mAR = calculate_mAP_mAR(matches, fp, fn)
    f1_score = calculate_f1_score(AP, AR)
    precision, recall = calculate_precision_recall(matches, fp, fn)
    AP_AR_categorized = calculate_AP_AR_50_categorized(predicted_masks[:4], predicted_masks)
    mAP_mAR_categorized= calculate_mAP_mAR_categorized(predicted_masks[:4], predicted_masks)

    print("AP@[IoU: 0.5][AREA = all]:", AP)
    print("AP@[IoU: 0.5][AREA = small]:", AP_AR_categorized['small']['AP'])
    print("AP@[IoU: 0.5][AREA = medium]:", AP_AR_categorized['medium']['AP'])
    print("AP@[IoU: 0.5][AREA = large]:", AP_AR_categorized['large']['AP'])
    print("AR@[IoU: 0.5][AREA = all]:", AR)
    print("AR@[IoU: 0.5][AREA = small]:", AP_AR_categorized['small']['AR'])
    print("AR@[IoU: 0.5][AREA = medium]:", AP_AR_categorized['medium']['AR'])
    print("AR@[IoU: 0.5][AREA = large]:", AP_AR_categorized['large']['AR'])
    print('\n')
    print("mAP@[IoU: 0.5:0.05:0.95][AREA = all]:", mAP)
    print("mAP@[IoU: 0.5:0.05:0.95][AREA = small]:", mAP_mAR_categorized['small']['mAP'])
    print("mAP@[IoU: 0.5:0.05:0.95][AREA = medium]:", mAP_mAR_categorized['medium']['mAP'])
    print("mAP@[IoU: 0.5:0.05:0.95][AREA = large]:", mAP_mAR_categorized['large']['mAP'])
    print("mAR@[IoU: 0.5:0.05:0.95][AREA = all]:", mAR)
    print("mAR@[IoU: 0.5:0.05:0.95][AREA = small]:", mAP_mAR_categorized['small']['mAR'])
    print("mAR@[IoU: 0.5:0.05:0.95][AREA = medium]:", mAP_mAR_categorized['medium']['mAR'])
    print("mAR@[IoU: 0.5:0.05:0.95][AREA = large]:", mAP_mAR_categorized['large']['mAR'])
    print('\n')
