import numpy as np

from metrics import calculate_map_mar, calculate_iou, build_cost_matrix

if __name__ == "__main__":
    # predicted_masks = np.array([
    #     np.array([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),  # Predicted mask 0
    #     np.array([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1]]),  # Predicted mask 1
    #     np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]]),  # Predicted mask 2
    #     np.array([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]]),  # Predicted mask 3
    #     np.array([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1], [1, 0, 0, 0, 1], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])   # Predicted mask 4 (False positive)
    # ])

    # ground_truth_masks = np.array([
    #     np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),  # Ground truth mask 0
    #     np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]]),  # Ground truth mask 1
    #     np.array([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1]]),  # Ground truth mask 2
    #     np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [0, 1, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]])   # Ground truth mask 3
    # ])

    predicted_masks = np.load("./SEVENLINES_WATER.npy")
    ground_truth_masks = np.load("./SEVENLINES_GT.npy")

    # iou_values = []
    # for i in range(4):  # we only have 4 ground truth masks
    #     iou = calculate_iou(predicted_masks[i], ground_truth_masks[i])
    #     iou_values.append(iou)

    # # Display the actual IoU values
    # print("Actual IoU values:", iou_values)

    # print("Cost matrix", build_cost_matrix(predicted_masks, ground_truth_masks))

    # We'll bypass the actual IoU calculation by providing pre-calculated IoU values
    iou_thresholds = np.arange(0.5, 1, 0.05)

    mAP, mAR = calculate_map_mar(predicted_masks, ground_truth_masks, iou_thresholds)

    print(f"mAP: {mAP}, mAR: {mAR}")