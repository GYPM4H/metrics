import numpy as np

def random_rectangle(H, W):
    x1, y1 = np.random.randint(0, W - 10), np.random.randint(0, H - 10)
    x2, y2 = np.random.randint(x1 + 10, W), np.random.randint(y1 + 10, H)
    return x1, y1, x2, y2

def generate_preds(N, H, W):
    # Creating synthetic predicted masks
    predicted_masks_enhanced = np.zeros((N, H, W), dtype=np.uint8)
    for i in range(N):
        x1, y1, x2, y2 = random_rectangle(H, W)
        predicted_masks_enhanced[i, y1:y2, x1:x2] = 1.  # random rectangle object
    return predicted_masks_enhanced

def generate_gts(M, H, W):
    # Creating synthetic ground truth masks
    ground_truth_masks_enhanced = np.zeros((M, H, W), dtype=np.uint8)
    for i in range(M):
        x1, y1, x2, y2 = random_rectangle(H, W)
        ground_truth_masks_enhanced[i, y1:y2, x1:x2] = 1.  # random rectangle object
    return ground_truth_masks_enhanced