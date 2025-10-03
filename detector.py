import cv2
import numpy as np


def detect_fruit(frame):
    """Return (label, confidence) where label is one of 'banana','apple','orange' or None."""
    # convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # banana: yellow range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pct = np.count_nonzero(mask_y) / (mask_y.size)

    # orange: orange range
    lower_orange = np.array([5, 120, 120])
    upper_orange = np.array([20, 255, 255])
    mask_o = cv2.inRange(hsv, lower_orange, upper_orange)
    orange_pct = np.count_nonzero(mask_o) / (mask_o.size)

    # apple: red range (two ranges for hue wrap)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_r = cv2.bitwise_or(mask_r1, mask_r2)
    red_pct = np.count_nonzero(mask_r) / (mask_r.size)

    # choose the highest percent if above threshold
    scores = {'banana': yellow_pct, 'orange': orange_pct, 'apple': red_pct}
    label, score = max(scores.items(), key=lambda kv: kv[1])

    # simple confidence mapping
    conf = float(score * 100)
    if score < 0.002:  # too small
        return (None, 0.0)
    return (label, conf)
