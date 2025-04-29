def generate_summary(features: dict) -> str:
    is_malignant = features["is_malignant"]
    conf = features["confidence_score"]
    iou = features["iou_score"]
    dice = features["dice_score"]
    if is_malignant:
        return f"The AI predicts a malignant nodule with {conf:.1f}% confidence. IoU: {iou:.2f}, Dice: {dice:.2f}. Immediate consultation is recommended."
    else:
        return f"The AI predicts a benign nodule with {conf:.1f}% confidence. IoU: {iou:.2f}, Dice: {dice:.2f}. A medical review is still advised."
