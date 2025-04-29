def generate_response(question: str, features: dict) -> str:
    q = question.lower()
    is_malignant = features["is_malignant"]
    conf = features["confidence_score"]
    iou = features["iou_score"]
    dice = features["dice_score"]

    if "dangerous" in q or "serious" in q:
        return f"{'Yes' if is_malignant else 'No,'} the model predicts {'malignancy' if is_malignant else 'benign findings'} with {conf}% confidence."
    if "confidence" in q:
        return f"The AI model’s confidence level is {conf}%."
    if "iou" in q or "dice" in q or "accuracy" in q:
        return f"Segmentation accuracy: IoU = {iou:.2f}, Dice = {dice:.2f}."
    if "next" in q or "should i do" in q:
        return "Schedule an appointment with a pulmonologist or oncologist." if is_malignant else "Consult your physician for confirmation."
    if "treat" in q:
        return "Treatment depends on stage; early intervention improves outcomes." if is_malignant else "Follow-up imaging may be recommended."
    return "I can help summarize the AI findings. For detailed medical advice, consult your doctor."
