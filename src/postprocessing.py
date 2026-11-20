from pathlib import Path
import cv2
import numpy as np

def annotate_and_save(image: np.ndarray, boxes: list, is_defective: bool, save_path: Path):
    annotated = image.copy()
    color = (0, 0, 255) if is_defective else (0, 200, 0)
    label = "FAIL" if is_defective else "PASS"

    for (x, y, w, h) in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

    cv2.putText(annotated, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), annotated)