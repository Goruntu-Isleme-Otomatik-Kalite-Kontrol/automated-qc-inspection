from pathlib import Path
import cv2

def load_images(input_dir: str):
    folder = Path(input_dir)
    for path in sorted(folder.glob("*.*")):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            img = cv2.imread(str(path))
            if img is not None:
                yield path.name, img