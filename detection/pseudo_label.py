"""
pseudo_label.py

Auto-labels a folder of unlabeled images using an existing YOLOv8 model
(e.g. your own yolov8s.pt), producing YOLO-format .txt label files.

This is a starting point, NOT ground truth — you should review and fix
the labels afterward (especially motorcycle <-> person/bicycle confusions),
using a tool like LabelImg, CVAT, or Roboflow's annotate view (which can
import these YOLO labels and let you correct them visually).

Usage:
    python pseudo_label.py --images india_roads/ --output india_roads_labeled/ --conf 0.35

Output structure:
    india_roads_labeled/
        images/   <- copies (or symlinks) of your original images
        labels/   <- one .txt per image, YOLO format: class_id x_center y_center w h (normalized)
        low_conf_review.txt  <- list of images with any detection below --review-conf,
                                 flagged for you to check first
"""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

# COCO classes relevant to traffic — restrict pseudo-labeling to these so you
# don't pull in irrelevant COCO classes (e.g. "kite", "chair") that sometimes
# false-fire in cluttered street scenes.
KEEP_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def main():
    parser = argparse.ArgumentParser(description="Pseudo-label images with an existing YOLOv8 model.")
    parser.add_argument("--images", required=True, help="Folder of unlabeled images")
    parser.add_argument("--output", required=True, help="Output folder (images/ + labels/ will be created)")
    parser.add_argument("--model", default="yolov8s.pt", help="Path to YOLOv8 model weights")
    parser.add_argument("--conf", type=float, default=0.35, help="Minimum confidence to keep a detection")
    parser.add_argument("--review-conf", type=float, default=0.5,
                         help="Images with any detection below this confidence get flagged for manual review first")
    args = parser.parse_args()

    img_dir = Path(args.images)
    out_dir = Path(args.output)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    image_exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in image_exts])

    if not image_paths:
        print(f"No images found in {img_dir}")
        return

    print(f"Found {len(image_paths)} images. Running detection with conf>={args.conf}...")

    review_list = []
    class_id_map = {orig_id: new_id for new_id, orig_id in enumerate(KEEP_CLASSES.keys())}
    # write class names file so it's obvious what index maps to what
    with open(out_dir / "classes.txt", "w") as f:
        for orig_id in KEEP_CLASSES:
            f.write(f"{KEEP_CLASSES[orig_id]}\n")

    for i, img_path in enumerate(image_paths, 1):
        results = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        shutil.copy(img_path, out_dir / "images" / img_path.name)

        label_lines = []
        flagged = False
        h, w = results.orig_shape

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in KEEP_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf < args.review_conf:
                flagged = True

            xc, yc, bw, bh = box.xywhn[0].tolist()  # already normalized
            new_cls = class_id_map[cls_id]
            label_lines.append(f"{new_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        label_path = out_dir / "labels" / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

        if flagged:
            review_list.append(img_path.name)

        if i % 200 == 0:
            print(f"  ...{i}/{len(image_paths)} done")

    with open(out_dir / "low_conf_review.txt", "w") as f:
        f.write("\n".join(review_list))

    print(f"\nDone. Labeled {len(image_paths)} images.")
    print(f"{len(review_list)} images flagged for manual review (low-confidence detections) -> low_conf_review.txt")
    print(f"Class mapping saved to {out_dir / 'classes.txt'}")
    print("\nNext: open these in a labeling tool and correct mistakes, especially")
    print("motorcycle-vs-person/bicycle mix-ups, before using for fine-tuning.")


if __name__ == "__main__":
    main()
