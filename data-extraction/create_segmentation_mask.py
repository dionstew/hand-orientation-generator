"""
generate_sam_masks_from_landmarks.py

Zero-shot hand segmentation dengan SAM memakai prompt bbox + landmark CSV.

Struktur input contoh:
    pseudo_dataset/
    ├── img_left/
    └── landmark/

Output contoh:
    pseudo_dataset/
    ├── semantic_new/
    └── semantic_new_overlay/

Install:
    pip install torch torchvision opencv-python pandas numpy
    pip install git+https://github.com/facebookresearch/segment-anything.git

Contoh run:
    python generate_sam_masks_from_landmarks.py \
        --dataset-root data-extraction/pseudo_dataset_08072026_synced \
        --sam-checkpoint sam_vit_b_01ec64.pth \
        --model-type vit_b \
        --device cpu \
        --margin 35
"""

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as exc:
    raise ImportError(
        "Package segment-anything belum tersedia. Install dengan:\n"
        "pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from exc


def natural_key(path):
    name = os.path.basename(str(path))
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_files(folder, exts):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder}")

    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=natural_key,
    )


def read_landmark_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Format utama dari pipeline sebelumnya: u, v
    if "u" in df.columns and "v" in df.columns:
        pts = df[["u", "v"]].values.astype(np.float32)
    # Fallback jika nama kolom berbeda
    elif "x" in df.columns and "y" in df.columns:
        pts = df[["x", "y"]].values.astype(np.float32)
    else:
        raise ValueError(
            f"CSV landmark harus punya kolom u,v atau x,y: {csv_path}\n"
            f"Kolom tersedia: {list(df.columns)}"
        )

    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]

    return pts


def landmark_to_bbox(points, image_shape, margin=35):
    h, w = image_shape[:2]

    x1 = int(np.min(points[:, 0]) - margin)
    y1 = int(np.min(points[:, 1]) - margin)
    x2 = int(np.max(points[:, 0]) + margin)
    y2 = int(np.max(points[:, 1]) + margin)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"BBox tidak valid: {(x1, y1, x2, y2)}")

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def create_overlay(image_bgr, mask, alpha=0.45):
    overlay = image_bgr.copy()

    binary = mask > 0
    color = np.zeros_like(image_bgr)
    color[binary] = (0, 0, 255)  # merah BGR

    overlay[binary] = cv2.addWeighted(
        image_bgr[binary],
        1.0 - alpha,
        color[binary],
        alpha,
        0,
    )

    return overlay


def clean_mask(mask, kernel_size=5, min_area=500):
    """Post-processing ringan: keep largest component + morphology."""
    mask = (mask > 0).astype(np.uint8) * 255

    if kernel_size and kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    largest_area = stats[largest_idx, cv2.CC_STAT_AREA]

    if largest_area < min_area:
        return mask

    clean = np.zeros_like(mask)
    clean[labels == largest_idx] = 255

    return clean


def build_sam_predictor(model_type, checkpoint, device):
    if model_type not in sam_model_registry:
        raise ValueError(
            f"model_type tidak valid: {model_type}. "
            f"Pilihan: {list(sam_model_registry.keys())}"
        )

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()

    return SamPredictor(sam)


def generate_masks(args):
    dataset_root = Path(args.dataset_root)
    image_dir = dataset_root / args.image_subdir
    landmark_dir = dataset_root / args.landmark_subdir
    mask_out_dir = dataset_root / args.output_mask_subdir
    overlay_out_dir = dataset_root / args.output_overlay_subdir

    mask_out_dir.mkdir(parents=True, exist_ok=True)
    overlay_out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_files(image_dir, {".jpg", ".jpeg", ".png"})
    landmark_paths = list_files(landmark_dir, {".csv"})

    n = min(len(image_paths), len(landmark_paths))

    if args.start_index < 0:
        raise ValueError("--start-index tidak boleh negatif")

    start = min(args.start_index, n)
    end = n if args.limit is None else min(n, start + args.limit)

    print("=== Input ===")
    print("Dataset root :", dataset_root)
    print("Images       :", len(image_paths))
    print("Landmarks    :", len(landmark_paths))
    print("Process      :", start, "to", end - 1, f"({max(0, end - start)} files)")
    print("Output mask  :", mask_out_dir)
    print("Output overlay:", overlay_out_dir)
    print()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled.")

    print("Device:", device)
    print("Model type:", args.model_type)
    print("Checkpoint:", args.sam_checkpoint)
    print()

    predictor = build_sam_predictor(
        model_type=args.model_type,
        checkpoint=args.sam_checkpoint,
        device=device,
    )

    processed = 0
    skipped = 0
    failed = 0

    for i in range(start, end):
        image_path = image_paths[i]
        landmark_path = landmark_paths[i]
        base = image_path.stem

        mask_path = mask_out_dir / f"{base}_sam_mask.png"
        overlay_path = overlay_out_dir / f"{base}_sam_overlay.png"

        if args.skip_existing and mask_path.exists():
            skipped += 1
            continue

        try:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise ValueError(f"Gagal membaca image: {image_path}")

            points = read_landmark_csv(landmark_path)
            if len(points) < args.min_points:
                raise ValueError(f"Landmark valid kurang dari {args.min_points}: {landmark_path}")

            bbox = landmark_to_bbox(points, image_bgr.shape, margin=args.margin)

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)

            if args.box_only:
                masks, scores, _ = predictor.predict(
                    box=bbox,
                    multimask_output=True,
                )
            else:
                point_coords = points.astype(np.float32)
                point_labels = np.ones(len(point_coords), dtype=np.int32)

                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=bbox,
                    multimask_output=True,
                )

            best_idx = int(np.argmax(scores))
            mask = masks[best_idx].astype(np.uint8) * 255

            if args.clean_mask:
                mask = clean_mask(
                    mask,
                    kernel_size=args.kernel_size,
                    min_area=args.min_area,
                )

            cv2.imwrite(str(mask_path), mask)

            if args.save_overlay:
                overlay = create_overlay(image_bgr, mask, alpha=args.alpha)
                cv2.imwrite(str(overlay_path), overlay)

            processed += 1

            if processed % args.print_every == 0 or i == start:
                print(
                    f"[{i}/{end - 1}] saved={mask_path.name} "
                    f"score={float(scores[best_idx]):.4f} bbox={bbox.astype(int).tolist()}"
                )

        except Exception as exc:
            failed += 1
            print(f"[FAILED] index={i} image={image_path.name} reason={exc}")

    print()
    print("=== Done ===")
    print("Processed:", processed)
    print("Skipped  :", skipped)
    print("Failed   :", failed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate zero-shot hand semantic masks using SAM + landmark CSV prompts."
    )

    parser.add_argument("--dataset-root", required=True, help="Root folder pseudo dataset.")
    parser.add_argument("--sam-checkpoint", required=True, help="Path checkpoint SAM .pth.")
    parser.add_argument(
        "--model-type",
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="Tipe model SAM sesuai checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Gunakan cpu jika CUDA/cuDNN bermasalah.",
    )
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="Matikan cuDNN untuk menghindari mismatch CUDA/cuDNN.",
    )

    parser.add_argument("--image-subdir", default="img_left")
    parser.add_argument("--landmark-subdir", default="landmark")
    parser.add_argument("--output-mask-subdir", default="semantic_new")
    parser.add_argument("--output-overlay-subdir", default="semantic_new_overlay")

    parser.add_argument("--margin", type=int, default=35, help="Margin bbox dari landmark dalam pixel.")
    parser.add_argument("--min-points", type=int, default=3, help="Minimal landmark valid.")
    parser.add_argument("--box-only", action="store_true", help="Gunakan bbox saja, tanpa point prompt landmark.")

    parser.add_argument("--clean-mask", action="store_true", help="Post-process mask dengan morphology + largest component.")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--min-area", type=int, default=500)

    parser.add_argument("--save-overlay", action="store_true", default=True)
    parser.add_argument("--no-overlay", dest="save_overlay", action="store_false")
    parser.add_argument("--alpha", type=float, default=0.45)

    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--show-every", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_masks(args)