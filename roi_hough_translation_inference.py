#!/usr/bin/env python3
"""
roi_hough_translation_inference.py

Inference translasi PoseCNN-style menggunakan:
1) model segmentasi -> predicted semantic mask
2) model vertex translation -> nx, ny, Tz
3) ROI-restricted Hough voting -> center 2D
4) camera intrinsic -> Tx, Ty, Tz

Contoh:
python roi_hough_translation_inference.py \
  --dataset-root data-extraction/pseudo_dataset_08072026_synced \
  --image-subdir img_left \
  --seg-model models/segmentation_model.keras \
  --vertex-model models/vertex_model.keras \
  --calib Kalibrasi/kalibrasi_dengan_rectify.npz \
  --output-dir hough_roi_output \
  --input-height 224 \
  --input-width 224 \
  --seg-preprocess rgb01 \
  --vertex-preprocess vgg \
  --vote-sign 1 \
  --save-vis
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

import time

# ============================================================
# File utilities
# ============================================================

def natural_key(path):
    name = os.path.basename(str(path))
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    folder = Path(folder)
    return sorted(
        [str(folder / f) for f in os.listdir(folder) if f.lower().endswith(exts)],
        key=natural_key,
    )


# ============================================================
# Camera utilities
# ============================================================

def resize_camera_intrinsics(K, original_size, target_size):
    """
    K             : 3x3 camera matrix untuk image original.
    original_size : (H, W)
    target_size   : (H, W)
    """
    K = np.asarray(K, dtype=np.float32).copy()

    original_h, original_w = original_size
    target_h, target_w = target_size

    scale_x = target_w / float(original_w)
    scale_y = target_h / float(original_h)

    K_resized = K.copy()
    K_resized[0, 0] *= scale_x
    K_resized[1, 1] *= scale_y
    K_resized[0, 2] *= scale_x
    K_resized[1, 2] *= scale_y

    return K_resized


def load_camera_matrix(calib_path, prefer_p1=True):
    with np.load(calib_path) as data:
        keys = set(data.files)

        if prefer_p1 and "P1" in keys:
            P1 = data["P1"]
            return P1[:3, :3].astype(np.float32), "P1[:3,:3]"

        if "K_left" in keys:
            return data["K_left"].astype(np.float32), "K_left"

        if "K" in keys:
            return data["K"].astype(np.float32), "K"

        raise KeyError(f"Tidak menemukan P1/K_left/K. Keys tersedia: {list(keys)}")


def recover_translation_from_center(cx, cy, Tz, K):
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    px = float(K[0, 2])
    py = float(K[1, 2])

    Tx = (cx - px) * Tz / fx
    Ty = (cy - py) * Tz / fy

    return np.array([Tx, Ty, Tz], dtype=np.float32)


# ============================================================
# Model preprocessing and prediction
# ============================================================

def preprocess_image_bgr(image_bgr, input_size, mode="rgb01"):
    """
    mode:
    - rgb01 : RGB, 0..1
    - bgr01 : BGR, 0..1
    - vgg   : RGB resized 0..255 lalu tf.keras.applications.vgg16.preprocess_input
    - raw255: RGB, 0..255
    """
    input_h, input_w = input_size

    if mode not in {"rgb01", "bgr01", "vgg", "raw255"}:
        raise ValueError("mode harus salah satu: rgb01, bgr01, vgg, raw255")

    if mode == "bgr01":
        img = cv2.resize(image_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        return x[None, ...]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    if mode == "rgb01":
        x = image_rgb.astype(np.float32) / 255.0
    elif mode == "raw255":
        x = image_rgb.astype(np.float32)
    else:  # vgg
        x = image_rgb.astype(np.float32)
        x = tf.keras.applications.vgg16.preprocess_input(x)

    return x[None, ...]


def as_numpy_output(y):
    if isinstance(y, dict):
        y = list(y.values())[0]

    if isinstance(y, (list, tuple)):
        y = y[0]

    if hasattr(y, "numpy"):
        y = y.numpy()

    y = np.asarray(y)

    if y.ndim == 4:
        y = y[0]

    return y


def maybe_sigmoid(mask_pred):
    mask_pred = np.asarray(mask_pred, dtype=np.float32)

    if mask_pred.ndim == 2:
        mask_pred = mask_pred[..., None]

    if np.nanmin(mask_pred) < 0.0 or np.nanmax(mask_pred) > 1.0:
        mask_pred = 1.0 / (1.0 + np.exp(-mask_pred))

    return mask_pred.astype(np.float32)


def predict_segmentation(seg_model, image_bgr, input_size, preprocess_mode="rgb01"):
    x = preprocess_image_bgr(image_bgr, input_size, mode=preprocess_mode)
    pred = seg_model(x, training=False)
    pred = as_numpy_output(pred)
    pred = maybe_sigmoid(pred)

    input_h, input_w = input_size

    if pred.shape[:2] != (input_h, input_w):
        pred = cv2.resize(pred, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        if pred.ndim == 2:
            pred = pred[..., None]

    return pred.astype(np.float32)


def predict_vertex(vertex_model, image_bgr, input_size, preprocess_mode="vgg"):
    x = preprocess_image_bgr(image_bgr, input_size, mode=preprocess_mode)
    pred = vertex_model(x, training=False)
    pred = as_numpy_output(pred).astype(np.float32)

    input_h, input_w = input_size

    if pred.shape[:2] != (input_h, input_w):
        pred = cv2.resize(pred, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    if pred.ndim != 3 or pred.shape[-1] < 3:
        raise ValueError(f"Output vertex harus HxWx3, tetapi dapat: {pred.shape}")

    return pred[..., :3]


# ============================================================
# ROI Hough voting
# ============================================================

def get_mask_bbox(mask, margin=8):
    """
    mask: HxW boolean
    return: x1, y1, x2, y2
    """
    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise ValueError("Mask kosong, bbox tidak bisa dibuat.")

    H, W = mask.shape

    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(W - 1, int(xs.max()) + margin)
    y2 = min(H - 1, int(ys.max()) + margin)

    return x1, y1, x2, y2


def mask_centroid(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def hough_vote_center_roi(
    vertex_pred,
    mask_prob,
    mask_threshold=0.5,
    pixel_stride=1,
    vote_step=1,
    bbox_margin=5,
    vote_sign=1,
    use_center_prior=True,
    prior_sigma_ratio=0.35,
    min_vector_norm=1e-4,
    gaussian_blur=3,
):
    """
    vertex_pred: HxWx3, channel 0=nx, 1=ny, 2=Tz
    mask_prob  : HxW atau HxWx1, nilai 0..1

    vote_sign=1  : ray dari pixel mengikuti arah nx, ny
    vote_sign=-1 : ray kebalikan arah nx, ny
    """
    H, W = vertex_pred.shape[:2]

    if mask_prob.ndim == 3:
        mask_prob = mask_prob[..., 0]

    mask_prob = mask_prob.astype(np.float32)
    foreground = mask_prob >= mask_threshold

    if np.sum(foreground) == 0:
        raise ValueError("Predicted semantic mask kosong.")

    x1, y1, x2, y2 = get_mask_bbox(foreground, margin=bbox_margin)

    roi_w = x2 - x1 + 1
    roi_h = y2 - y1 + 1
    max_vote_len = int(np.ceil(np.sqrt(roi_w**2 + roi_h**2)))

    nx = vertex_pred[..., 0].astype(np.float32)
    ny = vertex_pred[..., 1].astype(np.float32)

    norm = np.sqrt(nx**2 + ny**2)
    valid = foreground & np.isfinite(norm) & (norm > min_vector_norm)

    ys, xs = np.where(valid)

    if len(xs) == 0:
        raise ValueError("Tidak ada vector valid di area foreground.")

    keep = ((xs % pixel_stride) == 0) & ((ys % pixel_stride) == 0)
    xs = xs[keep]
    ys = ys[keep]

    if len(xs) == 0:
        raise ValueError("Tidak ada vector valid setelah pixel_stride.")

    accumulator = np.zeros((H, W), dtype=np.float32)

    for x0, y0 in zip(xs, ys):
        vx = vote_sign * nx[y0, x0] / norm[y0, x0]
        vy = vote_sign * ny[y0, x0] / norm[y0, x0]
        weight = float(mask_prob[y0, x0])

        for t in range(0, max_vote_len, vote_step):
            x = int(round(x0 + t * vx))
            y = int(round(y0 + t * vy))

            if x < x1 or x > x2 or y < y1 or y > y2:
                break

            accumulator[y, x] += weight

    roi_mask = np.zeros((H, W), dtype=np.float32)
    roi_mask[y1:y2 + 1, x1:x2 + 1] = 1.0
    accumulator *= roi_mask

    if use_center_prior:
        c = mask_centroid(foreground)
        if c is not None:
            cx_m, cy_m = c
            yy, xx = np.indices((H, W), dtype=np.float32)
            sigma = max(roi_w, roi_h) * prior_sigma_ratio
            sigma = max(sigma, 1.0)
            prior = np.exp(-((xx - cx_m) ** 2 + (yy - cy_m) ** 2) / (2 * sigma ** 2))
            accumulator *= prior.astype(np.float32)

    if gaussian_blur and gaussian_blur > 1:
        k = int(gaussian_blur)
        if k % 2 == 0:
            k += 1
        accumulator = cv2.GaussianBlur(accumulator, (k, k), 0)

    if np.max(accumulator) <= 0:
        raise ValueError("Accumulator kosong / semua nol.")

    cy, cx = np.unravel_index(np.argmax(accumulator), accumulator.shape)

    return {
        "cx": float(cx),
        "cy": float(cy),
        "accumulator": accumulator,
        "foreground": foreground,
        "bbox": (x1, y1, x2, y2),
    }


def recover_translation_from_hough(hough_out, vertex_pred, K_model, depth_scale=1.0, tz_method="median"):
    foreground = hough_out["foreground"]
    cx = hough_out["cx"]
    cy = hough_out["cy"]

    tz_map = vertex_pred[..., 2].astype(np.float32) * float(depth_scale)
    valid_tz = foreground & np.isfinite(tz_map) & (tz_map > 0)

    if np.sum(valid_tz) == 0:
        raise ValueError("Tidak ada Tz valid pada mask.")

    if tz_method == "mean":
        Tz = float(np.mean(tz_map[valid_tz]))
    elif tz_method == "center":
        yy = int(np.clip(round(cy), 0, tz_map.shape[0] - 1))
        xx = int(np.clip(round(cx), 0, tz_map.shape[1] - 1))
        Tz = float(tz_map[yy, xx])
    else:
        Tz = float(np.median(tz_map[valid_tz]))

    return recover_translation_from_center(cx, cy, Tz, K_model)


# ============================================================
# Visualization utilities
# ============================================================

def normalize_to_uint8(arr):
    arr = np.asarray(arr, dtype=np.float32)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if mx - mn < 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - mn) / (mx - mn)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def create_overlay_visualization(
    image_bgr,
    mask_prob,
    hough_out,
    input_size,
    mask_threshold=0.5,
    alpha=0.45,
):
    orig_h, orig_w = image_bgr.shape[:2]
    input_h, input_w = input_size

    if mask_prob.ndim == 3:
        mask_prob_2d = mask_prob[..., 0]
    else:
        mask_prob_2d = mask_prob

    mask_orig = cv2.resize(mask_prob_2d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask_bin = mask_orig >= mask_threshold

    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[mask_bin] = (0, 0, 255)
    overlay[mask_bin] = cv2.addWeighted(image_bgr[mask_bin], 1.0 - alpha, color[mask_bin], alpha, 0)

    sx = orig_w / float(input_w)
    sy = orig_h / float(input_h)

    cx_orig = hough_out["cx"] * sx
    cy_orig = hough_out["cy"] * sy

    cv2.circle(overlay, (int(round(cx_orig)), int(round(cy_orig))), 7, (0, 255, 255), -1)
    cv2.circle(overlay, (int(round(cx_orig)), int(round(cy_orig))), 12, (0, 0, 0), 2)

    x1, y1, x2, y2 = hough_out["bbox"]
    pt1 = (int(round(x1 * sx)), int(round(y1 * sy)))
    pt2 = (int(round(x2 * sx)), int(round(y2 * sy)))
    cv2.rectangle(overlay, pt1, pt2, (0, 255, 255), 2)

    return overlay


def create_accumulator_visualization(accumulator):
    acc_u8 = normalize_to_uint8(accumulator)
    return cv2.applyColorMap(acc_u8, cv2.COLORMAP_JET)


def create_direction_visualization(
    image_bgr,
    vertex_pred,
    foreground,
    hough_out,
    input_size,
    arrow_stride=8,
    arrow_length=18,
):
    orig_h, orig_w = image_bgr.shape[:2]
    input_h, input_w = input_size
    sx = orig_w / float(input_w)
    sy = orig_h / float(input_h)

    out = image_bgr.copy()
    nx = vertex_pred[..., 0].astype(np.float32)
    ny = vertex_pred[..., 1].astype(np.float32)

    ys, xs = np.where(foreground)
    keep = ((xs % arrow_stride) == 0) & ((ys % arrow_stride) == 0)
    xs = xs[keep]
    ys = ys[keep]

    for x, y in zip(xs, ys):
        vx = nx[y, x] * sx
        vy = ny[y, x] * sy
        n = np.sqrt(vx * vx + vy * vy)
        if not np.isfinite(n) or n < 1e-6:
            continue
        vx /= n
        vy /= n
        x0 = int(round(x * sx))
        y0 = int(round(y * sy))
        x1 = int(round(x0 + arrow_length * vx))
        y1 = int(round(y0 + arrow_length * vy))
        cv2.arrowedLine(out, (x0, y0), (x1, y1), (0, 0, 0), 1, tipLength=0.25)

    cx_orig = hough_out["cx"] * sx
    cy_orig = hough_out["cy"] * sy
    cv2.circle(out, (int(round(cx_orig)), int(round(cy_orig))), 7, (0, 255, 255), -1)
    cv2.circle(out, (int(round(cx_orig)), int(round(cy_orig))), 12, (0, 0, 0), 2)

    return out


# ============================================================
# Main inference
# ============================================================

def run_one_image(image_path, seg_model, vertex_model, K_orig, args):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Gagal membaca image: {image_path}")

    orig_h, orig_w = image_bgr.shape[:2]
    input_size = (args.input_height, args.input_width)

    K_model = resize_camera_intrinsics(
        K_orig,
        original_size=(orig_h, orig_w),
        target_size=input_size,
    )

    mask_prob = predict_segmentation(
        seg_model=seg_model,
        image_bgr=image_bgr,
        input_size=input_size,
        preprocess_mode=args.seg_preprocess,
    )

    vertex_pred = predict_vertex(
        vertex_model=vertex_model,
        image_bgr=image_bgr,
        input_size=input_size,
        preprocess_mode=args.vertex_preprocess,
    )

    hough_out = hough_vote_center_roi(
        vertex_pred=vertex_pred,
        mask_prob=mask_prob,
        mask_threshold=args.mask_threshold,
        pixel_stride=args.pixel_stride,
        vote_step=args.vote_step,
        bbox_margin=args.bbox_margin,
        vote_sign=args.vote_sign,
        use_center_prior=not args.no_center_prior,
        prior_sigma_ratio=args.prior_sigma_ratio,
        min_vector_norm=args.min_vector_norm,
        gaussian_blur=args.gaussian_blur,
    )

    translation = recover_translation_from_hough(
        hough_out=hough_out,
        vertex_pred=vertex_pred,
        K_model=K_model,
        depth_scale=args.depth_scale,
        tz_method=args.tz_method,
    )

    sx = orig_w / float(args.input_width)
    sy = orig_h / float(args.input_height)
    cx_orig = hough_out["cx"] * sx
    cy_orig = hough_out["cy"] * sy

    return {
        "image_bgr": image_bgr,
        "mask_prob": mask_prob,
        "vertex_pred": vertex_pred,
        "hough_out": hough_out,
        "K_model": K_model,
        "cx_model": hough_out["cx"],
        "cy_model": hough_out["cy"],
        "cx_orig": cx_orig,
        "cy_orig": cy_orig,
        "Tx": float(translation[0]),
        "Ty": float(translation[1]),
        "Tz": float(translation[2]),
    }


def save_outputs(image_path, result, args, out_dirs):
    base = Path(image_path).stem

    if args.save_mask:
        mask_prob = result["mask_prob"]
        if mask_prob.ndim == 3:
            mask_prob = mask_prob[..., 0]
        mask_bin = (mask_prob >= args.mask_threshold).astype(np.uint8) * 255
        cv2.imwrite(str(out_dirs["mask"] / f"{base}_mask.png"), mask_bin)

    if args.save_vis:
        overlay = create_overlay_visualization(
            result["image_bgr"],
            result["mask_prob"],
            result["hough_out"],
            input_size=(args.input_height, args.input_width),
            mask_threshold=args.mask_threshold,
        )
        cv2.imwrite(str(out_dirs["overlay"] / f"{base}_overlay_center.png"), overlay)

        direction = create_direction_visualization(
            result["image_bgr"],
            result["vertex_pred"],
            result["hough_out"]["foreground"],
            result["hough_out"],
            input_size=(args.input_height, args.input_width),
            arrow_stride=args.arrow_stride,
            arrow_length=args.arrow_length,
        )
        cv2.imwrite(str(out_dirs["direction"] / f"{base}_direction.png"), direction)

    if args.save_accumulator:
        acc_vis = create_accumulator_visualization(result["hough_out"]["accumulator"])
        cv2.imwrite(str(out_dirs["accumulator"] / f"{base}_accumulator.png"), acc_vis)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ROI Hough voting inference untuk PoseCNN-style translation branch."
    )

    parser.add_argument("--dataset-root", required=True, help="Root dataset.")
    parser.add_argument("--image-subdir", default="img_left", help="Subfolder image input.")
    parser.add_argument("--seg-model", required=True, help="Path model segmentasi .keras/.h5.")
    parser.add_argument("--vertex-model", required=True, help="Path model vertex translation .keras/.h5.")
    parser.add_argument("--calib", required=True, help="Path file kalibrasi .npz.")
    parser.add_argument("--output-dir", default="hough_roi_translation_output", help="Folder output.")

    parser.add_argument("--input-height", type=int, default=224)
    parser.add_argument("--input-width", type=int, default=224)

    parser.add_argument(
        "--seg-preprocess",
        default="rgb01",
        choices=["rgb01", "bgr01", "vgg", "raw255"],
        help="Preprocessing untuk segmentation model.",
    )
    parser.add_argument(
        "--vertex-preprocess",
        default="vgg",
        choices=["rgb01", "bgr01", "vgg", "raw255"],
        help="Preprocessing untuk vertex model.",
    )

    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument("--vote-step", type=int, default=1)
    parser.add_argument("--bbox-margin", type=int, default=5)
    parser.add_argument("--vote-sign", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--no-center-prior", action="store_true")
    parser.add_argument("--prior-sigma-ratio", type=float, default=0.35)
    parser.add_argument("--min-vector-norm", type=float, default=1e-4)
    parser.add_argument("--gaussian-blur", type=int, default=3)

    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--tz-method", default="median", choices=["median", "mean", "center"])

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prefer-k-left", action="store_true", help="Gunakan K_left, bukan P1, jika tersedia.")

    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--save-accumulator", action="store_true")
    parser.add_argument("--arrow-stride", type=int, default=8)
    parser.add_argument("--arrow-length", type=int, default=18)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--show-every", type=int, default=20)

    return parser.parse_args()


def main():
    args = parse_args()

    input_size = (args.input_height, args.input_width)

    image_dir = Path(args.dataset_root) / args.image_subdir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_dirs = {
        "overlay": output_dir / "overlay_center",
        "direction": output_dir / "direction_field",
        "accumulator": output_dir / "accumulator",
        "mask": output_dir / "predicted_mask",
    }

    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(image_dir)

    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    if len(image_paths) == 0:
        raise ValueError(f"Tidak ada image ditemukan di: {image_dir}")

    print("Images        :", len(image_paths))
    print("Input size    :", input_size)
    print("Seg preprocess:", args.seg_preprocess)
    print("Vtx preprocess:", args.vertex_preprocess)
    print("Vote sign     :", args.vote_sign)
    print("Output dir    :", output_dir)

    print("Loading models...")
    seg_model = tf.keras.models.load_model(args.seg_model, compile=False)
    vertex_model = tf.keras.models.load_model(args.vertex_model, compile=False)

    K_orig, K_source = load_camera_matrix(args.calib, prefer_p1=not args.prefer_k_left)
    print("Camera source :", K_source)
    print(K_orig)

    rows = []
    deployed_time = time.perf_counter()

    for i, image_path in enumerate(image_paths):
        try:
            start_time = time.perf_counter()
            result = run_one_image(
                image_path=image_path,
                seg_model=seg_model,
                vertex_model=vertex_model,
                K_orig=K_orig,
                args=args,
            )

            hough_out = result["hough_out"]
            x1, y1, x2, y2 = hough_out["bbox"]

            rows.append({
                "image": image_path,
                "cx_model": result["cx_model"],
                "cy_model": result["cy_model"],
                "cx_orig": result["cx_orig"],
                "cy_orig": result["cy_orig"],
                "Tx": result["Tx"],
                "Ty": result["Ty"],
                "Tz": result["Tz"],
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "foreground_pixels": int(np.sum(hough_out["foreground"])),
                "accumulator_max": float(np.max(hough_out["accumulator"])),
                "status": "ok",
                "error": "",
            })

            save_outputs(image_path, result, args, out_dirs)

            end_time = time.perf_counter()
            if args.verbose:
                print(
                    f"[{i}/{len(image_paths)}] "
                    f"center_model=({result['cx_model']:.1f},{result['cy_model']:.1f}) "
                    f"center_orig=({result['cx_orig']:.1f},{result['cy_orig']:.1f}) "
                    f"T=({result['Tx']:.4f},{result['Ty']:.4f},{result['Tz']:.4f}) "
                    f"Time Elapsed: {end_time-deployed_time:.2f} "
                    f"Process Time: {end_time-start_time:.2f}"
                )
            else:
                if i % args.show_every == 0:
                 print(
                    f"[{i}/{len(image_paths)}] "
                    f"center_model=({result['cx_model']:.1f},{result['cy_model']:.1f}) "
                    f"center_orig=({result['cx_orig']:.1f},{result['cy_orig']:.1f}) "
                    f"T=({result['Tx']:.4f},{result['Ty']:.4f},{result['Tz']:.4f}) "
                    f"Time Elapsed: {end_time-deployed_time:.2f} seconds "
                    f"Process Time: {end_time-start_time:.2f} seconds"
                )
                else :
                    pass
                

        except Exception as e:
            rows.append({
                "image": image_path,
                "cx_model": np.nan,
                "cy_model": np.nan,
                "cx_orig": np.nan,
                "cy_orig": np.nan,
                "Tx": np.nan,
                "Ty": np.nan,
                "Tz": np.nan,
                "bbox_x1": np.nan,
                "bbox_y1": np.nan,
                "bbox_x2": np.nan,
                "bbox_y2": np.nan,
                "foreground_pixels": np.nan,
                "accumulator_max": np.nan,
                "status": "error",
                "error": str(e),
            })
            print("[ERROR]", image_path, e)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "translation_predictions_roi_hough.csv"
    df.to_csv(csv_path, index=False)

    print("Done.")
    print("Saved CSV:", csv_path)


if __name__ == "__main__":
    main()
