#!/usr/bin/env python3
"""
Evaluate PoseCNN-style rotation branch with ROI Align.

Input pipeline must match train_rotation_branch_roi.py:
  full image -> VGG preprocess -> ROI box from semantic mask -> model -> quaternion

Outputs:
  - rotation_eval_predictions.csv
  - rotation_eval_summary.txt
  - optional visualization images with ROI boxes and quaternion angle error
"""

import os
import re
import gc
import json
import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PoseCNN-style rotation branch ROI model.")

    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--model", type=str, required=True, help="Path to .keras model from training script.")

    p.add_argument("--image-subdir", type=str, default="img_left")
    p.add_argument("--mask-subdir", type=str, default="semantic")
    p.add_argument("--summary-csv", type=str, default="pseudo_ground_truth_summary.csv")

    p.add_argument("--image-height", type=int, default=224)
    p.add_argument("--image-width", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)

    p.add_argument("--bbox-margin", type=int, default=8)
    p.add_argument("--min-mask-area", type=int, default=30)

    p.add_argument("--eval-split", type=str, default="all", choices=["all", "train", "val"],
                   help="Evaluate all samples or reproduce train/val split using val-ratio and seed.")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="0 means no limit.")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--cuda-malloc-async", action="store_true")
    p.add_argument("--tf-data-workers", type=int, default=2)
    p.add_argument("--prefetch", type=int, default=1)

    p.add_argument("--output-dir", type=str, default="rotation_eval_output")
    p.add_argument("--save-vis", action="store_true")
    p.add_argument("--vis-count", type=int, default=30)

    p.add_argument("--dry-run", action="store_true", help="Validate model/data on one batch only.")
    p.add_argument("--verbose", action='store_true')

    return p.parse_args()


def configure_environment(args):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == "gpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    if args.cuda_malloc_async:
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def natural_key(path):
    name = os.path.basename(str(path))
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_files(folder, exts):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder}")
    return sorted(
        [str(folder / f) for f in os.listdir(folder) if f.lower().endswith(exts)],
        key=natural_key,
    )


def normalize_quaternion_np(q, eps=1e-8):
    q = np.asarray(q, dtype=np.float32).reshape(4)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < eps:
        return None
    return q / norm


def quaternion_angle_error_deg_np(q_true, q_pred):
    q_true = normalize_quaternion_np(q_true)
    q_pred = normalize_quaternion_np(q_pred)
    if q_true is None or q_pred is None:
        return np.nan, np.nan
    dot = float(np.abs(np.sum(q_true * q_pred)))
    dot = float(np.clip(dot, 0.0, 1.0))
    angle = float(2.0 * np.arccos(dot) * 180.0 / np.pi)
    return angle, dot


def load_paths_labels_dataframe(cfg):
    dataset_root = Path(cfg.dataset_root)
    image_dir = dataset_root / cfg.image_subdir
    mask_dir = dataset_root / cfg.mask_subdir
    summary_path = dataset_root / cfg.summary_csv

    image_paths = list_files(image_dir, (".jpg", ".jpeg", ".png", ".bmp"))
    mask_paths = list_files(mask_dir, (".jpg", ".jpeg", ".png", ".bmp"))

    if not summary_path.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {summary_path}")

    df = pd.read_csv(summary_path)
    quat_cols = ["qw", "qx", "qy", "qz"]
    missing = [c for c in quat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom quaternion tidak ditemukan: {missing}")

    raw_q = df[quat_cols].values.astype(np.float32)
    n = min(len(image_paths), len(mask_paths), len(raw_q))

    image_paths = np.array(image_paths[:n])
    mask_paths = np.array(mask_paths[:n])
    raw_q = raw_q[:n]
    df_used = df.iloc[:n].copy().reset_index(drop=True)

    valid = []
    quaternions = []
    for q in raw_q:
        qn = normalize_quaternion_np(q)
        ok = qn is not None and np.isfinite(qn).all()
        valid.append(ok)
        if ok:
            quaternions.append(qn)

    valid = np.array(valid, dtype=bool)
    image_paths = image_paths[valid]
    mask_paths = mask_paths[valid]
    quaternions = np.array(quaternions, dtype=np.float32)
    df_used = df_used.loc[valid].reset_index(drop=True)

    print("=== Dataset summary ===")
    print("Images total      :", len(list_files(image_dir, (".jpg", ".jpeg", ".png", ".bmp"))))
    print("Masks total       :", len(list_files(mask_dir, (".jpg", ".jpeg", ".png", ".bmp"))))
    print("CSV rows          :", len(df))
    print("Used before filter:", n)
    print("Invalid quaternion:", int((~valid).sum()))
    print("Used after filter :", len(quaternions))

    if len(quaternions) == 0:
        raise ValueError("Tidak ada quaternion valid.")

    return image_paths, mask_paths, quaternions, df_used


def get_bbox_from_mask_resized(mask_bin, margin=8, min_area=30):
    """Return normalized box [y1, x1, y2, x2]."""
    H, W = mask_bin.shape[:2]
    ys, xs = np.where(mask_bin > 0)

    if len(xs) < min_area:
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(W - 1, int(xs.max()) + margin)
    y2 = min(H - 1, int(ys.max()) + margin)

    if x2 <= x1 or y2 <= y1:
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    box = np.array([
        y1 / float(H - 1),
        x1 / float(W - 1),
        y2 / float(H - 1),
        x2 / float(W - 1),
    ], dtype=np.float32)

    return np.clip(box, 0.0, 1.0).astype(np.float32)


def preprocess_sample_np(image_path, mask_path, image_size=(224, 224), bbox_margin=8, min_mask_area=30):
    if hasattr(image_path, "numpy"):
        image_path = image_path.numpy()
    if hasattr(mask_path, "numpy"):
        mask_path = mask_path.numpy()

    image_path = image_path.decode("utf-8") if isinstance(image_path, (bytes, bytearray)) else str(image_path)
    mask_path = mask_path.decode("utf-8") if isinstance(mask_path, (bytes, bytearray)) else str(mask_path)

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        raise ValueError(f"Gagal membaca image: {image_path}")
    if mask is None:
        raise ValueError(f"Gagal membaca mask: {mask_path}")

    target_h, target_w = image_size

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask_resized > 127).astype(np.uint8)

    roi_box = get_bbox_from_mask_resized(mask_bin, margin=bbox_margin, min_area=min_mask_area)
    image_vgg = preprocess_input(image_rgb.astype(np.float32))

    if not np.isfinite(image_vgg).all():
        raise ValueError(f"Image contains non-finite values: {image_path}")
    if not np.isfinite(roi_box).all():
        roi_box = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    return image_vgg.astype(np.float32), roi_box.astype(np.float32)


def reverse_vgg_preprocess(x):
    x = x.copy()
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    x = x[..., ::-1]
    return np.clip(x, 0, 255).astype(np.uint8)


def draw_normalized_bbox(image_rgb, box, color=(255, 255, 0)):
    img = image_rgb.copy()
    H, W = img.shape[:2]
    y1, x1, y2, x2 = box
    x1 = int(round(float(x1) * (W - 1)))
    x2 = int(round(float(x2) * (W - 1)))
    y1 = int(round(float(y1) * (H - 1)))
    y2 = int(round(float(y2) * (H - 1)))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


def write_summary(output_dir, metrics):
    output_dir = Path(output_dir)
    txt_path = output_dir / "rotation_eval_summary.txt"
    json_path = output_dir / "rotation_eval_summary.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved summary:", txt_path)
    print("Saved summary:", json_path)


def run(args):
    configure_environment(args)

    global tf, layers, preprocess_input
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.applications.vgg16 import preprocess_input

    tf.keras.backend.clear_session()
    gc.collect()

    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow version:", tf.__version__)
    print("Visible GPU:", gpus)
    print("Visible CPU:", tf.config.list_physical_devices("CPU"))

    if args.device != "cpu":
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print("Memory growth warning:", e)

    @tf.keras.utils.register_keras_serializable(package="PoseCNN")
    def safe_l2_normalize(x, axis=-1, eps=1e-6):
        x = tf.cast(x, tf.float32)
        square_sum = tf.reduce_sum(tf.square(x), axis=axis, keepdims=True)
        denom = tf.sqrt(tf.maximum(square_sum, eps))
        return x / denom

    @tf.keras.utils.register_keras_serializable(package="PoseCNN")
    def quaternion_cosine_loss(y_true, y_pred):
        y_true = safe_l2_normalize(y_true, axis=-1)
        y_pred = safe_l2_normalize(y_pred, axis=-1)
        dot = tf.reduce_sum(y_true * y_pred, axis=-1)
        dot = tf.clip_by_value(tf.abs(dot), 0.0, 1.0)
        loss = 1.0 - dot
        loss = tf.where(tf.math.is_finite(loss), loss, tf.ones_like(loss))
        return tf.reduce_mean(loss)

    @tf.keras.utils.register_keras_serializable(package="PoseCNN")
    def quaternion_angle_error_deg(y_true, y_pred):
        y_true = safe_l2_normalize(y_true, axis=-1)
        y_pred = safe_l2_normalize(y_pred, axis=-1)
        dot = tf.reduce_sum(y_true * y_pred, axis=-1)
        dot = tf.clip_by_value(tf.abs(dot), 0.0, 1.0)
        angle_rad = 2.0 * tf.acos(dot)
        angle_deg = angle_rad * 180.0 / np.pi
        angle_deg = tf.where(tf.math.is_finite(angle_deg), angle_deg, tf.zeros_like(angle_deg))
        return tf.reduce_mean(angle_deg)

    @tf.keras.utils.register_keras_serializable(package="PoseCNN")
    class SafeL2NormalizeLayer(layers.Layer):
        def __init__(self, axis=-1, eps=1e-6, **kwargs):
            super().__init__(**kwargs)
            self.axis = axis
            self.eps = eps

        def call(self, inputs):
            square_sum = tf.reduce_sum(tf.square(inputs), axis=self.axis, keepdims=True)
            denom = tf.sqrt(tf.maximum(square_sum, self.eps))
            return inputs / denom

        def get_config(self):
            config = super().get_config()
            config.update({
                "axis": self.axis,
                "eps": self.eps
            })
            return config
        
    @tf.keras.utils.register_keras_serializable(package="PoseCNN")
    class RoiPoolingLayer(layers.Layer):
        def __init__(self, pool_size=(7, 7), eps=1e-6, **kwargs):
            super().__init__(**kwargs)
            self.pool_size = tuple(pool_size)
            self.eps = eps

        def call(self, inputs):
            feature_map, boxes = inputs
            boxes = tf.cast(boxes, tf.float32)
            boxes = tf.reshape(boxes, [-1, 4])

            feature_shape = tf.shape(feature_map)
            batch_size = feature_shape[0]
            feat_h = tf.cast(feature_shape[1], tf.float32)
            feat_w = tf.cast(feature_shape[2], tf.float32)

            # safe boxes: [y1, x1, y2, x2], normalized 0-1
            default_boxes = tf.tile(
                tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=tf.float32),
                [tf.shape(boxes)[0], 1]
            )

            is_valid = tf.reduce_all(tf.math.is_finite(boxes), axis=-1, keepdims=True)
            boxes = tf.where(is_valid, boxes, default_boxes)

            y1, x1, y2, x2 = tf.split(boxes, 4, axis=-1)

            ya = tf.minimum(y1, y2)
            yb = tf.maximum(y1, y2)
            xa = tf.minimum(x1, x2)
            xb = tf.maximum(x1, x2)

            ya = tf.clip_by_value(ya, 0.0, 1.0)
            xa = tf.clip_by_value(xa, 0.0, 1.0)
            yb = tf.clip_by_value(yb, 0.0, 1.0)
            xb = tf.clip_by_value(xb, 0.0, 1.0)

            yb = tf.maximum(yb, ya + self.eps)
            xb = tf.maximum(xb, xa + self.eps)

            boxes = tf.concat([ya, xa, yb, xb], axis=-1)

            def pool_single(args):
                fmap, box = args

                y1, x1, y2, x2 = tf.unstack(box)

                # normalized box -> feature map coordinate
                y1 = y1 * (feat_h - 1.0)
                y2 = y2 * (feat_h - 1.0)
                x1 = x1 * (feat_w - 1.0)
                x2 = x2 * (feat_w - 1.0)

                roi_h = tf.maximum(y2 - y1, 1.0)
                roi_w = tf.maximum(x2 - x1, 1.0)

                pooled_rows = []

                for ph in range(self.pool_size[0]):
                    pooled_cols = []

                    for pw in range(self.pool_size[1]):
                        bin_y1 = y1 + tf.cast(ph, tf.float32) * roi_h / self.pool_size[0]
                        bin_y2 = y1 + tf.cast(ph + 1, tf.float32) * roi_h / self.pool_size[0]
                        bin_x1 = x1 + tf.cast(pw, tf.float32) * roi_w / self.pool_size[1]
                        bin_x2 = x1 + tf.cast(pw + 1, tf.float32) * roi_w / self.pool_size[1]

                        y_start = tf.cast(tf.math.floor(bin_y1), tf.int32)
                        y_end   = tf.cast(tf.math.ceil(bin_y2), tf.int32)
                        x_start = tf.cast(tf.math.floor(bin_x1), tf.int32)
                        x_end   = tf.cast(tf.math.ceil(bin_x2), tf.int32)

                        y_start = tf.clip_by_value(y_start, 0, tf.shape(fmap)[0] - 1)
                        y_end   = tf.clip_by_value(y_end, y_start + 1, tf.shape(fmap)[0])
                        x_start = tf.clip_by_value(x_start, 0, tf.shape(fmap)[1] - 1)
                        x_end   = tf.clip_by_value(x_end, x_start + 1, tf.shape(fmap)[1])

                        region = fmap[y_start:y_end, x_start:x_end, :]

                        pooled = tf.reduce_max(region, axis=[0, 1])
                        pooled_cols.append(pooled)

                    pooled_rows.append(tf.stack(pooled_cols, axis=0))

                pooled_feature = tf.stack(pooled_rows, axis=0)

                return pooled_feature

            pooled_features = tf.map_fn(
                pool_single,
                (feature_map, boxes),
                fn_output_signature=tf.float32
            )

            return pooled_features
        
        def get_config(self):
            config = super().get_config()
            config.update({
                "pool_size": self.pool_size,
                "eps": self.eps
            })
            return config
        
    cfg = SimpleNamespace(
        dataset_root=args.dataset_root,
        image_subdir=args.image_subdir,
        mask_subdir=args.mask_subdir,
        summary_csv=args.summary_csv,
        image_size=(args.image_height, args.image_width),
        bbox_margin=args.bbox_margin,
        min_mask_area=args.min_mask_area,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    image_paths, mask_paths, quaternions, df_used = load_paths_labels_dataframe(cfg)

    if args.eval_split in ["train", "val"]:
        idx = np.arange(len(quaternions))
        train_idx, val_idx = train_test_split(
            idx,
            test_size=cfg.val_ratio,
            random_state=cfg.seed,
            shuffle=True,
        )
        chosen = train_idx if args.eval_split == "train" else val_idx
        image_paths = image_paths[chosen]
        mask_paths = mask_paths[chosen]
        quaternions = quaternions[chosen]
        df_used = df_used.iloc[chosen].reset_index(drop=True)
        print(f"Eval split: {args.eval_split}, samples: {len(quaternions)}")
    else:
        print(f"Eval split: all, samples: {len(quaternions)}")

    if args.limit and args.limit > 0:
        image_paths = image_paths[:args.limit]
        mask_paths = mask_paths[:args.limit]
        quaternions = quaternions[:args.limit]
        df_used = df_used.iloc[:args.limit].reset_index(drop=True)
        print("Limited samples:", len(quaternions))

    def tf_load_sample(image_path, mask_path, quaternion):
        image, roi_box = tf.py_function(
            func=lambda img_p, msk_p: preprocess_sample_np(
                img_p,
                msk_p,
                image_size=cfg.image_size,
                bbox_margin=cfg.bbox_margin,
                min_mask_area=cfg.min_mask_area,
            ),
            inp=[image_path, mask_path],
            Tout=[tf.float32, tf.float32],
        )
        image.set_shape([cfg.image_size[0], cfg.image_size[1], 3])
        roi_box.set_shape([4])
        quaternion = tf.cast(quaternion, tf.float32)
        quaternion = safe_l2_normalize(quaternion, axis=-1)
        return {"image": image, "roi_box": roi_box}, quaternion

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths, quaternions))
    workers = args.tf_data_workers if args.tf_data_workers > 0 else tf.data.AUTOTUNE
    ds = ds.map(tf_load_sample, num_parallel_calls=workers)
    ds = ds.batch(cfg.batch_size, drop_remainder=False)
    if args.prefetch > 0:
        ds = ds.prefetch(args.prefetch)

    # Sanity check one batch.
    for batch_inputs, batch_q in ds.take(1):
        print("Image batch:", batch_inputs["image"].shape)
        print("ROI batch  :", batch_inputs["roi_box"].shape)
        print("Q batch    :", batch_q.shape)
        print("ROI min/max:", float(tf.reduce_min(batch_inputs["roi_box"])), float(tf.reduce_max(batch_inputs["roi_box"])))
        print("Q finite   :", bool(tf.reduce_all(tf.math.is_finite(batch_q))))
        print("ROI finite :", bool(tf.reduce_all(tf.math.is_finite(batch_inputs["roi_box"]))))
        print("Image finite:", bool(tf.reduce_all(tf.math.is_finite(batch_inputs["image"]))))

    custom_objects = {
        "RoiPoolingLayer": RoiPoolingLayer,
        "SafeL2NormalizeLayer": SafeL2NormalizeLayer,
        "quaternion_cosine_loss": quaternion_cosine_loss,
        "quaternion_angle_error_deg": quaternion_angle_error_deg,
    }

    try:
        model = tf.keras.models.load_model(
            args.model,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        model = tf.keras.models.load_model(
            args.model,
            custom_objects=custom_objects,
            compile=False,
        )

    print("Loaded model:", args.model)

    for batch_inputs, batch_q in ds.take(1):
        pred = model(batch_inputs, training=False)
        print("Pred shape :", pred.shape)
        print("Pred finite:", bool(tf.reduce_all(tf.math.is_finite(pred))))
        print("Pred sample:", pred[0].numpy())
        angle0, dot0 = quaternion_angle_error_deg_np(batch_q[0].numpy(), pred[0].numpy())
        print("First sample angle error:", angle0, "deg, dot:", dot0)
        if args.dry_run:
            print("Dry run selesai.")
            return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    q_pred_all = []
    roi_all = []

    # Predict batch-by-batch. Also collect ROI boxes from dataset order.
    for batch_inputs, _ in ds:
        pred = model.predict(batch_inputs, verbose=0)
        q_pred_all.append(pred.astype(np.float32))
        roi_all.append(batch_inputs["roi_box"].numpy().astype(np.float32))

    q_pred_all = np.concatenate(q_pred_all, axis=0)
    roi_all = np.concatenate(roi_all, axis=0)

    rows = []
    angles = []
    dots = []

    for i in range(len(q_pred_all)):
        q_true = quaternions[i]
        q_pred = q_pred_all[i]
        angle, dot = quaternion_angle_error_deg_np(q_true, q_pred)
        angles.append(angle)
        dots.append(dot)

        row = {
            "index": i,
            "image_path": image_paths[i],
            "mask_path": mask_paths[i],
            "image_name": os.path.basename(image_paths[i]),
            "mask_name": os.path.basename(mask_paths[i]),
            "roi_y1": float(roi_all[i, 0]),
            "roi_x1": float(roi_all[i, 1]),
            "roi_y2": float(roi_all[i, 2]),
            "roi_x2": float(roi_all[i, 3]),
            "q_true_w": float(q_true[0]),
            "q_true_x": float(q_true[1]),
            "q_true_y": float(q_true[2]),
            "q_true_z": float(q_true[3]),
            "q_pred_w": float(q_pred[0]),
            "q_pred_x": float(q_pred[1]),
            "q_pred_y": float(q_pred[2]),
            "q_pred_z": float(q_pred[3]),
            "dot_abs": float(dot) if np.isfinite(dot) else np.nan,
            "angle_error_deg": float(angle) if np.isfinite(angle) else np.nan,
            "pred_finite": bool(np.isfinite(q_pred).all()),
        }
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    csv_path = output_dir / "rotation_eval_predictions.csv"
    pred_df.to_csv(csv_path, index=False)
    print("Saved predictions:", csv_path)

    angles_np = np.asarray(angles, dtype=np.float32)
    valid_angles = angles_np[np.isfinite(angles_np)]

    metrics = {
        "model": str(args.model),
        "dataset_root": str(args.dataset_root),
        "eval_split": args.eval_split,
        "num_samples": int(len(angles_np)),
        "num_valid_angles": int(len(valid_angles)),
        "mean_angle_deg": float(np.mean(valid_angles)) if len(valid_angles) else None,
        "median_angle_deg": float(np.median(valid_angles)) if len(valid_angles) else None,
        "std_angle_deg": float(np.std(valid_angles)) if len(valid_angles) else None,
        "min_angle_deg": float(np.min(valid_angles)) if len(valid_angles) else None,
        "max_angle_deg": float(np.max(valid_angles)) if len(valid_angles) else None,
        "p90_angle_deg": float(np.percentile(valid_angles, 90)) if len(valid_angles) else None,
        "p95_angle_deg": float(np.percentile(valid_angles, 95)) if len(valid_angles) else None,
        "nan_prediction_count": int((~np.isfinite(q_pred_all).all(axis=1)).sum()),
    }
    write_summary(output_dir, metrics)

    print("=== Rotation evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.save_vis:
        count = min(args.vis_count, len(pred_df))
        for i in range(count):
            image_vgg, roi_box = preprocess_sample_np(
                image_paths[i],
                mask_paths[i],
                image_size=cfg.image_size,
                bbox_margin=cfg.bbox_margin,
                min_mask_area=cfg.min_mask_area,
            )
            img_rgb = reverse_vgg_preprocess(image_vgg)
            img_box = draw_normalized_bbox(img_rgb, roi_box)
            angle = pred_df.loc[i, "angle_error_deg"]
            q_t = quaternions[i]
            q_p = q_pred_all[i]

            # OpenCV writes BGR.
            img_bgr = cv2.cvtColor(img_box, cv2.COLOR_RGB2BGR)
            text1 = f"angle error: {angle:.2f} deg" if np.isfinite(angle) else "angle: NaN"
            text2 = f"GT: [{q_t[0]:.2f},{q_t[1]:.2f},{q_t[2]:.2f},{q_t[3]:.2f}]"
            text3 = f"PR: [{q_p[0]:.2f},{q_p[1]:.2f},{q_p[2]:.2f},{q_p[3]:.2f}]"
            cv2.putText(img_bgr, text1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_bgr, text2, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_bgr, text3, (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
            out_path = vis_dir / f"eval_{i:05d}_{angle:.2f}deg.jpg" if np.isfinite(angle) else vis_dir / f"eval_{i:05d}_nan.jpg"
            cv2.imwrite(str(out_path), img_bgr)
        print("Saved visualizations:", vis_dir)


if __name__ == "__main__":
    args = parse_args()
    run(args)
