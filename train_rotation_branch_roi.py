#!/usr/bin/env python3
"""
Train PoseCNN-style rotation branch with ROI Align on VGG feature maps.

Pipeline:
  full image -> VGG16 encoder -> feature map -> ROI Align from mask bbox -> FC -> quaternion

This script includes safeguards against loss=NaN:
  - filters invalid quaternion labels
  - safe quaternion normalization with epsilon
  - safe ROI boxes and fallback full-image ROI
  - low default learning rate
  - gradient clipping
  - TerminateOnNaN
  - optional CPU-only mode
"""

import os
import re
import gc
import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser(description="Train PoseCNN-style rotation branch with safe ROI Align.")

    p.add_argument("--dataset-root", type=str, required=True, help="Path to your dataset folder, must be defined")
    p.add_argument("--image-subdir", type=str, default="img_left")
    p.add_argument("--mask-subdir", type=str, default="semantic")
    p.add_argument("--summary-csv", type=str, default="pseudo_ground_truth_summary.csv")

    p.add_argument("--image-height", type=int, default=224)
    p.add_argument("--image-width", type=int, default=224)
    p.add_argument("--roi-height", type=int, default=7)
    p.add_argument("--roi-width", type=int, default=7)

    p.add_argument("--feature-layer", type=str, default="block5_conv3",
                   choices=["block4_conv3", "block5_conv3", "block5_pool"])

    p.add_argument("--fc-dim", type=int, default=512,
                   help="Use 512/1024 for small GPU. Use 4096 for PoseCNN-like FC on CPU/large RAM.")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--train-encoder", action="store_true")

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--bbox-margin", type=int, default=8)
    p.add_argument("--min-mask-area", type=int, default=30)

    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--clipnorm", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=1e-6)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--cuda-malloc-async", action="store_true")
    p.add_argument("--check-numerics", action="store_true")
    p.add_argument("--tf-data-workers", type=int, default=2)
    p.add_argument("--prefetch", type=int, default=1)

    p.add_argument("--save-model", type=str, default="rotation_branch_roi_align.keras")
    p.add_argument("--output-dir", type=str, default="rotation_training_output")
    p.add_argument("--dry-run", action="store_true", help="Validate data/model forward pass only; do not train.")

    return p.parse_args()


def configure_environment(args):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if args.device == "cpu":
        # Must be set before importing TensorFlow.
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


def load_paths_and_labels(cfg):
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

    valid = []
    quaternions = []
    for i, q in enumerate(raw_q):
        qn = normalize_quaternion_np(q)
        ok = qn is not None and np.isfinite(qn).all()
        valid.append(ok)
        if ok:
            quaternions.append(qn)

    valid = np.array(valid, dtype=bool)
    image_paths = image_paths[valid]
    mask_paths = mask_paths[valid]
    quaternions = np.array(quaternions, dtype=np.float32)

    print("=== Dataset summary ===")
    print("Images total      :", len(list_files(image_dir, (".jpg", ".jpeg", ".png", ".bmp"))))
    print("Masks total       :", len(list_files(mask_dir, (".jpg", ".jpeg", ".png", ".bmp"))))
    print("CSV rows          :", len(df))
    print("Used before filter:", n)
    print("Invalid quaternion:", int((~valid).sum()))
    print("Used after filter :", len(quaternions))

    if len(quaternions) == 0:
        raise ValueError("Tidak ada quaternion valid.")

    return image_paths, mask_paths, quaternions


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

    box = np.clip(box, 0.0, 1.0)
    return box


def preprocess_sample_np(image_path, mask_path, image_size=(224, 224), bbox_margin=8, min_mask_area=30):
    # TensorFlow tf.py_function can pass EagerTensor/bytes.
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

    roi_box = get_bbox_from_mask_resized(
        mask_bin,
        margin=bbox_margin,
        min_area=min_mask_area,
    )

    # VGG16 caffe preprocess expects RGB input and converts internally to BGR-like mean subtraction.
    # We import preprocess_input inside run() and attach to global namespace.
    image_vgg = preprocess_input(image_rgb.astype(np.float32))

    if not np.isfinite(image_vgg).all():
        raise ValueError(f"Image contains non-finite values: {image_path}")
    if not np.isfinite(roi_box).all():
        roi_box = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    return image_vgg.astype(np.float32), roi_box.astype(np.float32)


def run(args):
    configure_environment(args)

    global tf, layers, Model, VGG16, preprocess_input
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input

    tf.keras.backend.clear_session()
    gc.collect()

    if args.check_numerics:
        tf.debugging.enable_check_numerics()

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

    cfg = SimpleNamespace(
        dataset_root=args.dataset_root,
        image_subdir=args.image_subdir,
        mask_subdir=args.mask_subdir,
        summary_csv=args.summary_csv,
        image_size=(args.image_height, args.image_width),
        roi_size=(args.roi_height, args.roi_width),
        bbox_margin=args.bbox_margin,
        min_mask_area=args.min_mask_area,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    image_paths, mask_paths, quaternions = load_paths_and_labels(cfg)
    
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
    def safe_l2_normalize(x, axis=-1, eps=args.epsilon):
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
        angle_deg = tf.where(tf.math.is_finite(angle_deg), angle_deg, tf.ones_like(angle_deg) * 180.0)
        return tf.reduce_mean(angle_deg)

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

    def make_dataset(paths_img, paths_mask, qs, training=True):
        ds = tf.data.Dataset.from_tensor_slices((paths_img, paths_mask, qs))
        if training:
            ds = ds.shuffle(buffer_size=len(paths_img), seed=cfg.seed, reshuffle_each_iteration=True)
        workers = args.tf_data_workers if args.tf_data_workers > 0 else tf.data.AUTOTUNE
        ds = ds.map(tf_load_sample, num_parallel_calls=workers)
        ds = ds.batch(cfg.batch_size, drop_remainder=False)
        if args.prefetch > 0:
            ds = ds.prefetch(args.prefetch)
        return ds

    train_img, val_img, train_mask, val_mask, train_q, val_q = train_test_split(
        image_paths,
        mask_paths,
        quaternions,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        shuffle=True,
    )

    train_ds = make_dataset(train_img, train_mask, train_q, training=True)
    val_ds = make_dataset(val_img, val_mask, val_q, training=False)

    print("Train samples:", len(train_q))
    print("Val samples  :", len(val_q))
    print("Batch size   :", cfg.batch_size)

    # Data sanity check.
    for batch_inputs, batch_q in train_ds.take(1):
        print("Image batch:", batch_inputs["image"].shape)
        print("ROI batch  :", batch_inputs["roi_box"].shape)
        print("Q batch    :", batch_q.shape)
        print("ROI min/max:", float(tf.reduce_min(batch_inputs["roi_box"])), float(tf.reduce_max(batch_inputs["roi_box"])))
        print("Q finite   :", bool(tf.reduce_all(tf.math.is_finite(batch_q))))
        print("ROI finite :", bool(tf.reduce_all(tf.math.is_finite(batch_inputs["roi_box"]))))
        print("Image finite:", bool(tf.reduce_all(tf.math.is_finite(batch_inputs["image"]))))

    def create_model():
        image_input = layers.Input(shape=(cfg.image_size[0], cfg.image_size[1], 3), name="image")
        roi_box_input = layers.Input(shape=(4,), name="roi_box")

        encoder = VGG16(weights="imagenet", include_top=False, input_tensor=image_input)
        for layer in encoder.layers:
            layer.trainable = bool(args.train_encoder)

        feature_map1 = encoder.get_layer('block5_conv3').output
        feature_map2 = encoder.get_layer('block4_conv3').output

        roi_feature1 = RoiPoolingLayer(name="roi_pooling1")([feature_map1, roi_box_input])
        # roi_feature2 = RoiPoolingLayer(name="roi_pooling2")([feature_map2, roi_box_input])
        # roi_feature  = layers.Add(name="roi_feature_add")([roi_feature1, roi_feature2])

        x = layers.Flatten(name="roi_flatten")(roi_feature1)

        x = layers.Dense(args.fc_dim, activation="relu", name="fc6")(x)
        x = layers.Dropout(args.dropout, name="drop6")(x)
        x = layers.Dense(args.fc_dim, activation="relu", name="fc7")(x)
        x = layers.Dropout(args.dropout, name="drop7")(x)

        q_raw = layers.Dense(
            4,
            activation="linear",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-4),
            bias_initializer=tf.keras.initializers.Constant([1.0, 0.0, 0.0, 0.0]),
            name="poses_pred_unnormalized",
        )(x)

        q_tanh = layers.Activation("tanh", name="poses_tanh")(q_raw)
        # q_norm = layers.Lambda(lambda t: safe_l2_normalize(t, axis=-1), name="poses_pred", output_shape=(4,))(q_tanh)
        q_norm = SafeL2NormalizeLayer(axis=-1, eps=1e-6, name='poses_pred')(q_tanh)

        model = Model(inputs={"image": image_input, "roi_box": roi_box_input}, outputs=q_norm,
                      name="PoseCNN_Style_Rotation_Branch")
        return model

    model = create_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.clipnorm)
    model.compile(
        optimizer=optimizer,
        loss=quaternion_cosine_loss,
        metrics=[quaternion_angle_error_deg],
        jit_compile=False,
    )

    model.summary()

    # Forward sanity check before training.
    for batch_inputs, batch_q in train_ds.take(1):
        pred = model(batch_inputs, training=False)
        print("Pred shape :", pred.shape)
        print("Pred finite:", bool(tf.reduce_all(tf.math.is_finite(pred))))
        print("Pred sample:", pred[0].numpy())
        loss_val = quaternion_cosine_loss(batch_q, pred)
        print("Initial loss:", float(loss_val.numpy()))
        if not np.isfinite(float(loss_val.numpy())):
            raise RuntimeError("Initial loss is not finite. Check labels, ROI boxes, or model output.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / args.save_model),
            monitor="val_quaternion_angle_error_deg",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_quaternion_angle_error_deg",
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        )
    ]

    if args.dry_run:
        print("Dry run selesai. Training tidak dijalankan.")
        return

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    final_path = output_dir / (Path(args.save_model).stem + "_last.keras")
    model.save(str(final_path))
    print("Saved last model:", final_path)
    print("Saved best model:", output_dir / args.save_model)


if __name__ == "__main__":
    args = parse_args()
    run(args)
