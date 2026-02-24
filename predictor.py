"""
predictor.py
------------
Single-image prediction with handcrafted feature analysis and Grad-CAM.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from .feature_extractor import extract_handcrafted_features, FEATURE_NAMES
from .gradcam import display_gradcam


def predict_single_image(
    model,
    image_path: str,
    class_names: list,
    img_size: int = 224,
    last_conv_layer: str = "top_conv",
    save_dir: str = "results/plots",
) -> dict:
    """
    Run full multi-stage prediction on a single MRI image.

    Args:
        model:            Loaded Keras model.
        image_path:       Path to MRI image.
        class_names:      List of class label strings.
        img_size:         Target resolution.
        last_conv_layer:  Layer name for Grad-CAM.
        save_dir:         Where to save output figures.

    Returns:
        dict with keys: class, confidence, probabilities, handcrafted_features
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # ── Deep learning prediction + Grad-CAM ─────────────────────────
    pred_class, confidence = display_gradcam(
        img_path=image_path,
        model=model,
        class_names=class_names,
        img_size=img_size,
        last_conv_layer_name=last_conv_layer,
        save_path=f"{save_dir}/gradcam_result.png",
    )

    # Full probability vector
    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    probs = model.predict(img_array, verbose=0)[0]

    # ── Handcrafted feature analysis ─────────────────────────────────
    hc_feats = extract_handcrafted_features(image_path)

    print(f"\n🔬 Deep Learning Prediction : {pred_class} ({confidence:.1f}%)")
    print("\n🧪 Handcrafted Features:")
    for name, val in zip(FEATURE_NAMES, hc_feats):
        print(f"  {name:25s}: {val:.4f}")

    # Feature bar chart
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(hc_feats)), np.abs(hc_feats), color="teal", alpha=0.7)
    plt.title("Handcrafted Feature Vector")
    plt.xlabel("Feature Index")
    plt.ylabel("|Value|")
    plt.tight_layout()
    feat_path = f"{save_dir}/handcrafted_features.png"
    plt.savefig(feat_path, dpi=150)
    plt.show()
    print(f"  Saved: {feat_path}")

    return {
        "class": pred_class,
        "confidence": confidence,
        "probabilities": {cls: float(p) for cls, p in zip(class_names, probs)},
        "handcrafted_features": dict(zip(FEATURE_NAMES, hc_feats.tolist())),
    }
