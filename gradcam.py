"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization
for explainability of model predictions.

Reference: Selvaraju et al. (2017) — https://arxiv.org/abs/1610.02391
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: Model,
    last_conv_layer_name: str,
    pred_index: int = None,
) -> np.ndarray:
    """
    Compute the Grad-CAM heatmap for a single image.

    Args:
        img_array:            Preprocessed image array, shape (1, H, W, 3).
        model:                Trained Keras model.
        last_conv_layer_name: Name of the target convolutional layer.
        pred_index:           Class index to explain (None = top prediction).

    Returns:
        2-D heatmap as a float32 numpy array in [0, 1].
    """
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_array: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Superimpose a Grad-CAM heatmap on the original image.

    Args:
        img_array: Original image as float32 array in [0, 1], shape (H, W, 3).
        heatmap:   Grad-CAM heatmap in [0, 1].
        alpha:     Heatmap opacity.

    Returns:
        Blended image as float32 array in [0, 1].
    """
    h, w = img_array.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    superimposed = alpha * heatmap_color + img_array
    return np.clip(superimposed, 0, 1)


def display_gradcam(
    img_path: str,
    model: Model,
    class_names: list,
    img_size: int = 224,
    last_conv_layer_name: str = "top_conv",
    save_path: str = None,
):
    """
    Load an image, predict, generate Grad-CAM, and display results.

    Args:
        img_path:             Path to the MRI image.
        model:                Trained ensemble model.
        class_names:          List of class label strings.
        img_size:             Target image resolution.
        last_conv_layer_name: Convolutional layer for Grad-CAM.
        save_path:            If provided, saves the figure to this path.

    Returns:
        (pred_class, confidence): Predicted class name and confidence %.
    """
    img = load_img(img_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_exp = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_exp, verbose=0)
    pred_class = class_names[np.argmax(pred)]
    confidence = float(np.max(pred)) * 100

    try:
        heatmap = make_gradcam_heatmap(img_exp, model, last_conv_layer_name)
        overlay = overlay_gradcam(img_array, heatmap)
    except Exception as e:
        print(f"  Grad-CAM fallback (layer not found): {e}")
        overlay = img_array

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(img_array)
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Overlay")
    axes[1].axis("off")

    axes[2].barh(class_names, pred[0] * 100, color="steelblue")
    axes[2].set_xlabel("Confidence (%)")
    axes[2].set_title(f"Prediction: {pred_class}\n({confidence:.1f}%)")
    axes[2].set_xlim(0, 100)

    plt.suptitle("Multi-Stage Brain Tumor Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")

    plt.show()
    return pred_class, confidence
