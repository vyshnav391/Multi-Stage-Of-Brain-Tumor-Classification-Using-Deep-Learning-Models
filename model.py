"""
model.py
--------
Builds the multi-backbone ensemble model with attention and feature fusion.

Backbones:
  - EfficientNetB0  (lightweight, high accuracy)
  - ResNet50V2      (deep residual features)
  - VGG16           (classical texture features)
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D, Dropout,
    BatchNormalization, Concatenate,
)
from tensorflow.keras.optimizers import Adam

from .attention import cbam_block


def _build_branch(backbone_fn, input_tensor, freeze: bool = True):
    """
    Wraps a Keras backbone with CBAM attention and pooling.

    Args:
        backbone_fn:    Keras application function (e.g. EfficientNetB0).
        input_tensor:   Shared Keras Input tensor.
        freeze:         Whether to freeze all backbone weights.

    Returns:
        (feature_tensor, backbone_model)
    """
    base = backbone_fn(weights="imagenet", include_top=False, input_tensor=input_tensor)
    for layer in base.layers:
        layer.trainable = not freeze

    x = base.output
    x = cbam_block(x)                    # Channel + Spatial attention
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    return x, base


def build_ensemble_model(
    num_classes: int,
    img_size: int = 224,
    freeze_backbones: bool = True,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Construct the full multi-backbone ensemble model.

    Args:
        num_classes:       Number of output classes.
        img_size:          Input image size (square).
        freeze_backbones:  If True, all backbone layers are frozen.
        learning_rate:     Adam learning rate.

    Returns:
        Compiled Keras Model.
    """
    inp = Input(shape=(img_size, img_size, 3), name="input_mri")

    eff_feat, _ = _build_branch(EfficientNetB0, inp, freeze=freeze_backbones)
    res_feat, _ = _build_branch(ResNet50V2,    inp, freeze=freeze_backbones)
    vgg_feat, _ = _build_branch(VGG16,         inp, freeze=freeze_backbones)

    fused = Concatenate(name="feature_fusion")([eff_feat, res_feat, vgg_feat])
    fused = Dense(512, activation="relu")(fused)
    fused = BatchNormalization()(fused)
    fused = Dropout(0.5)(fused)
    fused = Dense(256, activation="relu")(fused)
    fused = BatchNormalization()(fused)
    fused = Dropout(0.3)(fused)
    output = Dense(num_classes, activation="softmax", name="predictions")(fused)

    model = Model(inputs=inp, outputs=output, name="BrainTumor_Ensemble")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def unfreeze_top_layers(model: Model, backbone_names: list, n_layers: int = 20):
    """
    Unfreeze the last n_layers of each named backbone for fine-tuning.

    Args:
        model:          The ensemble model.
        backbone_names: List of backbone sub-model names to target.
        n_layers:       How many layers from the top to unfreeze.
    """
    for name in backbone_names:
        sub = model.get_layer(name)
        for layer in sub.layers[-n_layers:]:
            layer.trainable = True
