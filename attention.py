"""
attention.py
------------
Channel Attention (Squeeze-and-Excitation) and Spatial Attention blocks.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Reshape, Multiply,
    Concatenate, Conv2D, Lambda,
)
import tensorflow.keras.backend as K


def channel_attention(x, ratio: int = 8):
    """
    Squeeze-and-Excitation channel attention.

    Args:
        x:     4-D feature map tensor (B, H, W, C).
        ratio: Reduction ratio for the bottleneck FC layer.

    Returns:
        Recalibrated feature map with same shape as x.
    """
    channel = x.shape[-1]
    avg = GlobalAveragePooling2D()(x)
    avg = Reshape((1, 1, channel))(avg)
    fc1 = Dense(channel // ratio, activation="relu")(avg)
    fc2 = Dense(channel, activation="sigmoid")(fc1)
    return Multiply()([x, fc2])


def spatial_attention(x):
    """
    Spatial attention using channel-wise avg + max pooling.

    Args:
        x: 4-D feature map tensor (B, H, W, C).

    Returns:
        Spatially recalibrated feature map with same shape as x.
    """
    avg_pool = Lambda(lambda t: K.mean(t, axis=-1, keepdims=True))(x)
    max_pool = Lambda(lambda t: K.max(t, axis=-1, keepdims=True))(x)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    spatial = Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)
    return Multiply()([x, spatial])


def cbam_block(x, ratio: int = 8):
    """
    Full CBAM (Convolutional Block Attention Module):
    Channel attention → Spatial attention applied sequentially.

    Args:
        x:     4-D feature map tensor.
        ratio: Channel reduction ratio.

    Returns:
        Attended feature map.
    """
    x = channel_attention(x, ratio=ratio)
    x = spatial_attention(x)
    return x
