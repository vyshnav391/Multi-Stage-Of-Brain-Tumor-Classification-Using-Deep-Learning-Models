"""
data_pipeline.py
----------------
Multi-level augmentation pipeline and Keras data generators.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(
    dataset_path: str,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    Build train/validation ImageDataGenerators with augmentation.

    Args:
        dataset_path: Root directory containing one subfolder per class.
        img_size:     Target image dimension (square).
        batch_size:   Mini-batch size.
        val_split:    Fraction of data reserved for validation.
        seed:         Random seed for reproducibility.

    Returns:
        (train_data, val_data): Keras DirectoryIterators.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
        rotation_range=20,
        zoom_range=0.25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
    )

    shared_kwargs = dict(
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        seed=seed,
    )

    train_data = train_datagen.flow_from_directory(
        dataset_path, subset="training", shuffle=True, **shared_kwargs
    )
    val_data = val_datagen.flow_from_directory(
        dataset_path, subset="validation", shuffle=False, **shared_kwargs
    )

    return train_data, val_data
