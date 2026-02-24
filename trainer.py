"""
trainer.py
----------
Two-phase training loop:
  Phase 1 — Frozen backbones, train classification head.
  Phase 2 — Unfreeze top layers of each backbone for fine-tuning.
"""

from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam


def get_callbacks(checkpoint_path: str = "models/best_model.h5"):
    return [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1),
    ]


def train_phase1(model, train_data, val_data, epochs: int = 15, checkpoint_path: str = "models/best_model.h5"):
    """Phase 1: frozen backbones — train head only."""
    print("\n🚀 PHASE 1: Training with frozen backbones...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=get_callbacks(checkpoint_path),
    )
    return history


def train_phase2(model, train_data, val_data, epochs: int = 10, lr: float = 1e-4, checkpoint_path: str = "models/best_model.h5"):
    """Phase 2: fine-tune top layers of each backbone."""
    print("\n🔓 PHASE 2: Fine-tuning top backbone layers...")

    # Unfreeze last 20 layers of every sub-model in the ensemble
    for layer in model.layers:
        if hasattr(layer, "layers"):          # it's a sub-model (backbone)
            for sub_layer in layer.layers[-20:]:
                sub_layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=get_callbacks(checkpoint_path),
    )
    return history


def train_model(model, train_data, val_data,
                epochs_phase1: int = 15,
                epochs_phase2: int = 10,
                checkpoint_path: str = "models/best_model.h5"):
    """Run both training phases and return combined histories."""
    h1 = train_phase1(model, train_data, val_data, epochs_phase1, checkpoint_path)
    h2 = train_phase2(model, train_data, val_data, epochs_phase2, checkpoint_path=checkpoint_path)
    return h1, h2
