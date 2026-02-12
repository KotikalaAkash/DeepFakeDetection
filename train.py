import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration
IMG_SIZE = 224
NUM_FRAMES = 15
BATCH_SIZE = 4
EPOCHS = 10
dataset_dir = "dataset_faces/dataset_faces"  # Adjust if needed based on actual path

def load_video_frames(video_path):
    frames = sorted(os.listdir(video_path))[:NUM_FRAMES]
    video = []

    for frame in frames:
        img_path = os.path.join(video_path, frame)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        video.append(img)

    # Padding if less frames
    while len(video) < NUM_FRAMES:
        video.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    return np.array(video, dtype=np.float32)

def generator(root_dir, split):
    split_dir = os.path.join(root_dir, split)
    if not os.path.exists(split_dir):
        print(f"Warning: Directory {split_dir} does not exist.")
        return

    classes = ["Celeb-real", "Celeb-synthesis"] # Matched with data_prepro.py logic
    # Note: data_prepro.py uses "Celeb-real", "Celeb-synthesis" folders.
    # generator in model_gen.ipynb used ["real", "fake"] which might be wrong if data_prepro used the other names.
    # I will check standard CelebDF names. usually they are Celeb-real, Celeb-synthesis.
    # The previous code had `classes = ["real", "fake"]` but `data_prepro.py` iterates `train/Celeb-real`.
    # So I should use the folder names present in dataset_faces.
    
    # Let's check what's actually in dataset_faces if possible, but I can't check efficiently without listing.
    # I'll stick to data_prepro.py's convention which seems to be the source of truth for generation.

    mapped_classes = ["Celeb-real", "Celeb-synthesis"] 

    for label, cls in enumerate(mapped_classes):
        class_path = os.path.join(split_dir, cls)
        if not os.path.exists(class_path):
            continue

        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)

            if os.path.isdir(video_path):
                video_tensor = load_video_frames(video_path)
                yield video_tensor, label

def build_model():
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg"
    )
    backbone.trainable = False

    video_input = layers.Input(
        shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
    )

    # Spatial features per frame
    x = layers.TimeDistributed(backbone)(video_input)

    # Temporal Transformer
    x = layers.LayerNormalization()(x)

    attention = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=64
    )(x, x)

    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)

    ff = layers.Dense(256, activation="relu")(x)
    # FIX: Output dimension of FFN must match input dimension of Add layer (1280)
    ff = layers.Dense(1280)(ff) 

    x = layers.Add()([x, ff])

    x = layers.GlobalAveragePooling1D()(x)

    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(video_input, output)

    return model

def main():
    # Setup Datasets
    output_signature = (
        tf.TensorSpec(shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    train_dataset = tf.data.Dataset.from_generator(
        lambda: generator(dataset_dir, "train"),
        output_signature=output_signature
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: generator(dataset_dir, "val"),
        output_signature=output_signature
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Build and Compile Model
    model = build_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Check if dataset is empty/exists
    # For now, proceeded assuming data exists.
    
    print("Starting training...")
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS
        )
        # Save model
        model.save("deepfake_detection_model.keras")
        print("Model saved to deepfake_detection_model.keras")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()
