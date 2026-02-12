import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- GPU CONFIGURATION ---
# Attempt to find and use GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✅ GPU DETECTED: {len(physical_devices)} device(s) found.")
    except RuntimeError as e:
        print(e)
else:
    print("❌ NO GPU DETECTED. TensorFlow is using CPU.")
    print("To enable GPU, you must install CUDA Toolkit 11.2 and cuDNN 8.1.")
    print("Download CUDA 11.2: https://developer.nvidia.com/cuda-11-2-2-download-archive")
    print("Download cuDNN 8.1: https://developer.nvidia.com/rdp/cudnn-archive (requires login)")

# --- CONSTANTS ---
IMG_SIZE = 224
NUM_FRAMES = 15
BATCH_SIZE = 4
EPOCHS = 10

# --- DATA GENERATOR ---
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

def generator(root_dir):
    classes = ["Celeb-real", "Celeb-synthesis"]

    for label, cls in enumerate(classes):
        class_path = os.path.join(root_dir, cls)
        if not os.path.exists(class_path):
             continue

        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)

            if os.path.isdir(video_path):
                # Only yield if we have frames
                video_tensor = load_video_frames(video_path)
                yield video_tensor, label

# --- DATASETS ---
# Using prefetch(2) instead of AUTOTUNE to avoid SystemError on Windows/CPU
output_signature = (
    tf.TensorSpec(shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int32)
)

train_dir = r"C:\Users\praveen tamminaina\Downloads\30 DEC TZ\Deepfake\Deepfake-detection\dataset_faces\dataset_faces\train"
val_dir = r"C:\Users\praveen tamminaina\Downloads\30 DEC TZ\Deepfake\Deepfake-detection\dataset_faces\dataset_faces\val"
test_dir = r"C:\Users\praveen tamminaina\Downloads\30 DEC TZ\Deepfake\Deepfake-detection\dataset_faces\dataset_faces\test"

train_dataset = tf.data.Dataset.from_generator(
    lambda: generator(train_dir),
    output_signature=output_signature
).batch(BATCH_SIZE).repeat().prefetch(2)

val_dataset = tf.data.Dataset.from_generator(
    lambda: generator(val_dir),
    output_signature=output_signature
).batch(BATCH_SIZE).prefetch(2)

test_dataset = tf.data.Dataset.from_generator(
    lambda: generator(test_dir),
    output_signature=output_signature
).batch(BATCH_SIZE).prefetch(2)


# --- MODEL DEFINITION ---
class PositionalEmbedding(layers.Layer):
    def __init__(self, num_frames, embed_dim):
        super().__init__()
        self.pos_emb = layers.Embedding(input_dim=num_frames, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        position_embeddings = self.pos_emb(positions)
        return x + position_embeddings

def build_model():
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg"
    )
    backbone.trainable = False

    video_input = layers.Input(shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    
    x = layers.TimeDistributed(backbone)(video_input)
    
    # Optional: Add PositionalEmbedding if defined in your notebook logic
    # x = PositionalEmbedding(NUM_FRAMES, 1280)(x) 

    # Transformer block (Simplified for testing)
    x1 = layers.LayerNormalization()(x)
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x1, x1)
    x2 = layers.Add()([x, attention])
    x3 = layers.LayerNormalization()(x2)
    ff = layers.Dense(512, activation="relu")(x3)
    ff = layers.Dense(1280)(ff)
    x4 = layers.Add()([x2, ff])
    
    x = layers.GlobalAveragePooling1D()(x4)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(video_input, output)
    return model

model = build_model()
model.summary()

# --- TRAINING ---
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Starting training...")
TRAIN_SAMPLES = 4570
VAL_SAMPLES = 978

try:
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=TRAIN_SAMPLES // BATCH_SIZE,
        validation_steps=VAL_SAMPLES // BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2),
            tf.keras.callbacks.ModelCheckpoint("best_vit_model.h5", save_best_only=True)
        ]
    )
    print("Training complete.")
except Exception as e:
    print(f"Training crashed: {e}")
