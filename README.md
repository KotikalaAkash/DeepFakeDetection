## 📌 Problem Statement

The rapid evolution of deepfake generation techniques, driven by advanced models such as **Generative Adversarial Networks (GANs)** and autoencoders, has made manipulated videos increasingly realistic and difficult to detect. Modern deepfakes exhibit high visual quality, minimizing obvious spatial artifacts such as irregular textures, lighting inconsistencies, and blending errors.

Traditional frame-based detection methods, primarily based on **Convolutional Neural Networks (CNNs)**, focus on identifying spatial inconsistencies within individual frames. However, these approaches often fail to capture **temporal inconsistencies** across video sequences, such as unnatural facial movements, blinking patterns, and lip synchronization errors.

Additionally, many existing models lack transparency in their decision-making process, making it difficult to interpret detection results.

> ⚠️ As a result, there is a growing need for robust deepfake detection systems that effectively analyze both **spatial and temporal features** while maintaining **reliability and interpretability**.

---

## 🎯 Objectives

* Analyze and understand underlying **spatial and temporal patterns** in deepfake videos
* Identify artifacts such as:

  * Facial inconsistencies
  * Motion irregularities
  * Synchronization errors
* Implement deep learning models to distinguish between **real and manipulated content**
* Utilize **EfficientNet** for spatial feature extraction from video frames
* Apply a **Temporal Transformer** to capture dependencies across frame sequences
* Evaluate model performance on benchmark datasets to ensure:

  * Accuracy
  * Robustness
  * Generalization

---


## 🎥 Demo Video

👉 [Click here to watch the demo](deepfake-gradcam.mp4)
