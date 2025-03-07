# 🌫️ Image Dehazing and Object Identification 🚗

This project focuses on enhancing visibility in hazy images using a **GAN-based dehazing model** and then identifying objects within the dehazed images using a **pretrained and fine-tuned MobileNet-SSD model**.

---

## ✨ Features

- ✅ **Image Dehazing:**  
  Removes haze from images using a Generative Adversarial Network (GAN) to restore clarity.

- ✅ **Object Detection:**  
  Detects objects in dehazed images using **MobileNet-SSD**, a lightweight and efficient deep learning model, fine-tuned for improved accuracy.

---

## 🏗️ Models Used

### 1️⃣ Dehazing Model
- **Architecture:** GAN (Generative Adversarial Network)
- **Purpose:** Enhance and restore hazy images by generating clear versions.
- **Training Data:** Hazy and corresponding clear images.
- **Output:** Dehazed image with improved visibility and details.

### 2️⃣ Object Detection Model
- **Architecture:** MobileNet-SSD
- **Pretrained:** Yes (fine-tuned on custom dataset)
- **Purpose:** Detect and classify objects in the dehazed images.
- **Output:** Images with bounding boxes and class labels.

---


