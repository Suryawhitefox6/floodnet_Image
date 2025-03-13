# 🌊 FloodNet Image Segmentation: U-Net for Boundary Detection & Object Extraction

## 📌 Introduction
Floods cause severe damage to **infrastructure, roads, and agriculture**, making **rapid assessment critical** for disaster response. **Traditional segmentation methods** struggle with **occlusions, water reflections, and noisy data**. This project leverages **deep learning (U-Net) and region-based processing** to **accurately segment flooded and non-flooded areas** from **aerial images**, aiding in relief efforts.

---

## 🎯 Objective
- Develop an **image segmentation model** for **multi-class semantic segmentation** of flood-affected regions.
- Improve **boundary detection & object separation** for precise classification.
- Use **U-Net & region-based processing** to enhance segmentation accuracy.

---

## 🗂 Dataset: FloodNet (Kaggle)
📌 **Dataset Link**: [FloodNet Kaggle Dataset](https://www.kaggle.com/datasets/imroze/floodnet-cropresized-512-wpartialmasks/data)  

- **Source**: FloodNet Challenge (Preprocessed version from Kaggle)  
- **Total Images**: **398**  
- **Image Format**: **RGB .png**  
- **Resolution**: Originally **4000×3000**, resized to **512×512**  
- **Segmentation Classes**:
  - 0: Background
  - 1: Building Flooded
  - 2: Building Non-Flooded
  - 3: Road Flooded
  - 4: Road Non-Flooded
  - 5: Water
  - 6: Tree
  - 7: Vehicle
  - 8: Pool
  - 9: Grass
- **Challenges**:
  - **Class Imbalance** (Some classes occur less frequently).
  - **High Class Similarity** (Flooded roads and water look similar).
  - **Noisy Labels & Occlusions** (Trees, shadows, and missing labels).

---

## 🏗 Methodology
### **1️⃣ Noise Reduction Techniques**
To improve segmentation quality, **four filtering methods** are applied:
- **Gaussian Blur**: Reduces sensor noise but may blur edges.
- **Median Filtering**: Removes salt-and-pepper noise while preserving edges.
- **BM3D (Block-Matching and 3D Filtering)**: Advanced denoising without excessive smoothing.
- **Bilateral Filtering**: Smooths noise while preserving boundaries.

### **2️⃣ Image Segmentation Techniques**
- **U-Net**: A **deep learning-based encoder-decoder model** for **precise segmentation**.
- **Region-Based Processing**:
  - **Region-Growing Algorithm**: Expands regions based on similarity.
  - **Connected Component Analysis (CCA)**: Labels & filters small noise regions.
- **Traditional Methods (for comparison)**:
  - **K-Means Clustering**
  - **Mean Shift Filtering**
  - **Graph-Based Segmentation (SLIC/Felzenszwalb)**

---

## 📊 Evaluation Metrics & Results
To measure segmentation quality, we use:
| Metric | Description | Score |
|---------|-------------|-------|
| **Intersection over Union (IoU)** | Measures overlap between prediction & ground truth | **0.8453** |
| **Dice Coefficient** | Evaluates segmentation similarity (F1-score for masks) | **0.8921** |
| **Pixel Accuracy** | Percentage of correctly classified pixels | **94.05%** |

**📌 Key Observations:**
- **U-Net achieves the best segmentation accuracy**.
- **Region-based processing improves object separation**.
- **K-Means & Mean Shift struggle with boundary precision**.

---

## 🚀 Results & Conclusion
- **U-Net achieves state-of-the-art segmentation**, with **high IoU & Dice scores**.
- **Region-based methods help refine segmentation** but require fine-tuning.
- **Future Work**: Integrate **Transformer-based models** for improved segmentation.

---

## 📌 How to Run the Project
### **Requirements**
- Python 3.x
- TensorFlow / PyTorch
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

### **Installation**
```bash
pip install numpy opencv-python matplotlib scikit-learn tensorflow
