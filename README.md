![image](https://github.com/user-attachments/assets/b8172e25-a558-4f36-8a92-49b47b9ac593)# Projects_new

---

# 🦠 Malaria Parasite Detection using Deep Learning

This project focuses on detecting malaria-infected cells and classifying parasite types using a two-stage deep learning approach. It aims to assist healthcare professionals with faster and more accurate diagnostics in resource-constrained environments.

---

## 📌 Project Overview

Malaria is a life-threatening disease caused by parasites transmitted through mosquito bites. Early detection plays a key role in treatment and prevention.  
In this project, we build a **two-stage deep learning pipeline**:

1. **Stage 1** – Detect whether a cell is infected or uninfected.  
2. **Stage 2** – If infected, classify the type of malaria parasite.

---

## 📁 Dataset

- **Source**: [NIH Malaria Dataset](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html)  
- The dataset contains over 27,000 cell images — split into:
  - Parasitized (infected)
  - Uninfected

Each image is 3-channel RGB and approximately 128x128 pixels.

---

## 🧠 Model Architecture

- **Stage 1**: Binary Classification  
  - CNN with convolutional, pooling, and dense layers  
  - Output: Infected / Uninfected

- **Stage 2**: Multi-Class Classification  
  - Another CNN for parasite classification  
  - Output: Specific parasite type (if applicable)

---

## 🛠️ Technologies Used

- **Languages**: Python  
- **Libraries**: TensorFlow / Keras, NumPy, Matplotlib, scikit-learn  
- **Tools**: Google Colab / Jupyter Notebook  

---

## 🔍 Results

- Achieved high accuracy on both detection and classification stages  
- Visualized training progress using accuracy/loss graphs  
- Confusion matrix for evaluation

---

## 📊 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix


## 📸 Sample Output
![image](https://github.com/user-attachments/assets/75817456-913c-43cc-90ca-9e655008d0f4)
