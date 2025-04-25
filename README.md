# Projects_new

---


# Here is the link for the dataset which is available in Kaggle.com(Large dataset - 700mb) :-

https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria?phase=FinishSSORegistration&returnUrl=%2Fdatasets%2Fiarunava%2Fcell-images-for-detecting-malaria%2Fversions%2F1%3Fresource%3Ddownload&SSORegistrationToken=CfDJ8HYJ4SW6YXhJj8CRciRldeTJ61iTYSe85EXT1pMXJCVAvSs8ogzrXF4HSsz84EelIzM7FsOOCQC1KjpkuijnOIIBJh-jYKZX5Iq0qRsri5qILKG8OQBW4foPWoqQdLN9Y-DXzO3FRp3OHF30hyPl8mBSouDszn7L_IJiejRepRpM2g8JMO0Gs3uKMwQzvgf6vuUA_lL_AcvBtTmm_u-iUFbXlo7rIDTWx_7geYPYC5QGqJyBAyMzEISFum30oDJi1C49978fHC9FZ0Pc00_LuKRBK5F0KWX6kbK7Qa135_CwA8PT4wiaQk7VEwFPXGM01WqmFvdpCFyjuRkl0tyULZIzMQeUMjz2b-AU&DisplayName=Bhargavi+Mekala



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
