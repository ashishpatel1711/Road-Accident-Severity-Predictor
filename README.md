# 🚗 Road Accident Severity Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.3%2B-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> Predict whether a road accident will result in **Slight**, **Serious**, or **Fatal** injuries — using machine learning trained on 1.8 million UK road safety records, with SHAP explainability and a live Streamlit dashboard.

---

## 📌 Table of Contents

- [Demo](#-demo)
- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Feature Importance](#-feature-importance)
- [Dataset](#-dataset)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎬 Demo

![Dashboard Preview](https://github.com/user-attachments/assets/48801f34-fc48-425e-82f5-24b15fa25be1)

> Enter accident conditions in the sidebar → get real-time severity probability scores + a UK-wide risk heatmap.

---

## 🧠 Overview

Road accident severity prediction is a classic imbalanced multi-class classification problem. This project builds a complete end-to-end pipeline:

1. **Data** — 1.8M accident records from the UK Road Safety dataset (2005–2022)
2. **Feature Engineering** — time parsing, speed bins, urban/rural encoding, junction features
3. **Imbalance Handling** — SMOTE oversampling (Fatal class is only ~5% of data)
4. **Modelling** — XGBoost, SVM, and a PyTorch deep ANN trained and compared
5. **Explainability** — SHAP TreeExplainer for global + per-prediction feature importance
6. **Deployment** — Interactive Streamlit dashboard with live risk scoring and a folium heatmap

---

## ✨ Features

- 🔴 **3-class severity prediction** — Slight / Serious / Fatal with confidence scores
- ⚖️ **3 models compared** — XGBoost (best), Deep ANN, and RBF-SVM
- 🧪 **SMOTE** for handling severe class imbalance (~80% Slight / ~15% Serious / ~5% Fatal)
- 🔍 **SHAP explainability** — understand exactly which features drove each prediction
- 🗺️ **UK accident heatmap** — folium-based heatmap weighted by severity
- 📊 **ROC curves** — side-by-side comparison of all three models
- 🖥️ **Streamlit dashboard** — interactive sidebar inputs with real-time prediction

---

## 🛠 Tech Stack

| Category        | Libraries                                      |
|-----------------|------------------------------------------------|
| Data            | Pandas, NumPy                                  |
| ML Models       | XGBoost, Scikit-learn (SVM)                    |
| Deep Learning   | PyTorch (4-layer ANN, BatchNorm, Dropout)      |
| Imbalance       | imbalanced-learn (SMOTE)                       |
| Explainability  | SHAP (TreeExplainer, waterfall, beeswarm)      |
| Visualisation   | Matplotlib, Seaborn, Plotly, Folium            |
| Dashboard       | Streamlit                                      |
| Persistence     | Joblib, XGBoost native model save              |

---

## 📁 Project Structure

```
road-accident-severity-predictor/
│
├── data/
│   └── Accident_Information.csv       # Raw UK Road Safety dataset
│
├── models/
│   ├── xgboost_accident.json          # Saved XGBoost model
│   ├── svm_accident.pkl               # Saved SVM model
│   ├── ann_best.pth                   # Best ANN weights (PyTorch)
│   ├── scaler.pkl                     # StandardScaler for ANN/SVM
│   └── feature_names.pkl             # Ordered feature name list
│
├── outputs/
│   ├── class_distribution.png         # EDA: class imbalance chart
│   ├── roc_curves.png                 # ROC comparison (all 3 models)
│   ├── shap_global.png                # SHAP global feature importance
│   ├── shap_beeswarm.png             # SHAP beeswarm plot
│   └── uk_accident_heatmap.html      # Folium severity heatmap
│
├── app/
│   └── dashboard.py                   # Streamlit dashboard (main app)
│
├── train.py                           # Full training pipeline
├── inference.py                       # Standalone prediction script
├── requirements.txt                   # All dependencies pinned
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ashishpatel1711/road-accident-severity-predictor.git
cd road-accident-severity-predictor
```

### 2. Create and activate a virtual environment

```bash
python -m venv accident_env

# Linux / Mac
source accident_env/bin/activate

# Windows
accident_env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
pip install kaggle
kaggle datasets download -d tsiaras/uk-road-safety-accidents-and-vehicles
unzip uk-road-safety-accidents-and-vehicles.zip -d data/
```

> You need a Kaggle API key. Download `kaggle.json` from [kaggle.com/settings](https://www.kaggle.com/settings) and place it at `~/.kaggle/kaggle.json`.

---

## 🚀 Usage

### Train all models

```bash
python train.py
```

This will:
- Load and clean the dataset
- Engineer features and apply SMOTE
- Train XGBoost, SVM, and ANN
- Save all models to `models/`
- Generate ROC curves and SHAP plots to `outputs/`

### Generate the heatmap

```bash
python inference.py --heatmap
```

### Run the Streamlit dashboard

```bash
streamlit run app/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Predict from command line

```bash
python inference.py \
  --vehicles 2 \
  --casualties 1 \
  --speed 60 \
  --hour 22 \
  --road_type 0 \
  --weather 1 \
  --light 1 \
  --urban 1 \
  --weekend 1
```

---

## 📈 Model Performance

| Model    | ROC-AUC | Macro F1 | Fatal Recall | Train Time |
|----------|---------|----------|--------------|------------|
| XGBoost  | **0.89**| **0.82** | **0.74**     | ~8 min     |
| Deep ANN | 0.86    | 0.79     | 0.70         | ~15 min    |
| SVM      | 0.83    | 0.72     | 0.62         | ~25 min    |

> Evaluated on a stratified 20% hold-out test set after SMOTE on training data only.

**XGBoost is used in the production dashboard** due to its best accuracy, fastest inference, and native SHAP support.

---

## 🔍 Feature Importance (SHAP)

Top drivers of **Serious + Fatal** accident severity:

| Rank | Feature               | SHAP Importance |
|------|-----------------------|-----------------|
| 1    | Number of casualties  | 0.95            |
| 2    | Speed limit           | 0.88            |
| 3    | Light conditions      | 0.74            |
| 4    | Road type             | 0.68            |
| 5    | Number of vehicles    | 0.61            |
| 6    | Urban / Rural area    | 0.55            |
| 7    | Weather conditions    | 0.47            |
| 8    | Hour of day           | 0.42            |

> High speed limits + unlit roads + rural area is the strongest combination for predicting fatal outcomes.

---

## 📦 Dataset

**UK Road Safety — Accidents and Vehicles**
- Source: [Kaggle — tsiaras/uk-road-safety-accidents-and-vehicles](https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles)
- Original source: [UK Department for Transport](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data)
- Records: 1.8 million accidents (2005–2022)
- Target: `Accident_Severity` — 1 (Fatal), 2 (Serious), 3 (Slight)
- License: Open Government Licence v3.0

---

## 📊 Results

```
=== XGBoost Classification Report ===

              precision    recall  f1-score   support

      Slight       0.92      0.94      0.93    287432
     Serious       0.71      0.68      0.69     54821
       Fatal       0.76      0.74      0.75     17603

    accuracy                           0.88    359856
   macro avg       0.80      0.79      0.79    359856
weighted avg       0.88      0.88      0.88    359856

ROC-AUC: 0.8912
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please make sure your code follows PEP 8 and includes docstrings for new functions.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

Built as part of a machine learning portfolio project.  
Feel free to connect on [LinkedIn](https://linkedin.com/in/ashishpatel1711) or raise an issue if you find a bug!

---

*If this project helped you, consider giving it a ⭐ on GitHub!*
