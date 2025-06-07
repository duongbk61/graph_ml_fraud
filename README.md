
# 🧠 FusionHinSAGE: Fraud Detection with Graph Embedding Fusion

This project applies **Graph Neural Networks (HinSAGE)** in combination with **tabular features** to improve fraud detection on credit card transaction data. It constructs a heterogeneous graph of users, merchants, and transactions, learns node embeddings, and fuses them with traditional features for classification.

---

## 📌 Project Structure

```
.
├── FusionHinSAGE.ipynb       # Main notebook for training and evaluation
├── requirements.txt          # Python package dependencies
├── kaggle_data/              # Folder to store the Kaggle dataset
│   └── fraudTrain.csv
└── README.md                 # This file
```

---

## ⚙️ Step 1: Install Dependencies

Before running the notebook, install required Python packages:

```bash
pip install -r requirements.txt
```

> Ensure your Python environment supports `tensorflow`, `stellargraph`, `xgboost`, `scikit-learn`, `pandas`, and `matplotlib`.

---

## 📥 Step 2: Download Dataset

Download the dataset manually from Kaggle:

📎 [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

Save the file `fraudTrain.csv` into a folder named `kaggle_data/`, like so:

```
kaggle_data/
└── fraudTrain.csv
```

---

## 🚀 Step 3: Run the Notebook

Execute the notebook step-by-step:

```bash
jupyter notebook FusionHinSAGE.ipynb
```

Inside the notebook:
- Loads and preprocesses the data
- Constructs a heterogeneous graph with nodes:
  - `client`, `merchant`, and `transaction`
- Learns node embeddings via **HinSAGE**
- Applies **PCA** on embeddings
- Fuses embeddings with tabular data
- Trains **XGBoost** classifiers
- Evaluates using **Precision**, **Recall**, **F1-score**, and **Confusion Matrix**

---

## 📊 Results

- **HinSAGE + Tabular Fusion** improves recall and F1-score over using tabular features alone.
- PCA was necessary to reduce the impact of 32-dimensional embeddings.
- Best performance achieved with **PCA dimension = 2**.

---

## 📌 Example Result

| Model                  | Precision | Recall | F1-score |
|------------------------|-----------|--------|----------|
| Tabular only           | 0.96      | 0.66   | 0.78     |
| HinSAGE + Tabular      | 0.98      | 0.72   | 0.83     |

---

## 📎 Notes

- Total transactions: **139,998**
- Unique clients: **4,069**
- Unique merchants: **34,386**
- Graph is undirected and has no self-loops.

---