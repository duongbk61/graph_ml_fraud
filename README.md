
# 🧠 FusionHinSAGE: Fraud Detection with Graph Embedding Fusion

This project applies **Graph Neural Networks (HinSAGE)** in combination with **tabular features** to improve fraud detection on credit card transaction data. It constructs a heterogeneous graph of users, merchants, and transactions, learns node embeddings, and fuses them with traditional features for classification.

---

## 📌 Project Structure

```
.
├── FusionHinSAGE.ipynb       # Main notebook for training and evaluation
├── requirements.txt          # Python package dependencies
├── kaggle_data/              # Folder to store the Kaggle dataset
│   └── credit_card_transactions-ibm_v2.csv
└── README.md                 # This file
```

---

## ⚙️ Step 1: Install Dependencies

Using Python version 3.8.10
Before running the notebook, install required Python packages:

```bash
pip install -r requirements.txt
```

> Ensure your Python environment supports `tensorflow`, `stellargraph`, `xgboost`, `scikit-learn`, `pandas`, and `matplotlib`.

---

## 📥 Step 2: Download Dataset

Download the dataset manually from Kaggle:

📎 [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions)

Save the file `credit_card_transactions-ibm_v2.csv` into a folder named `kaggle_data/`, like so:

```
kaggle_data/
└── credit_card_transactions-ibm_v2.csv
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
- Evaluates using **Precision**, **Recall**, **F1-score**, **AUC**, and **Confusion Matrix**

---

## 📊 Results

- **HinSAGE + Tabular Fusion** improves recall and F1-score over using tabular features alone.
- PCA was necessary to reduce the impact of 32-dimensional embeddings.
- Best performance achieved with **PCA dimension = 4**.


---

## 📎 Notes

- Total transactions: **698,941**
- Unique clients: **4,081**
- Unique merchants: **67,873**
- Graph is undirected and has no self-loops.

---