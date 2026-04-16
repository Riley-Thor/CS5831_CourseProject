# Data Diggers: Fraudulent Review Detection

This repository contains the codebase for the "Data Diggers" project, which proposes a multiview framework for detecting fraudulent and AI-generated reviews across e-commerce platforms. The framework bridges structural behavioral metadata with deep semantic embeddings to isolate sophisticated anomalies and opinion spam. 
  * **NOTE:**
  * `framework.py` is all of the code cells from the Notebook translated into one big file.
  * While the Notebook does not render in GitHub, it will once it is downloaded and opened.

This project was developed for the CS5831 Advanced Data Mining course at Michigan Technological University.

## Authors
* Felicia Huffman
* Elisabeth MacChesney
* Jessica Pamela Feliz Garrido
* Riley Meeves

## Dataset Overview
The dataset utilized for this study is a 10% stratified sample consisting of 35,896 total user reviews. The data exhibits a severe class imbalance, heavily skewing toward legitimate reviews. Specifically, the dataset contains:
* **32,210** non-spam (legitimate) reviews
* **3,686** spam (fraudulent) reviews

### Important Data Setup Instructions
To successfully run `framework.py`, the following three data files must be present in the root directory:
1. `reviewContent`: Contains user IDs, product IDs, dates, and the raw review text.
   * **NOTE:** Due to GitHub's file size constraints, this dataset is stored inside a `.zip` folder in the repository. **You must unzip this folder** and place the extracted `reviewContent` file directly in the same directory as the Python script before running.
2. `metadata`: Contains user IDs, product IDs, ratings, ground-truth labels, and dates.
3. `reviewGraph`: Contains graph representation labels.

## Prerequisites & Installation
The script requires Python 3.x and several machine learning and deep learning libraries. It is highly recommended to run this in an environment with GPU support (e.g., CUDA) for generating the DeBERTa and CNN embeddings efficiently.

Install the required dependencies using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow transformers keras optuna xgboost torch tqdm umap-learn
```

## Running the Pipeline
Execute the main script from your terminal:

```bash
python framework.py
```

### Pipeline Stages:
1. Data Preparation: Merges the content, metadata, and graph datasets and handles class formatting.
2. Deep Embeddings: * Uses `microsoft/deberta-v3-base` to extract rich semantic representations.
  * Uses a custom 1D Convolutional Neural Network (CNN) to extract structural n-gram patterns.
3. Feature Engineering: Computes 14 distinct behavioral and stylometric features (e.g., temporal cadence, rating variations, review lengths).
4. Multiview Spectral Clustering: Fuses the behavioral, semantic, and structural representations into a combined graph and optimizes cluster weights using Optuna.
5. Classification & Evaluation: Evaluates eight supervised baseline models (Logistic Regression, XGBoost, Random Forest, SVM, Neural Networks, Decision Trees, Naive Bayes, and a Stacking Classifier). Hyperparameters are optimized dynamically via Optuna.

## Outputs
Running the script will automatically generate and save several artifacts:
* `DebertEmbed.csv`: Extracted DeBERTa embeddings.
* `X_combined.csv`: The final fused feature set containing behavioral features and scaled embeddings.
* Visualizations: Generates UMAP projection plots for the embeddings, boxplots for behavioral feature distributions, and ROC-AUC curves for each evaluated model.

## Key Results
Based on the experimental evaluation:
* Logistic Regression achieved the highest overall performance with an $F_1$-score of 0.918 and an ROC-AUC of 0.701.
* UMAP visualizations confirmed that deep linguistic embeddings successfully isolate structural and semantic anomalies that pure behavioral metadata misses.
