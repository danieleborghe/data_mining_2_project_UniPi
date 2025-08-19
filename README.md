# Advanced Data Mining for Speech Emotion Recognition

This repository presents an advanced data mining project focused on **recognizing human emotions from speech**. Building upon foundational techniques, this work explores a sophisticated pipeline involving state-of-the-art classification models, imbalanced learning strategies, outlier detection, explainable AI (XAI), and time series analysis.

The project was developed for the "Data Mining II" course at the **University of Pisa (UniPi)**.

[![Read the Report](https://img.shields.io/badge/Read_the_Full-Report-red?style=for-the-badge&logo=adobeacrobatreader)](Data_Mining_2_examination_project_report.pdf)

---

## 📝 Table of Contents

- [Project Goal: Pushing the Boundaries of Emotion Recognition](#-project-goal-pushing-the-boundaries-of-emotion-recognition)
- [Our Approach: An Advanced Data Mining Workflow](#-our-approach-an-advanced-data-mining-workflow)
- [Technical Stack & Methodologies](#-technical-stack--methodologies)
- [Dataset: The RAVDESS Audio Dataset](#-dataset-the-ravdess-audio-dataset)
- [Project Workflow & Implemented Techniques](#-project-workflow--implemented-techniques)
- [Key Findings & Results](#-key-findings--results)
- [Repository Structure](#-repository-structure)
- [How to Run This Project](#-how-to-run-this-project)
- [Authors](#-authors)

---

## 🎯 Project Goal: Pushing the Boundaries of Emotion Recognition

While standard classification models can provide a solid baseline for speech emotion recognition, real-world data presents complex challenges like class imbalance, noisy outliers, and the need for model transparency. This project aims to address these challenges by asking: **"How can we enhance the performance, robustness, and interpretability of emotion recognition models by applying advanced data mining techniques?"**

We move beyond basic classification to build a more resilient and insightful pipeline, capable of handling complex data and providing explanations for its predictions.

---

## 💡 Our Approach: An Advanced Data Mining Workflow

This project is structured as a deep dive into several advanced topics in data mining. Each module tackles a specific challenge in the machine learning lifecycle, creating a comprehensive and robust solution. Our workflow includes:

1.  **Advanced Predictive Modeling**: Implementing and fine-tuning powerful ensemble models and neural networks.
2.  **Imbalanced Learning**: Systematically addressing the issue of unequal class distribution in the dataset.
3.  **Outlier Analysis**: Detecting and analyzing anomalous data points to improve model robustness.
4.  **Explainable AI (XAI)**: Opening the "black box" of our best-performing models to understand their decision-making processes.
5.  **Time Series Analysis**: Treating the audio features as time series to explore temporal patterns and classifications.

---

## 💻 Technical Stack & Methodologies

-   **Language**: **Python 3.x**
-   **Core Libraries**:
    -   **Pandas** & **NumPy**: For high-performance data manipulation and analysis.
    -   **scikit-learn**: For a wide range of tasks including preprocessing, ensemble modeling (Random Forest, Bagging, Boosting), SVM, and outlier detection.
    -   **TensorFlow** & **Keras**: Used for building, training, and evaluating a Multi-Layer Perceptron (MLP) neural network.
    -   **`imbalanced-learn`**: The key library for implementing SMOTE (oversampling) and various undersampling techniques.
    -   **`shap`**: For implementing SHapley Additive exPlanations, a state-of-the-art XAI technique to explain model predictions.
    -   **`pyod`**: Used for implementing the Angle-Based Outlier Detection (ABOD) algorithm.
    -   **Matplotlib** & **Seaborn**: For advanced data visualization.
    -   **Jupyter Notebook**: The environment for all experimentation and analysis.

---

## 📊 Dataset: The RAVDESS Audio Dataset

We use the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)** dataset. It contains audio recordings from 24 actors expressing 8 different emotions (`neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`). We work with a pre-processed version containing **60 acoustic features** extracted from the raw audio, as well as a time-series version based on Mel-spectrograms.

---

## ⚙️ Project Workflow & Implemented Techniques

The project is divided into several specialized modules, each focusing on an advanced data mining task.

1.  **Data Understanding and Preparation**
    -   **Activities**: EDA, feature analysis, and creation of a robust preprocessing pipeline for both the feature-based and time-series datasets.
    -   **Notebook**: `Data Understanding and Preparation/data_preparation.ipynb`

2.  **Advanced Classification**
    -   **Goal**: To achieve the highest possible accuracy in emotion prediction.
    -   **Models Implemented**:
        -   **Ensemble Methods**: Random Forest, Bagging, AdaBoost, and Gradient Boosting.
        -   **Support Vector Machines (SVM)**: Both linear and non-linear (RBF kernel) classifiers.
        -   **Neural Networks**: A Multi-Layer Perceptron (MLP) implemented in both scikit-learn and Keras/TensorFlow.
    -   **Process**: Rigorous hyperparameter tuning using Grid Search, feature selection via Recursive Feature Elimination (RFE), and detailed performance evaluation.

3.  **Imbalanced Learning**
    -   **Goal**: To mitigate the negative effects of the imbalanced emotion classes in the dataset.
    -   **Techniques**:
        -   **Oversampling**: Using **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic samples for minority classes.
        -   **Undersampling**: Using methods like RandomUnderSampler, NearMiss, and TomekLinks to reduce the number of samples in the majority classes.
    -   **Notebooks**: `Imbalanced Learning/oversampling_v2.ipynb`, `undersampling_v2.ipynb`

4.  **Outlier Detection**
    -   **Goal**: To identify and analyze data points that deviate significantly from the rest of the data.
    -   **Algorithms**:
        -   **Proximity-Based**: Angle-Based Outlier Detection (ABOD).
        -   **Ensemble-Based**: Isolation Forest.
        -   **Probabilistic**: Using Gaussian Mixture Models.
    -   **Notebooks**: `Outliers/ABOD_V2.ipynb`, `isolation_forest_approach.ipynb`

5.  **Explainable AI (XAI)**
    -   **Goal**: To interpret the predictions of our best-performing model (Random Forest).
    -   **Method**: We apply **SHAP (SHapley Additive exPlanations)** to understand which acoustic features are most influential in predicting each emotion, both globally and for individual predictions.
    -   **Notebook**: `Explainable AI/explainable_machine_learning.ipynb`

6.  **Time Series Analysis**
    -   **Goal**: To classify emotions by treating Mel-spectrogram data as time series.
    -   **Techniques**:
        -   **Feature Extraction**: Using statistical moments (mean, std, skew, kurtosis) from the time series as features.
        -   **Classification**: Applying a state-of-the-art time series classifier (ROCKET) for direct classification.
        -   **Clustering**: Using Piecewise Aggregate Approximation (PAA) to simplify the series for clustering.
    -   **Notebooks**: `Time Series/ts_classification_sota.ipynb`, `TS_clustering_PAA.ipynb`

---

## 📈 Key Findings & Results

-   **Best Performing Model**: **Random Forest** and **Gradient Boosting** emerged as the top-performing models, achieving accuracies of **~82-83%** after extensive tuning.
-   **Impact of Imbalanced Learning**: **Oversampling with SMOTE** provided a significant boost in performance, especially for minority classes like 'disgust' and 'surprised', improving the model's overall fairness and robustness.
-   **XAI Insights**: The SHAP analysis revealed that features related to Mel-Frequency Cepstral Coefficients (MFCCs) and spectral contrast were consistently the most important drivers for the model's predictions across all emotions.
-   **Time Series Classification**: The ROCKET classifier, applied directly to the time series data, demonstrated competitive performance, suggesting that temporal dynamics contain valuable information that is partially lost in feature-based aggregation.

---

## 📂 Repository Structure

```

.
├── Data Understanding and Preparation/
│   ├── data\_preparation.ipynb
│   └── DATASET PREPARED/
├── Classification/
│   ├── Random\_Forest.ipynb
│   ├── Boosting.ipynb
│   ├── SVM.ipynb
│   └── MLP\_classifier\_KERAS.ipynb
├── Imbalanced Learning/
│   ├── oversampling\_v2.ipynb
│   └── undersampling\_v2.ipynb
├── Outliers/
│   ├── outliers\_comparison.ipynb
│   └── ABOD\_V2.ipynb
├── Explainable AI/
│   └── explainable\_machine\_learning.ipynb
├── Time Series/
│   ├── TS\_data\_understanding\_preparation.ipynb
│   └── ts\_classification\_sota.ipynb
├── Data\_Mining\_2\_examination\_project\_report.pdf \# The final project report
└── README.md                                  \# This file

````

---

## 🚀 How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/danieleborghe/data_mining_2_project_UniPi.git](https://github.com/danieleborghe/data_mining_2_project_UniPi.git)
    cd data_mining_2_project_UniPi
    ```

2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras imbalanced-learn shap pyod jupyter
    ```

3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

4.  **Explore the Notebooks:**
    -   Start with `Data Understanding and Preparation/data_preparation.ipynb`.
    -   Each folder (`Classification`, `Imbalanced Learning`, etc.) contains standalone notebooks that can be run to reproduce the specific analyses.

---

## 👥 Authors

- **Daniele Borghesi**
- **Lucrezia Labardi**
- **Vincenzo Sammartino**
