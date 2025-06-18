# Predicting Road Crash Severity in Nigeria Using Using ML Optimization Techniques

- This project tackles Nigeria's high road traffic fatality rate (41,709 annually) using FRSC Road Transport Data from the National Bureau of Statistics 
- The dataset includes 481 records with causative factors (e.g., speeding 56.3%, tire bursts) and crash severity (fatal/non-fatal) across Q3 2021–Q3 2024.
- It implements optimized ML models (SVM, XGBoost) and neural networks with regularization to predict severity, addressing class imbalance with SMOTE.
- The goal is to improve FRSC interventions, aligning with Nigeria’s National Road Safety Strategy II.

**Project Scope:** I first started out with cleaning and engineering this data into 8 percentage-based features (e.g., `SPV_PCT` for speeding), addressing a severe 477:4 class imbalance with SMOTE, and developing a suite of machine learning models—Support Vector Machines (SVM), XGBoost, and neural networks with regularization.


## Project Structure

```
├── data/
│   └── Road Transport Data Q3 2024.xlsx
├── model_architecture/
│   └── Road Traffic Model Architecture.png
├── saved_models/
│   ├── no_optimization_model.keras               
│   ├── optimized_nn1_model.keras
│   ├── optimized_nn2_model.keras             
│   └── xgboost_best_model.pkl 
├── Summative_Intro_to_ml_[Excel_Asaph]_assignment.ipynb         
└── README.md                    
```


## Dataset

The dataset is derived from the **FRSC Road Transport Data** collection, accessible via the National Bureau of Statistics Microdata Catalog (https://microdata.nigerianstat.gov.ng/index.php/catalog/164/study-description#metadata-description). It is available for download as an Excel file [Road Transport Data Q3 2024.xlsx](data/Road Transport Data Q3 2024.xlsx), with the latest update on May 09, 2025

- **Features and Target:**

  - **Crash Data Features:** 
    - `FATAL`: Number of fatal crashes.
    - `SERIOUS`: Number of serious crashes.
    - `MINOR`: Number of minor crashes.
    - `TOTAL CASES`: Total crash incidents.
    - `NUMBER INJURED`: Number of injured individuals.
    - `NUMBER KILLED`: Number of fatalities.
    - `TOTAL CASUALTY`: Total casualties.
    - `PEOPLE INVOLVED`: Total individuals involved.
  
  - **Causative Factors (Processed into Percentages):**
    - `SPV_PCT`: Percentage of crashes due to speeding.
    - `TBT_PCT`: Percentage due to tire bursts.
    - `BFL_PCT`: Percentage due to brake failure.
    - `WOT_PCT`: Percentage due to wrong overtaking.
    - `DGD_PCT`: Percentage due to dangerous driving.
    - `RTV_PCT`: Percentage due to road traffic violations.
    - `SLV_PCT`: Percentage due to overloading.
    - `OTHERS_PCT`: Percentage due to other factors.
  
  - **Target Variable:** `Severity` (binary: 1 = FATAL, 0 = NON-FATAL, derived from `FATAL` > 0).
  
- **Dataset Statistics:** The merged dataset contains 481 samples across 37 states and 13 quarters (Q3 2021 to Q3 2024), with an initial imbalance of 477 fatal (99.2%) and 4 non-fatal (0.8%) cases. After applying SMOTE, the resampled dataset balances to 477:477, enabling fair model training. Missing values were addressed with group-wise interpolation and mean imputation, a process I approached with meticulous care.

- **Scope and Coverage:** The data covers Nigeria (country code NGA), providing state-level insights into road traffic crashes and causative factors, making it a robust resource for safety analysis.


## Model Comparisons

This section presents a comprehensive analysis of the five models I developed, each with distinct configurations and performance metrics evaluated on validation data. The table below summarizes key hyperparameters, architectures, and results, while the accompanying narratives reflect my observations and learning experiences.

| Model             | Optimizer    | Regularizer   | Epochs | Early Stopping | Layers (Neurons) | Learning Rate | Accuracy | F1-score | Precision | Recall | Loss  |
|-------------------|--------------|---------------|--------|----------------|------------------|---------------|----------|----------|-----------|--------|-------|
| No Optimization   | Adam         | None          | 10     | No             | 4 (8, 10, 12, 1) | Default       | 0.9580   | 0.9595   | 0.9600    | 0.9590 | 0.1829 |
| Optimized NN1     | Adam         | L1 (0.05)     | 100      | monitor='val_loss', patience=3,            | 4 (8, 10, 12, 1) | 0.5           | 0.4615   | 0.7000   | 0.5000    | 1.0000 | 0.8641 |
| Optimized NN2     | RMSprop      | L2 (0.009)    | 50      | monitor='val_loss', patience=5,            | 4 (8, 10, 12, 1) | 0.08          | 0.9161   | 0.9419   | 0.9300    | 0.9500 | 0.1574 |
| Optimized SVM     | N/A          | N/A           | N/A    | N/A            | N/A              | N/A           | 0.4615   | 0.5032   | 0.5000    | 0.5065 | N/A    |
| Optimized XGBoost | N/A          | N/A           | 100    | N/A            | N/A              | N/A           | 0.9720   | 0.9733   | 1.0000    | 0.9481 | N/A    |


- **No Optimization Model:** I began with a 4-layer neural network (8, 10, 12, 1 neurons) using the default Adam optimizer, running for 10 epochs without regularization or early stopping. This model achieved an impressive 0.9580 accuracy and 0.9595 F1-score, with a final loss of 0.1829. However, the initial high loss (0.6659) during training revealed a tendency to overfit, an issue I noticed as the validation loss initially lagged. This showed the need for optimization techniques, sparking my curiosity to explore further.

- **Optimized NN1 (L1 Regularization):** Inspired to refine the model, I added L1 regularization (0.05) to the first layer, paired it with a high learning rate of 0.5, and introduced early stopping with a 3-epoch patience. This configuration surprised me as it prioritized recall (1.0) for the fatal class, ensuring no fatal crashes were missed, but accuracy dropped to 0.4615 and F1 to 0.7000, with a loss of 0.8641. The high loss showed that the aggressive learning rate destabilized training.

- **Optimized NN2 (L2 Regularization and Dropout):** Building on my training, I switched to L2 regularization (0.009), added a 0.3 dropout layer after the third dense layer, and used RMSprop with a 0.08 learning rate, again with early stopping (5-epoch patience). This model stood out with a 0.9161 accuracy, 0.9419 F1-score, and a reduced loss of 0.1574 after 8 epochs. The dropout’s role in preventing overfitting was really helpful, and seeing the validation loss stabilize was truly fulfilling.

- **Optimized SVM:** I tuned an SVM with C=5, a sigmoid kernel, gamma=0.5, and balanced class weights to handle imbalance. However, it underperformed with a 0.4615 accuracy and 0.5032 F1-score, suggesting the sigmoid kernel struggled with the synthetic SMOTE data. This outcome made me reconsider kernel choices and explore other algorithms and techniques for future experiments.

- **Optimized XGBoost:** My final model, XGBoost, was tuned with max_depth=3, learning_rate=0.02, n_estimators=100, subsample=0.5, and colsample_bytree=0.4. It delivered an outstanding 0.9720 accuracy and 0.9733 F1-score, with perfect precision (1.0) and a strong 0.9481 recall. The tree-based approach’s synergy with SMOTE data showed really good results, and proving to be the best for this task.


## Summary

- **Best Performing Combination:** Among the neural networks, Optimized NN2 (L2 regularization, 0.3 dropout, 0.08 learning rate with RMSprop, and early stopping) emerged as my personal favorite, achieving a 0.9419 F1-score and 0.9161 accuracy. The L2 regularization penalized large weights effectively, while dropout reduced overfitting, dropping the loss from 0.6659 (No Optimization) to 0.1574. However, the overall champion was XGBoost, with a 0.9733 F1-score and 0.9720 accuracy.

- **Machine Learning vs. Neural Networks:** XGBoost’s dominance with a 0.9733 F1-score highlighted its strength in handling tabular data enhanced by SMOTE, particularly with a 0.9481 recall for the critical fatal class. The tuned hyperparameters (max_depth=3, learning_rate=0.02) ensured controlled complexity and stable learning, making it a good choice. My Optimized NN2 (0.9419 F1) marked a significant improvement over the No Optimization model (0.9595 F1), where adding no regularization led to early overfitting (initial loss 0.6659). Neural networks showed potential for growth with more data or layers, but their performance here suggested a data size limitation. The Optimized SVM’s 0.5032 F1 with a sigmoid kernel revealed its struggle with imbalance, making me to consider RBF kernels or further tuning in future iterations.


## Instructions

- **Setup:** To replicate this work, install the required Python libraries using the following command in your environment: `pip install tensorflow pandas numpy scikit-learn xgboost matplotlib imbalanced-learn==0.8.1`. This ensures compatibility with the specific version of `imbalanced-learn` I used for SMOTE.

- **Running the Notebook:** Open and execute `notebook.ipynb` in Google Colab or a local Jupyter environment. Update the dataset path in the code to point to your local copy of [Road Transport Data Q3 2024.xlsx](data/Road Transport Data Q3 2024.xlsx), ensuring the file structure matches the raw data layout (e.g., Sheet1 with crash and causative data blocks).

- **Loading the Best Model:** After training, load the best-performing model (XGBoost) for predictions using `saved_models/xgboost_best_model.pkl`. This model, saved as a `.pkl` file, can be used to predict on new data after scaling it with the same `StandardScaler` fitted on the training set.

- **Verification:** I learned the hard way to double-check outputs—after running the notebook, verify the `/content/sample_data/saved_models/` directory contains all five models (svm_optimized_model.pkl, no_optimization_model.keras, optimized_nn1_model.keras, optimized_nn2_model.keras, xgboost_best_model.pkl). If any are missing, rerun the respective training cells, as I once forgot to save the SVM due to a silent error.

- **Visualizations:** The notebook includes plots (e.g., confusion matrices, loss curves) to visualize performance. If the `/content/Road Traffic Model Architecture.png` image fails to load, ensure it’s uploaded to your Colab environment or replace the path with a local file.
