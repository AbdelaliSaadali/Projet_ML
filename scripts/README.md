# Company Classification ML Project
This project classifies companies into 4 categories based on their monthly worker data using Unsupervised Machine Learning (K-Means Clustering).

## Categories
1. **Entreprises stables**: Consistent work days, high full-time ratio.
2. **Entreprises saisonnières**: Fewer work days, lower pay (likely part-time/seasonal).
3. **Entreprises irrégulières**: High variance in work days, mixed workforce.
4. **Entreprises potentiellement frauduleuses**: Anomalous data (e.g., extremely high salaries).

## Files
- `analysis.ipynb`: Jupyter Notebook containing the data analysis, feature extraction, and model implementation from scratch.
- `train.py`: Python script to execute the training process efficiently.
- `app.py`: Streamlit application to visualize the results.
- `processed_features.csv`: The dataset with extracted features and assigned clusters.
- `kmeans_centroids.npy`: Saved model centroids.

## How to Run

1. **Install Dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit
   ```

2. **Train the Model**
   You can run the notebook `analysis.ipynb` or the script `train.py`.
   ```bash
   python3 train.py
   ```
   This will generate `processed_features.csv` and model files.

3. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

## Methodology
The model extracts features such as average days worked, salary variance, and full-time worker ratio for each company. It then uses K-Means clustering (implemented from scratch) to group companies into 4 clusters, which are then mapped to the target categories based on their statistical profiles.
