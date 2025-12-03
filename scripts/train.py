import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.preprocessing import StandardScaler

# Path to data
EXTRACTED_CSVS_PATH = '../extracted_csvs'

# Feature Extraction Function
def extract_features(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        
        # Basic stats
        num_workers = len(df)
        
        # Days stats
        avg_days = df['nombre_jours'].mean()
        std_days = df['nombre_jours'].std()
        min_days = df['nombre_jours'].min()
        max_days = df['nombre_jours'].max()
        
        # Salary stats
        avg_salary = df['salaire'].mean()
        std_salary = df['salaire'].std()
        total_salary = df['salaire'].sum()
        
        # Derived features
        # Ratio of full month workers (assuming 26 days is full)
        full_time_ratio = (df['nombre_jours'] >= 26).mean()
        
        id_adherent = df['ID_adherent'].iloc[0]
        
        return {
            'ID_adherent': id_adherent,
            'num_workers': num_workers,
            'avg_days': avg_days,
            'std_days': std_days if not np.isnan(std_days) else 0,
            'min_days': min_days,
            'max_days': max_days,
            'avg_salary': avg_salary,
            'std_salary': std_salary if not np.isnan(std_salary) else 0,
            'total_salary': total_salary,
            'full_time_ratio': full_time_ratio
        }
    except Exception as e:
        # print(f"Error processing {file_path}: {e}")
        return None

# Process all files
print("Scanning files...")
csv_files = glob(os.path.join(EXTRACTED_CSVS_PATH, "*.csv"))
print(f"Found {len(csv_files)} files.")

features_list = []
for i, f in enumerate(csv_files):
    if i % 1000 == 0:
        print(f"Processed {i} files...")
    feat = extract_features(f)
    if feat:
        features_list.append(feat)
        
features_df = pd.DataFrame(features_list)
features_df = features_df.fillna(0)
print(f"Features shape: {features_df.shape}")

# Normalize Data
cols_to_use = ['num_workers', 'avg_days', 'std_days', 'avg_salary', 'std_salary', 'full_time_ratio']
X = features_df[cols_to_use].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Implement K-Means from Scratch
class KMeansScratch:
    def __init__(self, k=4, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly (seed for reproducibility)
        np.random.seed(42)
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[idx]
        
        for i in range(self.max_iters):
            distances = self._calc_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.k)])
            
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
                
            self.centroids = new_centroids
            
    def _calc_distances(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)
        return distances

print("Training KMeans...")
kmeans = KMeansScratch(k=4)
kmeans.fit(X_scaled)
features_df['cluster'] = kmeans.labels

# Save artifacts
features_df.to_csv('processed_features.csv', index=False)
np.save('kmeans_centroids.npy', kmeans.centroids)
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

# Print Cluster Analysis for mapping
print("\nCluster Analysis (Mean values):")
print(features_df.groupby('cluster')[cols_to_use].mean())
