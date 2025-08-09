import pandas as pd
from pycaret.clustering import load_model, predict_model

# Wczytaj wytrenowany model
model = load_model('welcome_survey_clustering_pipeline_v2')

# Wczytaj dane
df = pd.read_csv('welcome_survey_simple_v2.csv', sep=';')

# Przewiduj klastry
df_with_clusters = predict_model(model, data=df)

# Zapisz dane z klastrami do CSV (do ewentualnego podglądu)
df_with_clusters.to_csv('clustered_data_v2.csv', index=False)

# Analiza: pokaż najczęstsze wartości w każdej kolumnie dla każdego klastra
for cluster_id in sorted(df_with_clusters['Cluster'].unique()):
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
    print(f"\n==== Cluster {cluster_id} ====")
    print("Liczba osób w klastrze:", len(cluster_data))
    print("Najczęstsze wartości:")
    print(cluster_data.mode().iloc[0])
    print("-" * 40)
