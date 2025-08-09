# train_model.py
import pandas as pd
from pycaret.clustering import setup, create_model, save_model

# Wczytaj dane
df = pd.read_csv("welcome_survey_simple_v2.csv", sep=';')

# Konfiguracja PyCaret
s = setup(data=df, session_id=123, verbose=False, normalize=True)
# Trenuj model klastrowania
model = create_model('kmeans', num_clusters=8)  # lub inna liczba klastrów

# Zapisz model
save_model(model, 'welcome_survey_clustering_pipeline_v2')

