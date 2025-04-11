import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

df = pd.read_csv("Movies - Annotation (3 Annotators).csv")

df.columns = df.columns.str.strip()

ratings = df[['A1', 'A2', 'A3']]

ratings = ratings.apply(pd.to_numeric, errors='coerce').dropna().astype(int)

all_categories = sorted(ratings.stack().unique())

def to_fleiss_matrix(df, categories):
    matrix = []
    for _, row in df.iterrows():
        counts = [list(row).count(cat) for cat in categories]
        matrix.append(counts)
    return np.array(matrix)

fleiss_matrix = to_fleiss_matrix(ratings, all_categories)

kappa = fleiss_kappa(fleiss_matrix, method='fleiss')
print(f"Fleiss' Kappa: {kappa:.4f}")
