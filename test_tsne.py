import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Leer datos desde un archivo CSV con columnas adicionales
df = pd.read_csv("findings.csv")

# Solo usamos ID_DEFECTO y DESCRIPCION para el clustering
descriptions = df["DESCRIPCION"].tolist()

# Modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(descriptions)

# KMeans para clusterizar
kmeans = KMeans(n_clusters=4, random_state=42)
df["CLUSTER"] = kmeans.fit_predict(embeddings)

# Reducimos dimensiones para graficar
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_result = tsne.fit_transform(embeddings)
df["TSNE_1"], df["TSNE_2"] = tsne_result[:, 0], tsne_result[:, 1]

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="TSNE_1", y="TSNE_2", hue="CLUSTER", palette="Set2", s=100)
for i in range(len(df)):
    plt.text(df.TSNE_1[i] + 0.3, df.TSNE_2[i], df.ID_DEFECTO[i], fontsize=9)
plt.title("Clusterización de Findings por Descripción")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# --- Cluster Analysis ---
print("\n--- Análisis de Clústeres por Descripción ---")
for cluster_id in sorted(df['CLUSTER'].unique()):
    print(f"\nCluster {cluster_id}:")
    cluster_data = df[df['CLUSTER'] == cluster_id]
    for _, row in cluster_data.iterrows():
        print(f"  - {row['ID_DEFECTO']}: {row['DESCRIPCION']}")
