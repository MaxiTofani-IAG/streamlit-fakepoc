import pandas as pd
from sklearn.cluster import KMeans
# Importar UMAP en lugar de TSNE
import umap
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Nota: Asumimos que el archivo 'findings.csv' contiene los datos
# con las columnas ID_DEFECTO, DESCRIPCION, etc., como se indicó en el input original.
# Si no tienes un CSV, deberás crear un DataFrame con los datos proporcionados.

# Leer datos desde un archivo CSV con columnas adicionales
df = pd.read_csv("findings.csv")

# Solo usamos ID_DEFECTO y DESCRIPCION para el clustering


# 2. Generar Embeddings y Clusterizar (igual que tu código original)
descriptions = df["DESCRIPCION"].tolist()
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(descriptions)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["CLUSTER"] = kmeans.fit_predict(embeddings)

# 3. Reducir dimensiones con UMAP (en lugar de t-SNE)
# Usamos parámetros por defecto para una buena base, pero n_neighbors y min_dist son importantes para ajustar.
reducer = umap.UMAP(n_components=2, random_state=42)
umap_result = reducer.fit_transform(embeddings)

# 4. Asignar resultados de UMAP al DataFrame
df["UMAP_1"], df["UMAP_2"] = umap_result[:, 0], umap_result[:, 1]

# 5. Visualizar con UMAP
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="UMAP_1", y="UMAP_2", hue="CLUSTER", palette="Set2", s=100)
for i in range(len(df)):
    plt.text(df.UMAP_1[i] + 0.05, df.UMAP_2[i], df.ID_DEFECTO[i], fontsize=9)

plt.title("Clusterización de Findings por Descripción (UMAP)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# 6. Análisis de Clústeres (igual que tu código original)
print("\n--- Análisis de Clústeres por Descripción (UMAP) ---")
for cluster_id in sorted(df['CLUSTER'].unique()):
    print(f"\nCluster {cluster_id}:")
    cluster_data = df[df['CLUSTER'] == cluster_id]
    for _, row in cluster_data.iterrows():
        print(f"  - {row['ID_DEFECTO']}: {row['DESCRIPCION']}")