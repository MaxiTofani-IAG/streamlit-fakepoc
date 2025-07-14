import streamlit as st
import pandas as pd

# Cargar CSV
df = pd.read_csv("findings.csv", parse_dates=["FECHA_INICIO", "FECHA_FIN"])

st.title("Findings Aeronáuticos")

# ---------------------
# Filtros en la barra lateral
# ---------------------
st.sidebar.header("Filtros")

descripcion_input = st.sidebar.text_input("Buscar en descripción")

# Filtro por ID_DEFECTO
all_ids = df["ID_DEFECTO"].astype(str).unique().tolist()
selected_ids = st.sidebar.multiselect(
    "ID Defecto",
    options=["Todos"] + all_ids,
    default=["Todos"]
)
if "Todos" in selected_ids or not selected_ids:
    selected_ids = all_ids

# Categorías con opción "Todos"
all_categorias = df["CATEGORIA"].unique().tolist()
selected_categorias = st.sidebar.multiselect(
    "Categorías",
    options=["Todos"] + all_categorias,
    default=["Todos"]
)
if "Todos" in selected_categorias or not selected_categorias:
    selected_categorias = all_categorias

# ATA Numbers con opción "Todos"
all_ata = df["ATA_NUMBER"].unique().tolist()
selected_ata = st.sidebar.multiselect(
    "ATA Numbers",
    options=["Todos"] + all_ata,
    default=["Todos"]
)
if "Todos" in selected_ata or not selected_ata:
    selected_ata = all_ata

# Rango de fechas
start_date = st.sidebar.date_input("Desde", value=df["FECHA_INICIO"].min())
end_date = st.sidebar.date_input("Hasta", value=df["FECHA_FIN"].max())

# Rango de horas
min_horas = float(df["CANTIDAD_HORAS"].min())
max_horas = float(df["CANTIDAD_HORAS"].max())
selected_horas = st.sidebar.slider(
    "Horas empleadas",
    min_value=int(min_horas),
    max_value=int(max_horas) + 1,
    value=(int(min_horas), int(max_horas)),
    step=1
)
# ---------------------
# Aplicar filtros
# ---------------------
filtered_df = df[
    (df["CATEGORIA"].isin(selected_categorias)) &
    (df["ATA_NUMBER"].isin(selected_ata)) &
    (df["FECHA_INICIO"] >= pd.to_datetime(start_date)) &
    (df["FECHA_FIN"] <= pd.to_datetime(end_date)) &
    (df["CANTIDAD_HORAS"].between(*selected_horas)) &
    (df["ID_DEFECTO"].astype(str).isin(selected_ids)) &
    (df["DESCRIPCION"].str.contains(descripcion_input, case=False, na=False))
]

st.write(f"### Registros filtrados: {len(filtered_df)}")
st.dataframe(filtered_df)

# ---------------------
# Visualizaciones clave
# ---------------------

# 1. Total de horas por Categoría
st.subheader("Total de horas por Categoría")
horas_por_categoria = filtered_df.groupby("CATEGORIA")["CANTIDAD_HORAS"].sum().sort_values(ascending=False)
st.bar_chart(horas_por_categoria)

# 2. Cantidad de defectos por ATA
st.subheader("Cantidad de defectos por ATA Number")
defectos_por_ata = filtered_df["ATA_NUMBER"].value_counts().sort_index()
st.bar_chart(defectos_por_ata)

# 3. Top 5 defectos con mayor cantidad de horas
st.subheader("Top 5 defectos con mayor cantidad de horas")
top_defectos = filtered_df.sort_values("CANTIDAD_HORAS", ascending=False).head(5)
st.table(top_defectos[["ID_DEFECTO", "DESCRIPCION", "CANTIDAD_HORAS", "CATEGORIA"]])

# 4. Evolución mensual de defectos
st.subheader("Evolución mensual de defectos reportados")
filtered_df["MES"] = filtered_df["FECHA_INICIO"].dt.to_period("M").dt.to_timestamp()
defectos_por_mes = filtered_df.groupby("MES").size()
st.line_chart(defectos_por_mes)
