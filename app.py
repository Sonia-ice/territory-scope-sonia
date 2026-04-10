import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

st.set_page_config(page_title="Territory Scope Sonia", layout="wide", page_icon="🌸")

# --- Fond rose pastel via CSS ---
st.markdown("""
    <style>
        /* Fond général rose pastel */
        .stApp {
            background-color: #ffe4ec;
        }
        /* Sidebar rose plus foncé */
        [data-testid="stSidebar"] {
            background-color: #ffc2d4;
        }
        /* Titres */
        h1, h2, h3 {
            color: #c2185b;
        }
        /* Boutons */
        .stButton > button {
            background-color: #f48fb1;
            color: white;
            border: none;
            border-radius: 8px;
        }
        /* Métriques */
        [data-testid="metric-container"] {
            background-color: #fff0f5;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #f48fb1;
        }
        /* Dataframe */
        .stDataFrame {
            border: 1px solid #f48fb1;
        }
    </style>
""", unsafe_allow_html=True)

# --- Chargement des données ---
@st.cache_data
def load_data():
    with open("data_llm_territoires.json", "r", encoding="utf-8") as f:
        territories = json.load(f)
    rows = []
    for t in territories:
        row = {"code": t.get("code", ""), "libelle": t.get("libelle", "")}
        row.update(t.get("indicateurs", {}))
        rows.append(row)
    df = pd.DataFrame(rows)
    return territories, df

territories, df = load_data()

numeric_cols = [c for c in df.columns if c not in ["code", "libelle"]]

short_names = {
    "Nombre de chômeurs de 15-64 ans 2022": "Chômeurs total",
    "Nombre de chômeurs de 15-64 ans 2022.1": "Chômeurs hommes",
    "Nombre de chômeurs de 15-64 ans 2022.2": "Chômeurs femmes",
    "Part des demandeurs d'emploi en activité réduite (catégories B et C) parmi les demandeurs d'emploi de catégorie ABC 2025": "Demandeurs activité réduite (%)",
    "Taux de pauvreté (seuil à 60% du revenu médian) 2021": "Taux de pauvreté (%)",
}

territory_names = sorted(df["libelle"].dropna().unique().tolist())
national_means = df[numeric_cols].mean()

dept_coords = {
    "01": (46.2, 5.2), "02": (49.5, 3.5), "03": (46.3, 3.1), "04": (44.1, 6.2),
    "05": (44.7, 6.3), "06": (43.9, 7.1), "07": (44.8, 4.5), "08": (49.7, 4.7),
    "09": (42.9, 1.6), "10": (48.3, 4.1), "11": (43.1, 2.4), "12": (44.3, 2.6),
    "13": (43.5, 5.4), "14": (49.1, -0.4), "15": (45.1, 2.6), "16": (45.7, 0.2),
    "17": (45.8, -0.7), "18": (47.1, 2.5), "19": (45.4, 1.9), "21": (47.4, 4.9),
    "22": (48.4, -2.8), "23": (46.1, 2.1), "24": (45.0, 0.7), "25": (47.2, 6.4),
    "26": (44.7, 5.1), "27": (49.1, 1.2), "28": (48.4, 1.4), "29": (48.2, -4.0),
    "30": (44.0, 4.2), "31": (43.4, 1.3), "32": (43.7, 0.6), "33": (44.8, -0.6),
    "34": (43.6, 3.6), "35": (48.1, -1.7), "36": (46.8, 1.6), "37": (47.3, 0.7),
    "38": (45.2, 5.7), "39": (46.7, 5.6), "40": (43.9, -0.8), "41": (47.6, 1.3),
    "42": (45.5, 4.1), "43": (45.1, 3.9), "44": (47.4, -1.7), "45": (47.9, 2.2),
    "46": (44.6, 1.7), "47": (44.4, 0.5), "48": (44.5, 3.5), "49": (47.5, -0.6),
    "50": (49.1, -1.4), "51": (49.0, 4.0), "52": (48.1, 5.3), "53": (48.1, -0.7),
    "54": (48.7, 6.2), "55": (49.0, 5.4), "56": (47.8, -2.8), "57": (49.1, 6.9),
    "58": (47.2, 3.5), "59": (50.5, 3.1), "60": (49.4, 2.4), "61": (48.6, 0.1),
    "62": (50.5, 2.3), "63": (45.8, 3.1), "64": (43.3, -0.8), "65": (43.1, 0.2),
    "66": (42.6, 2.7), "67": (48.5, 7.5), "68": (47.8, 7.3), "69": (45.7, 4.8),
    "70": (47.6, 6.1), "71": (46.6, 4.5), "72": (47.9, 0.2), "73": (45.5, 6.4),
    "74": (46.0, 6.4), "75": (48.9, 2.3), "76": (49.7, 1.1), "77": (48.6, 2.9),
    "78": (48.8, 1.9), "79": (46.5, -0.3), "80": (49.9, 2.3), "81": (43.9, 2.1),
    "82": (44.1, 1.3), "83": (43.5, 6.3), "84": (44.0, 5.1), "85": (46.7, -1.3),
    "86": (46.6, 0.4), "87": (45.8, 1.3), "88": (48.1, 6.5), "89": (47.8, 3.6),
    "90": (47.6, 6.9), "91": (48.5, 2.3), "92": (48.9, 2.2), "93": (48.9, 2.5),
    "94": (48.8, 2.5), "95": (49.1, 2.2),
}

# --- Sidebar ---
st.sidebar.markdown("# 🌸 Territory Scope Sonia")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🗺️ Carte interactive",
    "📊 Dashboard",
    "📈 Comparaison",
    "🏆 Classement"
])

# =====================
# PAGE : CARTE INTERACTIVE
# =====================
if page == "🗺️ Carte interactive":
    st.title("🌸 Territory Scope Sonia")
    st.subheader("🗺️ Carte des indicateurs par département")

    indicateur_carte = st.selectbox(
        "Indicateur à afficher sur la carte",
        numeric_cols,
        format_func=lambda x: short_names.get(x, x)
    )

    df_map = df[["code", "libelle", indicateur_carte]].copy()
    df_map = df_map[df_map["code"].isin(dept_coords.keys())]
    df_map["lat"] = df_map["code"].map(lambda c: dept_coords.get(c, (None, None))[0])
    df_map["lon"] = df_map["code"].map(lambda c: dept_coords.get(c, (None, None))[1])
    df_map = df_map.dropna(subset=["lat", "lon"])

    val_min = df_map[indicateur_carte].min()
    val_max = df_map[indicateur_carte].max()
    norm = mcolors.Normalize(vmin=val_min, vmax=val_max)
    cmap = plt.cm.RdYlGn_r

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor("#fff0f5")
    fig.patch.set_facecolor("#ffe4ec")
    ax.set_xlim(-5.5, 10)
    ax.set_ylim(41, 52)
    ax.set_aspect("equal")

    sc = ax.scatter(
        df_map["lon"], df_map["lat"],
        c=df_map[indicateur_carte],
        cmap=cmap, norm=norm,
        s=320, alpha=0.88,
        edgecolors="white", linewidths=0.8, zorder=5
    )

    for _, row_map in df_map.iterrows():
        ax.annotate(
            row_map["code"],
            (row_map["lon"], row_map["lat"]),
            ha="center", va="center",
            fontsize=5.5, fontweight="bold",
            color="white", zorder=6
        )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(short_names.get(indicateur_carte, indicateur_carte), fontsize=10)

    ax.set_title(
        f"{short_names.get(indicateur_carte, indicateur_carte)} — France",
        fontsize=14, fontweight="bold", color="#c2185b", pad=15
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3, linestyle="--")

    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    col1.success("🟢 Valeur faible")
    col2.warning("🟡 Valeur moyenne")
    col3.error("🔴 Valeur élevée")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔴 Top 5 les plus élevés")
        top5 = df_map.nlargest(5, indicateur_carte)[["libelle", indicateur_carte]]
        top5.columns = ["Département", short_names.get(indicateur_carte, indicateur_carte)]
        st.dataframe(top5.reset_index(drop=True), use_container_width=True)
    with c2:
        st.markdown("### 🟢 Top 5 les plus bas")
        bot5 = df_map.nsmallest(5, indicateur_carte)[["libelle", indicateur_carte]]
        bot5.columns = ["Département", short_names.get(indicateur_carte, indicateur_carte)]
        st.dataframe(bot5.reset_index(drop=True), use_container_width=True)

# =====================
# PAGE : DASHBOARD
# =====================
elif page == "📊 Dashboard":
    st.title("🌸 Territory Scope Sonia — Dashboard")
    name = st.selectbox("Choisir un département", territory_names)
    row = df[df["libelle"] == name].iloc[0]

    st.markdown("---")
    st.subheader(f"🏛️ {name} (Dép. {row['code']})")

    cols = st.columns(len(numeric_cols))
    for i, col in enumerate(numeric_cols):
        val = row[col]
        moy = national_means[col]
        delta = val - moy
        label = short_names.get(col, col[:30])
        try:
            cols[i].metric(
                label=label,
                value=f"{float(val):,.1f}".replace(",", " "),
                delta=f"{delta:+.1f} vs France"
            )
        except Exception:
            cols[i].metric(label=label, value=str(val))

    st.markdown("---")
    st.subheader("📋 Territoire vs Moyenne nationale")
    data_table = []
    for col in numeric_cols:
        data_table.append({
            "Indicateur": short_names.get(col, col),
            "Département": round(float(row[col]), 2),
            "Moyenne France": round(float(national_means[col]), 2),
            "Écart": round(float(row[col]) - float(national_means[col]), 2)
        })
    st.dataframe(pd.DataFrame(data_table), use_container_width=True)

    st.subheader("📊 Graphique comparatif")
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#ffe4ec")
    ax.set_facecolor("#fff0f5")
    x = np.arange(len(numeric_cols))
    w = 0.35
    labels = [short_names.get(c, c[:20]) for c in numeric_cols]
    ax.bar(x - w/2, [float(row[c]) for c in numeric_cols], width=w, label=name, color="#f48fb1")
    ax.bar(x + w/2, [float(national_means[c]) for c in numeric_cols], width=w, label="Moyenne France", color="#c2185b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylabel("Valeur")
    st.pyplot(fig)

# =====================
# PAGE : COMPARAISON
# =====================
elif page == "📈 Comparaison":
    st.title("🌸 Territory Scope Sonia — Comparaison")
    col1, col2 = st.columns(2)
    with col1:
        A = st.selectbox("Département A", territory_names, key="A")
    with col2:
        B = st.selectbox("Département B", territory_names, index=1, key="B")

    rowA = df[df["libelle"] == A].iloc[0]
    rowB = df[df["libelle"] == B].iloc[0]

    st.markdown("---")
    data_comp = []
    for col in numeric_cols:
        data_comp.append({
            "Indicateur": short_names.get(col, col),
            A: round(float(rowA[col]), 2),
            B: round(float(rowB[col]), 2),
            "Moyenne France": round(float(national_means[col]), 2),
        })
    st.dataframe(pd.DataFrame(data_comp), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#ffe4ec")
    ax.set_facecolor("#fff0f5")
    x = np.arange(len(numeric_cols))
    w = 0.25
    labels = [short_names.get(c, c[:20]) for c in numeric_cols]
    ax.bar(x - w, [float(rowA[c]) for c in numeric_cols], width=w, label=A, color="#f48fb1")
    ax.bar(x,     [float(rowB[c]) for c in numeric_cols], width=w, label=B, color="#ad1457")
    ax.bar(x + w, [float(national_means[c]) for c in numeric_cols], width=w, label="France", color="#ff8a65")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylabel("Valeur")
    st.pyplot(fig)

# =====================
# PAGE : CLASSEMENT
# =====================
else:
    st.title("🌸 Territory Scope Sonia — Classement")
    indicateur = st.selectbox(
        "Choisir un indicateur",
        numeric_cols,
        format_func=lambda x: short_names.get(x, x)
    )
    ordre = st.radio("Ordre", ["Du plus élevé au plus bas", "Du plus bas au plus élevé"], horizontal=True)
    ascending = ordre == "Du plus bas au plus élevé"

    df_rank = df[["libelle", indicateur]].copy()
    df_rank.columns = ["Département", short_names.get(indicateur, indicateur)]
    df_rank = df_rank.sort_values(by=df_rank.columns[1], ascending=ascending).reset_index(drop=True)
    df_rank.index += 1
    st.dataframe(df_rank, use_container_width=True)

    st.subheader("Top 10")
    top10 = df_rank.head(10)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#ffe4ec")
    ax.set_facecolor("#fff0f5")
    ax.barh(top10["Département"][::-1], top10.iloc[::-1, 1], color="#f48fb1")
    ax.set_xlabel(short_names.get(indicateur, indicateur))
    st.pyplot(fig)
