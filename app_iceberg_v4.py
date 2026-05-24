"""
╔══════════════════════════════════════════════════════════════════╗
║  ICEBERG v5 — Intelligence Territoriale · Dép. 91 & 94          ║
║  Refonte complète : ML prédictif · Timeline · 3D amélioré       ║
║  Architecture modulaire · Performance optimisée                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════
# 0. IMPORTS
# ══════════════════════════════════════════════════════════════════
import json, os, random, datetime, io
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False

# LLM
try:
    from mistralai import Mistral as MistralClient
except ImportError:
    MistralClient = None
try:
    from groq import Groq as _GroqClient
    _GROQ_OK = True
except ImportError:
    _GroqClient = None
    _GROQ_OK = False

# ML (optionnel — graceful degradation)
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    _ML_OK = True
except ImportError:
    _ML_OK = False

# ══════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY",    "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")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "riVNPI2YPh7aQQHmU9t1bdzLqaqXkFTD")
USE_MISTRAL     = True

APP_VERSION     = "5.0"
DEPT_MAP        = {"91": "Essonne", "94": "Val-de-Marne"}
SCORE_SEUILS    = {"prioritaire": 0.70, "favorable": 0.50, "possible": 0.30}

# Couleurs zones
ZONE_COLORS = {
    "Zone Prioritaire": [22, 163, 74,  230],
    "Zone Favorable":   [59, 130, 246, 220],
    "Zone Possible":    [251,191,  36, 220],
    "Non Recommandé":   [203,213, 225, 160],
}

# ══════════════════════════════════════════════════════════════════
# 2. PAGE CONFIG & CSS GLOBAL
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ICEBERG v5",
    layout="wide",
    page_icon="🧊",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:        #F0F2F7;
  --surface:   #FFFFFF;
  --surface-2: #F8FAFC;
  --border:    #E2E8F2;
  --border-2:  #C8D5E8;
  --text-1:    #0A0F1E;
  --text-2:    #3D4A63;
  --text-3:    #8895AA;
  --blue:      #1A56DB;
  --blue-lt:   #EBF1FF;
  --blue-md:   #B8CCFF;
  --green:     #059669;
  --green-lt:  #D1FAE5;
  --amber:     #D97706;
  --amber-lt:  #FEF3C7;
  --red:       #DC2626;
  --red-lt:    #FEE2E2;
  --purple:    #6D28D9;
  --purple-lt: #EDE9FE;
  --cyan:      #0891B2;
  --cyan-lt:   #CFFAFE;
  --r:         16px;
  --r-lg:      22px;
  --sh:        0 1px 3px rgba(10,15,30,.06), 0 2px 8px rgba(10,15,30,.04);
  --sh-md:     0 4px 16px rgba(10,15,30,.10), 0 2px 4px rgba(10,15,30,.06);
  --sh-lg:     0 8px 32px rgba(10,15,30,.14), 0 4px 8px rgba(10,15,30,.08);
  --sh-glow:   0 0 0 3px rgba(26,86,219,.15);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
  background: var(--bg);
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
  color: var(--text-1);
}
.main .block-container { padding: 1.75rem 2.25rem; max-width: 1600px; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: #FFFFFF;
  border-right: 1px solid #E2E8F2;
  box-shadow: 2px 0 12px rgba(10,15,30,.06);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color: #3D4A63 !important; }

/* ── HIDE STREAMLIT UI ── */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebarNavToggleButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[aria-label="Open sidebar"],
button[aria-label="Close sidebar"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"] {
  display: none !important; visibility: hidden !important;
  width: 0 !important; height: 0 !important;
}
[data-testid="stSidebar"] {
  transform: none !important; min-width: 290px !important; max-width: 305px !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
  transform: none !important; display: flex !important;
  visibility: visible !important; margin-left: 0 !important;
}

/* ── TYPOGRAPHY ── */
h1 { font-size: 26px !important; font-weight: 800 !important; color: var(--text-1) !important; letter-spacing: -.6px !important; }
h2 { font-size: 19px !important; font-weight: 700 !important; color: var(--text-1) !important; letter-spacing: -.3px !important; }
h3 { font-size: 11px !important; font-weight: 700 !important; color: var(--text-3) !important; text-transform: uppercase !important; letter-spacing: .9px !important; }

/* ── METRICS ── */
[data-testid="metric-container"] {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r); padding: 18px 20px 14px;
  box-shadow: var(--sh); transition: all .18s ease;
}
[data-testid="metric-container"]:hover {
  border-color: var(--blue-md); box-shadow: var(--sh-md); transform: translateY(-2px);
}
[data-testid="metric-container"] label {
  color: var(--text-3) !important; font-size: 10px !important;
  font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: .8px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--text-1) !important; font-size: 26px !important;
  font-weight: 800 !important; letter-spacing: -.6px !important;
}

/* ── BUTTONS ── */
.stButton > button {
  background: var(--blue) !important; color: #fff !important;
  border: none !important; border-radius: 11px !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 600 !important; font-size: 13px !important;
  padding: 9px 18px !important;
  box-shadow: 0 2px 8px rgba(26,86,219,.30) !important;
  transition: all .15s !important;
}
.stButton > button:hover {
  background: #1344B8 !important;
  box-shadow: 0 4px 16px rgba(26,86,219,.45) !important;
  transform: translateY(-1px) !important;
}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {
  background: var(--surface) !important; color: var(--blue) !important;
  border: 1.5px solid var(--blue) !important; border-radius: 11px !important;
  font-weight: 600 !important;
}
.stDownloadButton > button:hover {
  background: var(--blue-lt) !important;
  box-shadow: 0 2px 10px rgba(26,86,219,.15) !important;
}

/* ── INPUTS ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important; border-radius: 11px !important;
  font-size: 13px !important; color: var(--text-1) !important;
}
.stSelectbox > div > div:focus-within { border-color: var(--blue) !important; box-shadow: var(--sh-glow) !important; }
.stTextInput > div > div > input {
  background: var(--surface) !important; border: 1.5px solid var(--border) !important;
  border-radius: 11px !important; color: var(--text-1) !important;
  font-size: 13px !important; padding: 10px 14px !important;
}
.stTextInput > div > div > input:focus { border-color: var(--blue) !important; box-shadow: var(--sh-glow) !important; }

/* ── SLIDER ── */
.stSlider > div > div > div { background: var(--blue) !important; }
.stSlider > div > div > div > div { background: var(--blue) !important; box-shadow: 0 0 0 4px rgba(26,86,219,.18) !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface-2); border-radius: 13px; padding: 4px; gap: 2px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: var(--text-3) !important;
  border-radius: 10px !important; font-size: 12px !important; font-weight: 600 !important;
  padding: 8px 16px !important; transition: all .15s !important;
}
.stTabs [aria-selected="true"] {
  background: var(--surface) !important; color: var(--text-1) !important;
  font-weight: 700 !important; box-shadow: var(--sh) !important;
}

/* ── DATAFRAME ── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: var(--r) !important; overflow: hidden !important; box-shadow: var(--sh) !important; }
.stDataFrame table { background: var(--surface) !important; }
.stDataFrame th { background: #F4F7FC !important; color: var(--text-3) !important; font-size: 10px !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: .7px !important; border-bottom: 1px solid var(--border) !important; padding: 11px 14px !important; }
.stDataFrame td { color: var(--text-2) !important; font-size: 12px !important; border-bottom: 1px solid #F4F7FC !important; padding: 10px 14px !important; }

/* ── RADIO SIDEBAR ── */
[data-testid="stSidebar"] .stRadio label {
  background: transparent !important; border: none !important;
  border-radius: 10px !important; padding: 9px 13px !important;
  color: #3D4A63 !important; font-size: 13px !important; font-weight: 500 !important;
  display: flex !important; align-items: center !important; gap: 9px !important;
  transition: all .15s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: #EBF1FF !important; color: #1A56DB !important;
}

/* ── ALERTBOX ── */
.stAlert { border-radius: var(--r) !important; font-size: 12px !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #EEF2F8; }
::-webkit-scrollbar-thumb { background: #B8C5D6; border-radius: 3px; }

/* ── PYDECK ── */
.stDeckGlJsonChart { border-radius: var(--r-lg) !important; overflow: hidden !important; border: 1px solid var(--border) !important; box-shadow: var(--sh) !important; }

/* ── HR ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 20px 0 !important; }

/* ── ML BADGE ── */
.ml-badge {
  display: inline-flex; align-items: center; gap: 5px;
  background: linear-gradient(135deg,#6D28D9,#8B5CF6);
  color: #fff; padding: 3px 10px; border-radius: 20px;
  font-size: 10px; font-weight: 700; letter-spacing: .4px;
}

/* ── TIMELINE ── */
.timeline-bar {
  height: 6px; border-radius: 3px; overflow: hidden;
  background: var(--border); margin: 4px 0;
}
.timeline-fill {
  height: 100%; border-radius: 3px;
  background: linear-gradient(90deg, var(--blue), #3B82F6);
  transition: width .6s ease;
}

/* ── CARD HOVER ── */
.iceberg-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r); padding: 16px 18px;
  box-shadow: var(--sh); transition: all .18s ease;
  cursor: default;
}
.iceberg-card:hover {
  border-color: var(--blue-md); box-shadow: var(--sh-md);
  transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 3. HELPERS GÉNÉRAUX
# ══════════════════════════════════════════════════════════════════
def safe_val(val, decimals=0):
    try:
        v = float(val)
        if np.isnan(v): return "N/A"
        if decimals == 0: return f"{int(v):,}".replace(",", "\u00a0")
        return round(v, decimals)
    except:
        return "N/A"

def score_badge(score: float, prefix: str = "") -> str:
    """Retourne un badge HTML coloré selon le score (0-1)."""
    pct = int(score * 100)
    if score >= 0.70: bg, fg = "#DCFCE7", "#15803D"
    elif score >= 0.50: bg, fg = "#DBEAFE", "#1D4ED8"
    elif score >= 0.30: bg, fg = "#FEF3C7", "#B45309"
    else: bg, fg = "#FEE2E2", "#B91C1C"
    return (f'<span style="background:{bg};color:{fg};padding:2px 9px;'
            f'border-radius:20px;font-size:11px;font-weight:700;">'
            f'{prefix}{pct}%</span>')

def urgence_badge(urgence: str) -> str:
    colors = {
        "Critique": ("#FEE2E2","#DC2626"),
        "Élevé":    ("#FEF3C7","#D97706"),
        "Modéré":   ("#DBEAFE","#1D4ED8"),
        "Faible":   ("#D1FAE5","#059669"),
    }
    bg, fg = colors.get(urgence, ("#F1F5F9","#64748B"))
    return (f'<span style="background:{bg};color:{fg};padding:2px 9px;'
            f'border-radius:20px;font-size:11px;font-weight:700;">{urgence}</span>')

def page_header(icon: str, title: str, subtitle: str, badge: str = ""):
    badge_html = f'<span class="ml-badge">🤖 {badge}</span>' if badge else ""
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0B1F5C 0%,#1A56DB 55%,#2563EB 100%);'
        f'border-radius:{22}px;padding:26px 30px;margin-bottom:26px;'
        f'box-shadow:0 8px 32px rgba(26,86,219,.25);">'
        f'<div style="display:flex;align-items:center;gap:14px;">'
        f'<div style="width:50px;height:50px;background:rgba(255,255,255,.14);border-radius:14px;'
        f'display:flex;align-items:center;justify-content:center;font-size:24px;flex-shrink:0;">{icon}</div>'
        f'<div style="flex:1;">'
        f'<div style="font-size:20px;font-weight:800;color:#fff;letter-spacing:-.4px;line-height:1.25;">{title} {badge_html}</div>'
        f'<div style="font-size:12px;color:rgba(255,255,255,.60);margin-top:3px;">{subtitle}</div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

def metric_card(label: str, value: str, delta: str = "", color: str = "var(--blue)",
                bg: str = "var(--surface)") -> str:
    delta_html = ""
    if delta:
        d_color = "#DC2626" if delta.startswith("↗") or delta.startswith("+") else "#059669"
        delta_html = f'<div style="font-size:11px;color:{d_color};font-weight:600;margin-top:5px;">{delta}</div>'
    return (
        f'<div class="iceberg-card">'
        f'<div style="font-size:10px;font-weight:700;color:var(--text-3);'
        f'text-transform:uppercase;letter-spacing:.8px;margin-bottom:7px;">{label}</div>'
        f'<div style="font-size:28px;font-weight:800;color:{color};letter-spacing:-.5px;line-height:1;">{value}</div>'
        f'{delta_html}</div>'
    )


# ══════════════════════════════════════════════════════════════════
# 4. CHARGEMENT & ENRICHISSEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_data() -> pd.DataFrame:
    # ── Chargement source ──────────────────────────────────────────
    for path, fmt in [
        ("data-iceberg-v4_rows.csv", "csv"),
        ("iceberg_dataset_v4.json", "json"),
        ("iceberg_dataset_v3.json", "json"),
    ]:
        if os.path.exists(path):
            if fmt == "csv":
                df = pd.read_csv(path, dtype={"code_commune": str, "code_postal": str})
            else:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
                df = pd.DataFrame(data)
            break
    else:
        st.error("❌ Aucun fichier de données trouvé.")
        st.stop()

    # ── Mappings scores pipeline → app ────────────────────────────
    _map = {
        "score_attractivite":        "potentiel_investissement",
        "score_signal_faible":       "score_fragilite",
        "score_reindustrialisation": "score_emergence",
        "score_employabilite":       "score_momentum",
    }
    for dst, src in _map.items():
        if src in df.columns:
            df[dst] = df[src]

    if "score_desert_medical" not in df.columns or df["score_desert_medical"].isna().all():
        df["score_desert_medical"] = df.get("score_freins_invisibles", 0)
    if "score_desert_commercial" not in df.columns or df["score_desert_commercial"].isna().all():
        df["score_desert_commercial"] = df.get("risque_credit_local", 0)
    df["score_desert_mobilite"] = df.get("risque_credit_local", df.get("score_desert_mobilite", 0))

    # ── Colonnes numériques (0 si absentes) ───────────────────────
    _cols_num = [
        "taux_chomage","revenu_median","taux_pauvrete","prix_m2_median",
        "nb_entreprises_actives","population","surface_km2","densite_hab_km2",
        "entreprises_1000hab","nb_gares","nb_transactions",
        "nb_medecin_generaliste","nb_medecin_specialiste","nb_dentiste",
        "nb_pharmacie","nb_hopital","nb_urgences","nb_infirmier","nb_kinesitherapeute",
        "medecins_10k_hab",
        "nb_supermarche","nb_epicerie","nb_boulangerie","nb_banque","nb_poste",
        "nb_boucherie","est_desert_medical","est_desert_commercial",
        "nb_ecole_primaire","nb_college","nb_lycee",
        # Scores enrichis
        "score_attractivite","score_signal_faible","score_reindustrialisation",
        "score_employabilite","score_desert_medical","score_desert_commercial","score_desert_mobilite",
    ]
    for c in _cols_num:
        if c not in df.columns: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # ── Normalisation scores 0-1 ──────────────────────────────────
    for sc in ["score_attractivite","score_signal_faible","score_desert_medical",
               "score_desert_commercial","score_desert_mobilite","score_reindustrialisation","score_employabilite"]:
        mx = df[sc].max(); mn = df[sc].min()
        if mx > mn: df[sc] = (df[sc] - mn) / (mx - mn)
        df[sc] = df[sc].clip(0, 1).fillna(0)

    # ── Métadonnées ───────────────────────────────────────────────
    df["dept_nom"] = df["code_dept"].astype(str).map(DEPT_MAP).fillna("Inconnu")

    # ── Catégories globales ───────────────────────────────────────
    def _cat_zone(s):
        if s >= SCORE_SEUILS["prioritaire"]: return "Zone Prioritaire"
        if s >= SCORE_SEUILS["favorable"]:   return "Zone Favorable"
        if s >= SCORE_SEUILS["possible"]:    return "Zone Possible"
        return "Non Recommandé"

    df["cat_attractivite"]  = df["score_attractivite"].apply(_cat_zone)
    df["cat_signal_faible"] = df["score_signal_faible"].apply(
        lambda s: "Signal Fort" if s >= .65 else ("Signal Modéré" if s >= .40 else "Signal Faible"))
    for col in ["cat_desert_medical","cat_desert_commercial","cat_desert_mobilite"]:
        sc = col.replace("cat_","score_")
        df[col] = df[sc].apply(lambda s: "Fort" if s>=.65 else ("Modéré" if s>=.40 else "Faible"))

    # ── Sous-catégories médicales ─────────────────────────────────
    pop = df["population"].clip(lower=1)
    if df["medecins_10k_hab"].sum() == 0:
        df["medecins_10k_hab"] = (df["nb_medecin_generaliste"] * 10000 / pop).clip(0, 100)

    def _cat_med(r, s1=5, s2=12):
        if r == 0: return "Désert"
        if r < s1: return "Sous-doté"
        if r < s2: return "Correct"
        return "Bien doté"

    df["cat_desert_generaliste"]   = df["medecins_10k_hab"].apply(_cat_med)
    df["cat_desert_specialiste"]   = (df["nb_medecin_specialiste"]*10000/pop).apply(lambda r: _cat_med(r,3,8))
    df["cat_desert_dentiste"]      = (df["nb_dentiste"]*10000/pop).apply(lambda r: _cat_med(r,3,8))
    df["score_desert_generaliste"] = (1-(df["medecins_10k_hab"]/15).clip(0,1))
    df["score_desert_specialiste"] = (1-((df["nb_medecin_specialiste"]*10000/pop)/10).clip(0,1))
    df["score_desert_dentiste"]    = (1-((df["nb_dentiste"]*10000/pop)/10).clip(0,1))

    # ── Sous-catégories commerciales ──────────────────────────────
    for col2, seuil in [("epicerie",5),("boulangerie",5),("banque",5),("boucherie",5),("poste",3)]:
        df[f"score_desert_{col2}"] = (1-((df[f"nb_{col2}"]*10000/pop)/seuil).clip(0,1))
        df[f"cat_desert_{col2}"]   = df[f"score_desert_{col2}"].apply(
            lambda s: "Fort" if s>=.65 else ("Modéré" if s>=.40 else "Faible"))

    # ── Sous-catégories scolaires ─────────────────────────────────
    for col2, seuil in [("primaire",8),("college",3),("lycee",2)]:
        nb_col = {"primaire":"nb_ecole_primaire","college":"nb_college","lycee":"nb_lycee"}[col2]
        df[f"score_desert_{col2}"] = (1-((df[nb_col]*10000/pop)/seuil).clip(0,1))
        df[f"cat_desert_{col2}"]   = df[f"score_desert_{col2}"].apply(
            lambda s: "Fort" if s>=.65 else ("Modéré" if s>=.40 else "Faible"))

    # ── Score composite global (v5 : combinaison pondérée) ───────
    df["score_global_v5"] = (
        df["score_attractivite"]        * 0.30 +
        df["score_reindustrialisation"] * 0.25 +
        df["score_employabilite"]       * 0.20 +
        (1 - df["score_desert_medical"])* 0.15 +
        (1 - df["score_signal_faible"]) * 0.10
    ).clip(0, 1)

    return df


# ══════════════════════════════════════════════════════════════════
# 5. MODULE ML — PRÉDICTIF (scikit-learn)
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les features ML et ajoute les prédictions au DataFrame."""
    if not _ML_OK:
        df["pred_attractivite_2026"] = df["score_attractivite"]
        df["pred_chomage_2026"]      = df["taux_chomage"]
        df["risk_score"]             = 0.5
        df["opportunity_score"]      = df["score_attractivite"]
        df["ml_cluster"]             = "N/A"
        return df

    feats = [
        "taux_chomage","taux_pauvrete","revenu_median","prix_m2_median",
        "nb_entreprises_actives","population","densite_hab_km2",
        "entreprises_1000hab","nb_gares","score_attractivite",
        "score_signal_faible","score_desert_medical","score_desert_commercial",
    ]
    X = df[feats].fillna(0).values

    # ── Modèle 1 : Prédiction attractivité 2026 ──────────────────
    y_attr = df["score_attractivite"].values
    model_attr = Pipeline([("scl", StandardScaler()), ("gb", GradientBoostingRegressor(n_estimators=80, random_state=42))])
    # Simulation de variation temporelle (+/- bruit)
    rng = np.random.RandomState(42)
    y_2026 = (y_attr + rng.normal(0, 0.05, len(y_attr))).clip(0, 1)
    model_attr.fit(X, y_2026)
    df["pred_attractivite_2026"] = model_attr.predict(X).clip(0, 1)

    # ── Modèle 2 : Score de risque ───────────────────────────────
    y_risk = (
        df["score_signal_faible"] * 0.40 +
        df["taux_pauvrete"].clip(0,50) / 50 * 0.30 +
        df["taux_chomage"].clip(0,30) / 30 * 0.30
    ).clip(0, 1).values
    model_risk = Pipeline([("scl", StandardScaler()), ("gb", GradientBoostingRegressor(n_estimators=60, random_state=0))])
    model_risk.fit(X, y_risk)
    df["risk_score"] = model_risk.predict(X).clip(0, 1)

    # ── Modèle 3 : Score opportunité ─────────────────────────────
    df["opportunity_score"] = (
        df["pred_attractivite_2026"] * 0.50 +
        (1 - df["risk_score"])       * 0.30 +
        df["score_global_v5"]        * 0.20
    ).clip(0, 1)

    # ── Clustering simple (5 profils) ────────────────────────────
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5, random_state=42, n_init="auto")
    labels = km.fit_predict(StandardScaler().fit_transform(X))
    cluster_names = {0:"Territoire dynamique",1:"Zone de vigilance",2:"Désert de services",
                     3:"Potentiel émergent",4:"Territoire stable"}
    df["ml_cluster"] = [cluster_names.get(l,"Autre") for l in labels]

    # ── Prédiction chômage 2026 ───────────────────────────────────
    drift = rng.normal(0, 0.3, len(df))
    df["pred_chomage_2026"] = (df["taux_chomage"] + drift).clip(0, 40)

    return df


def generate_timeline_data(df: pd.DataFrame, ville: str) -> pd.DataFrame:
    """Génère des données temporelles simulées 2019-2030 pour une commune (tous indicateurs)."""
    row = df[df["ville"] == ville].iloc[0]
    random.seed(hash(ville) % 9999)
    annees     = list(range(2019, 2031))
    base_attr  = float(row["score_attractivite"])
    base_cho   = float(row["taux_chomage"])
    base_ent   = float(row["nb_entreprises_actives"])
    base_rev   = float(row.get("revenu_median",    22000))
    base_prix  = float(row.get("prix_m2_median",    3000))
    base_pauv  = float(row.get("taux_pauvrete",       15))
    base_med   = float(row.get("score_desert_medical", 0.5))
    base_opp   = float(row.get("opportunity_score", base_attr))
    base_risk  = float(row.get("risk_score",          0.4))
    cluster    = str(row.get("ml_cluster", "N/A"))

    trend_map = {
        "Territoire dynamique": (+0.025, -0.15, +0.05),
        "Potentiel émergent":   (+0.020, -0.10, +0.04),
        "Territoire stable":    (+0.005, -0.05, +0.02),
        "Zone de vigilance":    (-0.010, +0.10, -0.01),
        "Désert de services":   (-0.015, +0.15, -0.02),
    }
    t_attr, t_cho, t_ent = trend_map.get(cluster, (+0.010, -0.05, +0.02))

    records = []
    for i, y in enumerate(annees):
        is_pred = y >= 2025
        noise   = 1 if not is_pred else 0.5
        yr_attr  = round(max(0, min(1,   base_attr  + t_attr*(i-3) + random.gauss(0, 0.025*noise))), 3)
        yr_cho   = round(max(0, min(35,  base_cho   + t_cho *(i-3) + random.gauss(0, 0.20 *noise))), 1)
        yr_ent   = int(max(0,            base_ent   * (1 + t_ent*(i-3) + random.gauss(0, 0.015*noise))))
        yr_rev   = int(max(12000,        base_rev   * (1 + 0.018*(i-3) + random.gauss(0, 0.008*noise))))
        yr_prix  = int(max(1000,         base_prix  * (1 + 0.022*(i-3) + random.gauss(0, 0.010*noise))))
        yr_pauv  = round(max(0, min(60,  base_pauv  - 0.12*(i-3)      + random.gauss(0, 0.15 *noise))), 1)
        yr_med   = round(max(0, min(1,   base_med   - 0.012*(i-3)     + random.gauss(0, 0.015*noise))), 3)
        yr_opp   = round(max(0, min(1,   base_opp   + t_attr*0.8*(i-3)+ random.gauss(0, 0.02 *noise))), 3)
        yr_risk  = round(max(0, min(1,   base_risk  - t_attr*0.5*(i-3)+ random.gauss(0, 0.02 *noise))), 3)
        yr_cat   = ("Zone Prioritaire" if yr_attr >= .70 else
                    ("Zone Favorable"  if yr_attr >= .50 else
                    ("Zone Possible"   if yr_attr >= .30 else "Non Recommandé")))
        records.append({
            "annee": y, "attractivite": yr_attr, "chomage": yr_cho,
            "entreprises": yr_ent, "revenu": yr_rev, "prix_m2": yr_prix,
            "pauvrete": yr_pauv, "desert_med": yr_med,
            "opportunite": yr_opp, "risque": yr_risk,
            "cat_zone": yr_cat,
            "type": "Prévision IA" if is_pred else "Historique",
        })
    return pd.DataFrame(records)


def project_df_to_year(df: pd.DataFrame, annee: int) -> pd.DataFrame:
    """Recalcule les scores de toutes les communes pour une année cible (2026-2030)."""
    if annee <= 2026:
        return df.copy()
    df2 = df.copy()
    rng   = np.random.RandomState(annee)
    delta = annee - 2026   # 2026 = base actuelle
    cluster_trend = {
        "Territoire dynamique": +0.025, "Potentiel émergent":  +0.020,
        "Territoire stable":    +0.005, "Zone de vigilance":   -0.010,
        "Désert de services":   -0.015,
    }
    for idx, row in df2.iterrows():
        trend    = cluster_trend.get(str(row.get("ml_cluster","N/A")), 0.008)
        new_attr = max(0.0, min(1.0, float(row["score_attractivite"]) + trend*delta + rng.normal(0,0.015)))
        new_opp  = max(0.0, min(1.0, float(row.get("opportunity_score", new_attr)) + trend*0.8*delta + rng.normal(0,0.01)))
        new_risk = max(0.0, min(1.0, float(row.get("risk_score",0.4)) - trend*0.5*delta + rng.normal(0,0.01)))
        df2.at[idx, "score_attractivite"]     = new_attr
        df2.at[idx, "pred_attractivite_2026"] = new_attr
        df2.at[idx, "opportunity_score"]      = new_opp
        df2.at[idx, "risk_score"]             = new_risk
    sc = df2["score_attractivite"]
    mn, mx = sc.min(), sc.max()
    df2["score_custom"] = ((sc-mn)/(mx-mn)).clip(0,1) if mx > mn else sc.clip(0,1)
    df2["cat_custom"] = df2["score_custom"].apply(
        lambda s: "Zone Prioritaire" if s>=.70 else ("Zone Favorable" if s>=.50 else ("Zone Possible" if s>=.30 else "Non Recommandé")))
    return df2


# ══════════════════════════════════════════════════════════════════
# 6. DONNÉES INITIALES SESSION STATE
# ══════════════════════════════════════════════════════════════════
def init_session_state():
    """Initialise les valeurs de session state une seule fois."""
    if "role" not in st.session_state:
        st.session_state.role = None
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "Bonjour 👋 Je suis votre assistant IA ICEBERG v5, spécialisé en intelligence "
                "territoriale pour les départements 91 et 94. Posez-moi n'importe quelle question "
                "sur les 241 communes — attractivité, déserts, signaux faibles, comparaisons, "
                "prévisions 2026."
            ),
        }]
    if "alertes_territoire" not in st.session_state:
        st.session_state.alertes_territoire = {
            "Isabelle COLIN": {"alertes": [
                {"type":"🏘️ Social — précarité / pauvreté","ville":"Grigny","urgence":"Critique",
                 "description":"Taux de pauvreté record à 48%. Nombreuses familles sans ressources stables.","actions":"Réunion préfecture planifiée","nb_personnes":3200,"date":"17/04/2026 08:30"},
                {"type":"💼 Emploi — fermeture d'entreprise","ville":"Grigny","urgence":"Élevé",
                 "description":"Fermeture d'un entrepôt logistique, 180 emplois menacés.","actions":"Contact France Travail initié","nb_personnes":180,"date":"16/04/2026 14:00"},
                {"type":"🏥 Désert médical — manque de médecins","ville":"Grigny","urgence":"Élevé",
                 "description":"Départ de 2 médecins généralistes. 8000 habitants sans médecin traitant.","actions":"","nb_personnes":8000,"date":"15/04/2026 10:00"},
            ]},
            "Frédéric DUTERTRE": {"alertes": [
                {"type":"💼 Emploi — fermeture d'entreprise","ville":"Évry-Courcouronnes","urgence":"Critique",
                 "description":"Fin du contrat Safran : 3400 emplois menacés sur le bassin d'Évry.","actions":"Cellule de crise activée","nb_personnes":3400,"date":"17/04/2026 09:00"},
                {"type":"🏗️ Foncier — terrain disponible signalé","ville":"Évry-Courcouronnes","urgence":"Modéré",
                 "description":"Terrain de 2 hectares disponible zone industrielle.","actions":"Dossier transmis à BPI","nb_personnes":0,"date":"14/04/2026 11:30"},
            ]},
            "Nasser BENKHEMIS": {"alertes": [
                {"type":"🛒 Désert commercial — fermeture de commerces","ville":"Créteil","urgence":"Modéré",
                 "description":"Fermeture de 12 commerces en centre-ville en 3 mois.","actions":"Réunion CCI programmée","nb_personnes":500,"date":"16/04/2026 16:00"},
                {"type":"🚉 Désert mobilité — manque de transport","ville":"Créteil","urgence":"Faible",
                 "description":"Fréquence du Bus 393 réduite de 30% depuis janvier.","actions":"","nb_personnes":2000,"date":"15/04/2026 09:00"},
            ]},
        }
    if "signal_selectionne" not in st.session_state:
        st.session_state.signal_selectionne = 0


# ══════════════════════════════════════════════════════════════════
# 7. SIDEBAR
# ══════════════════════════════════════════════════════════════════
def render_sidebar(df: pd.DataFrame):
    _n      = len(df)
    _prio   = len(df[df["cat_attractivite"] == "Zone Prioritaire"])
    _sig    = len(df[df["cat_signal_faible"] == "Signal Fort"])
    _desert = len(df[df["cat_desert_medical"] == "Fort"])

    with st.sidebar:
        # Header brand
        st.markdown(f"""
        <div style="padding:22px 18px 14px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
            <div style="width:40px;height:40px;background:linear-gradient(135deg,#1A56DB,#3B82F6);
                 border-radius:12px;display:flex;align-items:center;justify-content:center;
                 font-size:18px;box-shadow:0 4px 14px rgba(26,86,219,.25);">🧊</div>
            <div>
              <div style="font-size:16px;font-weight:800;color:#0A0F1E;letter-spacing:.3px;">ICEBERG <span style="font-size:10px;font-weight:500;color:#94A3B8;">v{APP_VERSION}</span></div>
              <div style="font-size:9px;color:#94A3B8;text-transform:uppercase;letter-spacing:1.8px;">Dép. 91 &amp; 94</div>
            </div>
          </div>

          <div style="background:#F8FAFC;border:1px solid #E2E8F2;
               border-radius:14px;padding:14px;margin-bottom:6px;">
            <div style="font-size:9px;font-weight:700;color:#94A3B8;text-transform:uppercase;
                 letter-spacing:1.2px;margin-bottom:10px;">Vue d'ensemble</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
              <div style="background:#FFFFFF;border:1px solid #E2E8F2;border-radius:10px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:800;color:#0A0F1E;">{_n}</div>
                <div style="font-size:10px;color:#64748B;margin-top:1px;">Communes</div>
              </div>
              <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:10px 12px;cursor:pointer;">
                <div style="font-size:20px;font-weight:800;color:#16A34A;">{_prio}</div>
                <div style="font-size:10px;color:#64748B;margin-top:1px;">Prioritaires ↗</div>
              </div>
              <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;padding:10px 12px;cursor:pointer;">
                <div style="font-size:20px;font-weight:800;color:#D97706;">{_sig}</div>
                <div style="font-size:10px;color:#64748B;margin-top:1px;">Signaux forts ↗</div>
              </div>
              <div style="background:#FFF1F2;border:1px solid #FECDD3;border-radius:10px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:800;color:#DC2626;">{_desert}</div>
                <div style="font-size:10px;color:#64748B;margin-top:1px;">Déserts méd.</div>
              </div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:6px;">
        </div>
        """, unsafe_allow_html=True)

        # Boutons invisibles alignés sur les cartes — directeur uniquement
        if st.session_state.get("role") == "directeur":
            _ba, _bb = st.columns(2)
            with _ba:
                if st.button("📋 Rapport Prioritaires", key="btn_goto_prio",
                             use_container_width=True, type="secondary"):
                    st.session_state["_nav_override"] = "📋  Rapport Prioritaires"
                    st.rerun()
            with _bb:
                if st.button("🚨 Rapport Signaux", key="btn_goto_sig",
                             use_container_width=True, type="secondary"):
                    st.session_state["_nav_override"] = "🚨  Rapport Signaux"
                    st.rerun()

        if st.session_state.role is not None:
            st.markdown('<p style="font-size:13px;color:#0A0F1E;text-transform:uppercase;'
                        'letter-spacing:2px;font-weight:800;padding:0 18px;'
                        'margin-bottom:4px;">Navigation</p>', unsafe_allow_html=True)
            st.markdown('<div style="padding:0 10px;">', unsafe_allow_html=True)

            if st.session_state.role == "administrateur":
                nav_items = [
                    "🏆  Classement admins",
                    "🏥  Carte Déserts",
                    "📊  Communes",
                    "💬  Assistant IA",
                ]
            else:
                nav_items = [
                    "🏆  Classement admins",
                    "⭐  Carte Attractivité",
                    "📋  Rapport Prioritaires",
                    "🚨  Rapport Signaux",
                    "🏥  Carte Déserts",
                    "📈  Indicateurs",
                    "💡  Opportunités IA",
                    "📊  Communes",
                    "💬  Assistant IA",
                ]

            # Calcul de l'index de départ (nav_override ou 0)
            _default_idx = 0
            _override = st.session_state.get("_nav_override", "")
            if _override in nav_items:
                _default_idx = nav_items.index(_override)
                del st.session_state["_nav_override"]

            page = st.radio("", nav_items, index=_default_idx, label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div style="padding:10px 18px 18px;">', unsafe_allow_html=True)
            st.markdown('<hr style="border:none;border-top:1px solid #E2E8F2;margin:0 0 10px;">', unsafe_allow_html=True)
            icon = "🏢" if st.session_state.role == "directeur" else "⚙️"
            label = "Directeur d'Agence" if st.session_state.role == "directeur" else "Administrateur"
            st.markdown(f"""
            <div style="background:#F8FAFC;border:1px solid #E2E8F2;
                 border-radius:10px;padding:10px 12px;margin-bottom:10px;">
              <div style="font-size:9px;color:#94A3B8;margin-bottom:3px;text-transform:uppercase;letter-spacing:.9px;">Connecté</div>
              <div style="font-size:13px;font-weight:600;color:#0A0F1E;">{icon} {label}</div>
            </div>""", unsafe_allow_html=True)
            if st.button("Déconnexion", use_container_width=True):
                st.session_state.role = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            page = None

    return page


# ══════════════════════════════════════════════════════════════════
# 8. PAGE LOGIN
# ══════════════════════════════════════════════════════════════════
def page_login(df: pd.DataFrame):
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;padding:60px 20px 40px;">
      <div style="width:72px;height:72px;background:linear-gradient(135deg,#1A56DB,#3B82F6);
           border-radius:22px;display:flex;align-items:center;justify-content:center;
           font-size:32px;box-shadow:0 12px 32px rgba(26,86,219,.35);margin-bottom:20px;">🧊</div>
      <div style="font-size:40px;font-weight:800;color:#0A0F1E;letter-spacing:-1.5px;">ICEBERG</div>
      <div style="font-size:11px;color:#8895AA;margin-top:8px;letter-spacing:2.5px;
           text-transform:uppercase;font-weight:600;">Intelligence Territoriale · Dép. 91 &amp; 94</div>
      <div style="width:32px;height:2px;background:linear-gradient(90deg,#1A56DB,#3B82F6);
           border-radius:2px;margin:20px 0 6px;"></div>
      <div style="font-size:14px;color:#3D4A63;font-weight:500;">Choisissez votre espace de travail</div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        ca, cb = st.columns(2)
        for col, role_key, ico, titre, desc, btn_key in [
            (ca, "directeur",      "🏢", "Directeur d'Agence",  "Analyses · Cartes · Rapports · Prédictions ML", "btn_dir"),
            (cb, "administrateur", "⚙️", "Administrateur",      "Alertes · Données complètes · Export",          "btn_adm"),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:#fff;border:1.5px solid #E2E8F2;border-radius:20px;
                     padding:34px 24px 26px;text-align:center;
                     box-shadow:0 4px 20px rgba(10,15,30,.07);">
                  <div style="width:56px;height:56px;background:linear-gradient(135deg,#EBF1FF,#C7D7FD);
                       border-radius:16px;display:flex;align-items:center;justify-content:center;
                       margin:0 auto 16px;font-size:26px;">{ico}</div>
                  <div style="font-size:16px;font-weight:700;color:#0A0F1E;margin-bottom:7px;">{titre}</div>
                  <div style="font-size:12px;color:#8895AA;line-height:1.6;">{desc}</div>
                </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Accéder →", key=btn_key, use_container_width=True):
                    st.session_state.role = role_key
                    st.rerun()

    # ── Podium villes actives ─────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _, mid2, _ = st.columns([1, 3, 1])
    with mid2:
        st.markdown("""
        <div style="text-align:center;margin-bottom:14px;">
          <div style="font-size:14px;font-weight:700;color:#0A0F1E;letter-spacing:-.2px;">🏙️ Villes les plus actives</div>
          <div style="font-size:11px;color:#8895AA;margin-top:3px;">Classement en temps réel · activité terrain</div>
        </div>""", unsafe_allow_html=True)

        villes_scores = {}
        for adm, data in st.session_state.alertes_territoire.items():
            for al in data["alertes"]:
                v = al["ville"]
                if v not in villes_scores:
                    villes_scores[v] = {"score":0,"nb":0,"admins":set(),"critiques":0}
                pts = {"Critique":40,"Élevé":25,"Modéré":15,"Faible":5}.get(al.get("urgence","Faible"),5)
                villes_scores[v]["score"] += pts + 10
                villes_scores[v]["nb"]    += 1
                villes_scores[v]["admins"].add(adm)
                if al.get("urgence") == "Critique": villes_scores[v]["critiques"] += 1

        top3 = sorted(villes_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:3]
        if top3:
            max_score = top3[0][1]["score"]
            medals = [("🥇","#D97706",130,True),("🥈","#64748B",90,False),("🥉","#B45309",60,False)]
            podium_order = [1,0,2] if len(top3)>=3 else list(range(len(top3)))
            cols_p = st.columns(3)
            for i, idx in enumerate(podium_order):
                if idx >= len(top3): continue
                ville_p, data_p = top3[idx]
                med, clr, ph, is_win = medals[idx]
                pct = int(data_p["score"]/max_score*100)
                with cols_p[i]:
                    crit_badge = (f'<span style="background:#FEE2E2;color:#DC2626;padding:1px 7px;'
                                  f'border-radius:20px;font-size:9px;font-weight:700;">'
                                  f'🔴 {data_p["critiques"]}</span>') if data_p["critiques"] else ""
                    bar_grad = {"🥇":"linear-gradient(180deg,#FDE68A,#F59E0B)",
                                "🥈":"linear-gradient(180deg,#E2E8F0,#94A3B8)",
                                "🥉":"linear-gradient(180deg,#FED7AA,#B45309)"}[med]
                    st.markdown(
                        f'<div style="background:#fff;border:1px solid #E2E8F2;'
                        f'border-radius:18px 18px 0 0;padding:20px 14px 14px;text-align:center;'
                        f'box-shadow:0 2px 10px rgba(10,15,30,.07);'
                        f'margin-top:{30-(ph-60)//3}px;">'
                        f'<div style="font-size:{"36px" if is_win else "28px"};margin-bottom:7px;">{med}</div>'
                        f'<div style="font-size:{"15px" if is_win else "13px"};font-weight:800;color:#0A0F1E;">{ville_p}</div>'
                        f'<div style="display:flex;justify-content:center;gap:5px;flex-wrap:wrap;margin:7px 0;">'
                        f'<span style="background:#EBF1FF;color:#1A56DB;padding:1px 7px;border-radius:20px;font-size:9px;font-weight:700;">{data_p["nb"]} alerte(s)</span>'
                        f'<span style="background:#D1FAE5;color:#059669;padding:1px 7px;border-radius:20px;font-size:9px;font-weight:700;">{len(data_p["admins"])} admin(s)</span>'
                        f'</div>{crit_badge}'
                        f'<div style="background:#F0F4F8;border-radius:4px;height:5px;margin:8px 0 4px;overflow:hidden;">'
                        f'<div style="width:{pct}%;height:5px;border-radius:4px;background:linear-gradient(90deg,#1A56DB,#3B82F6);"></div></div>'
                        f'<div style="font-size:{"19px" if is_win else "16px"};font-weight:800;color:{clr};margin-top:3px;">{data_p["score"]} pts</div>'
                        f'</div>'
                        f'<div style="background:{bar_grad};height:{ph}px;border-radius:0 0 8px 8px;'
                        f'display:flex;align-items:center;justify-content:center;'
                        f'font-size:22px;font-weight:900;color:rgba(255,255,255,.65);">'
                        f'{["1","2","3"][idx]}</div>',
                        unsafe_allow_html=True
                    )


# ══════════════════════════════════════════════════════════════════
# 9. PAGE CLASSEMENT ADMINS
# ══════════════════════════════════════════════════════════════════
def page_classement(df: pd.DataFrame):
    page_header("🏆", "Classement administrateurs", "Alertes territoire · Scores temps réel · Top 3 villes")

    def score_admin(alertes):
        s = 0
        for al in alertes:
            urgence_pts = {"Critique":40,"Élevé":25,"Modéré":15,"Faible":5}.get(al.get("urgence","Faible"),5)
            s += 10 + urgence_pts + min(20,len(al.get("description",""))//10) + (10 if al.get("actions","").strip() else 0)
        return s

    if st.session_state.role == "administrateur":
        tab_dep, tab_rank = st.tabs(["🚨 Déposer une alerte", "🏅 Classement"])
    else:
        tab_rank, = st.tabs(["🏅 Classement"])
        tab_dep = None

    # ── DÉPÔT D'ALERTE ───────────────────────────────────────────
    if tab_dep is not None:
        with tab_dep:
            st.markdown("""
            <div style="background:#fff;border:1px solid #E2E8F2;border-radius:16px;padding:18px 22px;margin-bottom:18px;">
              <div style="font-size:14px;font-weight:700;color:#0A0F1E;margin-bottom:4px;">🚨 Signalement territorial</div>
              <div style="font-size:12px;color:#64748B;">Remontez une situation à surveiller. Chaque alerte augmente votre score.</div>
            </div>""", unsafe_allow_html=True)

            ville_names = sorted(df["ville"].dropna().unique().tolist())
            with st.form("form_alerte_v5", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1:
                    admin_nom  = st.text_input("👤 Votre nom / identifiant", placeholder="Ex : Marie Dupont")
                    ville_al   = st.selectbox("📍 Commune concernée", ville_names)
                with c2:
                    type_al    = st.selectbox("🔖 Type d'alerte", [
                        "🏥 Désert médical — manque de médecins",
                        "🛒 Désert commercial — fermeture de commerces",
                        "🚉 Désert mobilité — manque de transport",
                        "💼 Emploi — fermeture d'entreprise",
                        "🏘️ Social — précarité / pauvreté",
                        "🏗️ Foncier — terrain disponible signalé",
                        "📢 Autre — situation à surveiller",
                    ])
                    urgence_al = st.select_slider("🔴 Urgence", ["Faible","Modéré","Élevé","Critique"], "Modéré")

                desc_al    = st.text_area("📝 Description", height=110)
                c3, c4 = st.columns(2)
                with c3: actions_al = st.text_input("✅ Actions déjà entreprises")
                with c4: nb_pers = st.number_input("👥 Nb personnes concernées", min_value=0, value=0, step=10)

                submitted = st.form_submit_button("📤 Soumettre l'alerte", use_container_width=True)
                if submitted:
                    if not admin_nom.strip() or not desc_al.strip():
                        st.error("⚠️ Renseignez votre nom et une description.")
                    else:
                        nom_k = admin_nom.strip()
                        if nom_k not in st.session_state.alertes_territoire:
                            st.session_state.alertes_territoire[nom_k] = {"alertes": []}
                        st.session_state.alertes_territoire[nom_k]["alertes"].append({
                            "type": type_al, "ville": ville_al, "urgence": urgence_al,
                            "description": desc_al.strip(), "actions": actions_al.strip(),
                            "nb_personnes": nb_pers,
                            "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
                        })
                        st.success("✅ Alerte soumise ! Votre score a été mis à jour.")
                        st.balloons()

    # ── CLASSEMENT ───────────────────────────────────────────────
    with tab_rank:
        sub_adm, sub_villes = st.tabs(["🏅 Admins", "🏙️ Top villes"])

        with sub_adm:
            if not st.session_state.alertes_territoire:
                st.info("Aucune alerte déposée pour l'instant.")
                return

            ranking = []
            for adm, data in st.session_state.alertes_territoire.items():
                als = data["alertes"]
                ranking.append({
                    "admin": adm, "score": score_admin(als), "nb_alertes": len(als),
                    "nb_critiques": sum(1 for a in als if a["urgence"]=="Critique"),
                    "villes": ", ".join(set(a["ville"] for a in als))[:40],
                })
            ranking.sort(key=lambda x: x["score"], reverse=True)

            medals = [("🥇","#D97706",130,True),("🥈","#64748B",90,False),("🥉","#B45309",60,False)]
            max_s = ranking[0]["score"] if ranking else 1
            podium_order = [1,0,2] if len(ranking)>=3 else list(range(len(ranking)))
            cols_p = st.columns(min(3,len(ranking)))
            for i, idx in enumerate(podium_order):
                if idx >= len(ranking): continue
                r = ranking[idx]; med, clr, ph, is_win = medals[idx]
                bar_grad = {"🥇":"linear-gradient(180deg,#FDE68A,#F59E0B)",
                            "🥈":"linear-gradient(180deg,#E2E8F0,#94A3B8)",
                            "🥉":"linear-gradient(180deg,#FED7AA,#B45309)"}[med]
                bg_inner = {"#D97706":"#FEF3C7","#64748B":"#F1F5F9","#B45309":"#FEF3C7"}.get(clr,"#F1F5F9")
                crit_badge = (f'<span style="background:#FEE2E2;color:#DC2626;padding:2px 8px;'
                              f'border-radius:20px;font-size:10px;font-weight:700;">'
                              f'🔴 {r["nb_critiques"]}</span>') if r["nb_critiques"] else ""
                with cols_p[i]:
                    st.markdown(
                        f'<div style="background:#fff;border:1px solid #E2E8F2;border-radius:20px 20px 0 0;'
                        f'padding:22px 16px 16px;text-align:center;box-shadow:0 2px 8px rgba(10,15,30,.07);'
                        f'margin-top:{30-(ph-60)//3}px;">'
                        f'<div style="font-size:{"38px" if is_win else "30px"};margin-bottom:8px;">{med}</div>'
                        f'<div style="font-size:{"15px" if is_win else "14px"};font-weight:800;color:#0A0F1E;">{r["admin"]}</div>'
                        f'<div style="font-size:10px;color:#8895AA;margin:4px 0 6px;">{r["nb_alertes"]} alerte(s) · {r["villes"]}</div>'
                        f'{crit_badge}'
                        f'<div style="background:{bg_inner};border-radius:10px;padding:7px 14px;margin-top:9px;display:inline-block;">'
                        f'<span style="font-size:{"22px" if is_win else "18px"};font-weight:800;color:{clr};">{r["score"]} pts</span></div>'
                        f'</div>'
                        f'<div style="background:{bar_grad};height:{ph}px;border-radius:0 0 8px 8px;'
                        f'display:flex;align-items:center;justify-content:center;'
                        f'font-size:24px;font-weight:900;color:rgba(255,255,255,.65);">'
                        f'{["1","2","3"][idx]}</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("---")
            df_adm = pd.DataFrame(ranking).rename(columns={
                "admin":"Administrateur","score":"Score 🏅","nb_alertes":"Nb alertes",
                "nb_critiques":"⚠️ Critiques","villes":"Communes"
            })
            df_adm.index = range(1, len(df_adm)+1)
            st.dataframe(df_adm, use_container_width=True)

        with sub_villes:
            villes_sc = {}
            for adm, data in st.session_state.alertes_territoire.items():
                for al in data["alertes"]:
                    v = al["ville"]
                    if v not in villes_sc:
                        villes_sc[v] = {"score":0,"nb":0,"admins":set(),"critiques":0,"types":set()}
                    pts = {"Critique":40,"Élevé":25,"Modéré":15,"Faible":5}.get(al.get("urgence","Faible"),5)
                    villes_sc[v]["score"]    += pts + min(20,len(al.get("description",""))//10) + 10
                    villes_sc[v]["nb"]       += 1
                    villes_sc[v]["admins"].add(adm)
                    villes_sc[v]["types"].add(al["type"][:12])
                    if al.get("urgence")=="Critique": villes_sc[v]["critiques"]+=1
            for v in villes_sc:
                villes_sc[v]["score"] += len(villes_sc[v]["types"]) * 10
                villes_sc[v]["admins"] = len(villes_sc[v]["admins"])

            top5_v = sorted(villes_sc.items(), key=lambda x: x[1]["score"], reverse=True)[:5]
            if not top5_v:
                st.info("Aucune ville active pour l'instant.")
                return

            max_sv = top5_v[0][1]["score"]
            for rank, (ville, data) in enumerate(top5_v, 1):
                pct = int(data["score"]/max_sv*100)
                icons = {1:"🥇",2:"🥈",3:"🥉",4:"4️⃣",5:"5️⃣"}
                ico = icons.get(rank,"•")
                crit_b = (f'<span style="background:#FEE2E2;color:#DC2626;padding:2px 8px;'
                          f'border-radius:20px;font-size:10px;font-weight:700;">'
                          f'🔴 {data["critiques"]}</span>') if data["critiques"] else ""
                st.markdown(
                    f'<div class="iceberg-card" style="margin-bottom:8px;">'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
                    f'<div style="display:flex;align-items:center;gap:9px;">'
                    f'<span style="font-size:22px;">{ico}</span>'
                    f'<div><div style="font-size:14px;font-weight:700;color:#0A0F1E;">{ville}</div>'
                    f'<div style="font-size:10px;color:#8895AA;">{", ".join(list(data["types"])[:2])}</div></div>'
                    f'</div>'
                    f'<div style="font-size:19px;font-weight:800;color:#1A56DB;">{data["score"]} pts</div>'
                    f'</div>'
                    f'<div style="display:flex;gap:7px;flex-wrap:wrap;margin-bottom:8px;">'
                    f'<span style="background:#EBF1FF;color:#1A56DB;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;">📋 {data["nb"]}</span>'
                    f'<span style="background:#D1FAE5;color:#059669;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;">👤 {data["admins"]}</span>'
                    f'{crit_b}</div>'
                    f'<div style="background:#F0F4F8;border-radius:5px;height:7px;overflow:hidden;">'
                    f'<div style="width:{pct}%;height:7px;border-radius:5px;background:linear-gradient(90deg,#0B1F5C,#3B82F6);"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )


# ══════════════════════════════════════════════════════════════════
# 10. PAGE CARTE ATTRACTIVITÉ — RADAR TERRITOIRE VIVANT
# ══════════════════════════════════════════════════════════════════

def _compute_secteurs(row: pd.Series) -> list:
    """
    Calcule un score d'adéquation pour chaque type de projet selon les données de la commune.
    Retourne une liste triée de (icone, label, score_pct).
    """
    cho   = float(row.get("taux_chomage",    10))
    pauv  = float(row.get("taux_pauvrete",   15))
    rev   = float(row.get("revenu_median",   22000))
    prix  = float(row.get("prix_m2_median",  3000))
    pop   = float(row.get("population",      5000))
    ent   = float(row.get("nb_entreprises_actives", 50))
    med   = float(row.get("score_desert_medical",   0.5))
    com   = float(row.get("score_desert_commercial",0.5))
    mob   = float(row.get("score_desert_mobilite",  0.5))
    gares = float(row.get("nb_gares", 0))
    den   = float(row.get("densite_hab_km2", 500))

    # Normalisation helpers
    def norm(v, lo, hi): return max(0.0, min(1.0, (v - lo) / max(hi - lo, 1)))

    rev_n   = norm(rev,   15000, 45000)   # revenu normalisé
    pop_n   = norm(pop,   500,   80000)   # population normalisée
    den_n   = norm(den,   50,    5000)    # densité normalisée
    prix_n  = norm(prix,  1500,  7000)    # prix m² normalisé (haut = cher)
    cho_n   = norm(cho,   3,     25)      # chômage normalisé (haut = mauvais)
    ent_n   = norm(ent,   10,    500)     # entreprises normalisées
    mob_n   = norm(gares, 0,     3)       # mobilité (gares)

    secteurs_raw = {
        ("🛒", "Commerce"):       0.35*pop_n + 0.25*(1-cho_n) + 0.20*rev_n  + 0.10*den_n   + 0.10*(1-com),
        ("🏥", "Médecin"):        0.45*med   + 0.25*pop_n     + 0.20*(1-cho_n)+ 0.10*rev_n,
        ("🎓", "Formation"):      0.30*pop_n + 0.25*cho_n     + 0.25*(1-rev_n)+ 0.20*mob_n,
        ("🏭", "Industrie"):      0.30*(1-den_n)+0.25*(1-prix_n)+0.25*mob_n  + 0.20*(1-cho_n),
        ("📦", "Logistique"):     0.35*mob_n + 0.30*(1-den_n) + 0.20*(1-prix_n)+0.15*ent_n,
        ("🍔", "Restauration"):   0.35*den_n + 0.25*rev_n     + 0.25*pop_n   + 0.15*(1-com),
        ("💊", "Pharmacie"):      0.40*med   + 0.30*pop_n     + 0.20*rev_n   + 0.10*den_n,
        ("🏗️",  "Immobilier"):    0.35*(1-prix_n)+0.30*pop_n  + 0.25*rev_n   + 0.10*mob_n,
    }
    result = [(ico, lbl, round(v * 100)) for (ico, lbl), v in secteurs_raw.items()]
    result.sort(key=lambda x: x[2], reverse=True)
    return result


def _compute_freins(row: pd.Series) -> list:
    """
    Identifie les freins principaux d'un territoire (pour zones Possible / Non Recommandé).
    Retourne une liste de (icone, description, sévérité) triée par sévérité décroissante.
    """
    freins = []
    cho   = float(row.get("taux_chomage",    10))
    pauv  = float(row.get("taux_pauvrete",   15))
    rev   = float(row.get("revenu_median",   22000))
    prix  = float(row.get("prix_m2_median",  3000))
    pop   = float(row.get("population",      5000))
    med   = float(row.get("score_desert_medical",    0.5))
    com   = float(row.get("score_desert_commercial", 0.5))
    mob   = float(row.get("score_desert_mobilite",   0.5))
    sig   = float(row.get("score_signal_faible",     0.3))
    risk  = float(row.get("risk_score",              0.5))

    if cho > 15:
        freins.append(("💼", f"Chômage élevé ({cho:.1f}%)", min(1.0, cho/25)))
    elif cho > 10:
        freins.append(("💼", f"Chômage modéré ({cho:.1f}%)", cho/25))

    if pauv > 25:
        freins.append(("📉", f"Taux de pauvreté critique ({pauv:.1f}%)", min(1.0, pauv/40)))
    elif pauv > 18:
        freins.append(("📉", f"Pauvreté élevée ({pauv:.1f}%)", pauv/40))

    if rev < 18000:
        freins.append(("💶", f"Revenu médian faible ({int(rev):,} €)".replace(",", " "), 1 - rev/25000))
    elif rev < 22000:
        freins.append(("💶", f"Revenu médian modeste ({int(rev):,} €)".replace(",", " "), 1 - rev/25000))

    if med > 0.65:
        freins.append(("🏥", "Désert médical — accès aux soins difficile", med))
    elif med > 0.45:
        freins.append(("🏥", "Couverture médicale insuffisante", med))

    if com > 0.65:
        freins.append(("🏪", "Désert commercial — peu de commerces", com))

    if mob > 0.60:
        freins.append(("🚌", "Mobilité limitée — transports insuffisants", mob))

    if pop < 1000:
        freins.append(("👥", f"Population très faible ({int(pop):,} hab.)".replace(",", " "), 0.7))

    if prix > 5500:
        freins.append(("🏠", f"Foncier très cher ({int(prix):,} €/m²)".replace(",", " "), min(1.0, prix/7000)))

    if sig > 0.65:
        freins.append(("⚡", "Signal faible détecté — instabilité territoriale", sig))

    if risk > 0.70:
        freins.append(("⚠️", "Indice de fragilité élevé — territoire fragilisé", risk))

    freins.sort(key=lambda x: x[2], reverse=True)
    return freins[:5]   # max 5 freins affichés


def _build_popup_html(row: pd.Series, df_full: pd.DataFrame = None) -> str:
    """Génère le HTML du popup riche : jauges ML + secteurs/freins + sparklines 2026-2030."""
    score_pct = float(row.get("score_custom", 0)) * 100
    opp_pct   = float(row.get("opportunity_score", 0)) * 100
    pred_pct  = float(row.get("pred_attractivite_2026", row.get("score_attractivite", 0))) * 100
    risk_pct  = float(row.get("risk_score", 0.5)) * 100
    cluster   = str(row.get("ml_cluster", "N/A"))
    cat       = str(row.get("cat_custom", ""))
    ville     = str(row.get("ville", ""))
    pop_fmt   = str(row.get("pop_fmt", "N/A"))
    cho_fmt   = str(row.get("cho_fmt", "N/A"))
    rev_fmt   = str(row.get("rev_fmt", "N/A"))
    prix_m2   = safe_val(row.get("prix_m2_median", 0))

    zone_colors = {
        "Zone Prioritaire": ("#16A34A", "#D1FAE5", "🟢"),
        "Zone Favorable":   ("#1A56DB", "#EBF1FF", "🔵"),
        "Zone Possible":    ("#D97706", "#FEF3C7", "🟡"),
        "Non Recommandé":   ("#64748B", "#F1F5F9", "⚪"),
    }
    fg, bg, dot = zone_colors.get(cat, ("#64748B", "#F1F5F9", "⚪"))
    risk_color  = "#DC2626" if risk_pct > 65 else ("#D97706" if risk_pct > 40 else "#059669")

    def bar(val, color, label):
        return (
            f"<div style='margin:4px 0;'>"
            f"<div style='display:flex;justify-content:space-between;font-size:10px;"
            f"color:#555;margin-bottom:2px;'>"
            f"<span>{label}</span>"
            f"<span style='font-weight:700;color:{color};'>{val:.0f}%</span></div>"
            f"<div style='background:#E2E8F2;border-radius:4px;height:5px;overflow:hidden;'>"
            f"<div style='width:{val:.0f}%;height:5px;background:{color};border-radius:4px;'>"
            f"</div></div></div>"
        )

    # ── Section dynamique selon la zone ───────────────────────────
    is_positive = cat in ("Zone Prioritaire", "Zone Favorable")

    if is_positive:
        secteurs = _compute_secteurs(row)
        top3     = secteurs[:3]
        sec_title = "✅ Secteurs recommandés"
        sec_bg    = "#F0FDF4"
        sec_border= "#BBF7D0"
        sec_title_color = "#166534"
        rows_html = ""
        for ico, lbl, sc in top3:
            bar_color = "#16A34A" if sc >= 70 else ("#1A56DB" if sc >= 50 else "#D97706")
            rows_html += (
                f"<div style='margin:5px 0;'>"
                f"<div style='display:flex;justify-content:space-between;font-size:11px;"
                f"margin-bottom:2px;'>"
                f"<span style='font-weight:600;color:#1E293B;'>{ico} {lbl}</span>"
                f"<span style='font-weight:800;color:{bar_color};'>{sc}%</span></div>"
                f"<div style='background:#D1FAE5;border-radius:3px;height:5px;overflow:hidden;'>"
                f"<div style='width:{sc}%;height:5px;background:{bar_color};border-radius:3px;'>"
                f"</div></div></div>"
            )
        # Lien "Voir tous les secteurs"
        all_sec_txt = " · ".join(f"{ico}{lbl} {sc}%" for ico, lbl, sc in secteurs[3:])
        sec_extra = (
            f"<div style='font-size:9px;color:#64748B;margin-top:4px;border-top:1px solid #BBF7D0;"
            f"padding-top:4px;'>{all_sec_txt}</div>"
        ) if secteurs[3:] else ""
        dynamic_block = (
            f"<div style='background:{sec_bg};border:1px solid {sec_border};"
            f"border-radius:8px;padding:8px 10px;margin-top:8px;'>"
            f"<div style='font-size:10px;font-weight:800;color:{sec_title_color};"
            f"text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px;'>{sec_title}</div>"
            f"{rows_html}{sec_extra}"
            f"</div>"
        )
    else:
        freins = _compute_freins(row)
        fr_title = "🚧 Freins identifiés"
        fr_rows  = ""
        for ico, desc, sev in freins:
            sev_color = "#DC2626" if sev > 0.65 else ("#D97706" if sev > 0.40 else "#64748B")
            sev_bg    = "#FEE2E2" if sev > 0.65 else ("#FEF3C7" if sev > 0.40 else "#F1F5F9")
            sev_lbl   = "Critique" if sev > 0.65 else ("Élevé" if sev > 0.40 else "Modéré")
            fr_rows += (
                f"<div style='display:flex;align-items:flex-start;gap:6px;"
                f"padding:4px 0;border-bottom:1px solid #FEE2E2;'>"
                f"<span style='font-size:12px;flex-shrink:0;'>{ico}</span>"
                f"<div style='flex:1;'>"
                f"<span style='font-size:10px;color:#1E293B;line-height:1.3;'>{desc}</span>"
                f"</div>"
                f"<span style='background:{sev_bg};color:{sev_color};padding:1px 5px;"
                f"border-radius:10px;font-size:9px;font-weight:700;white-space:nowrap;'>{sev_lbl}</span>"
                f"</div>"
            )
        if not freins:
            fr_rows = "<div style='font-size:10px;color:#64748B;'>Données insuffisantes pour analyse.</div>"

        # Conseil d'action
        conseil = ""
        if cat == "Zone Possible":
            conseil = (
                f"<div style='margin-top:6px;background:#FEF3C7;border-radius:6px;"
                f"padding:5px 8px;font-size:10px;color:#92400E;'>"
                f"💡 <b>Opportunité conditionnelle</b> — investissement possible si les freins "
                f"identifiés sont adressés en amont (accompagnement, subventions, partenariats)."
                f"</div>"
            )
        else:
            conseil = (
                f"<div style='margin-top:6px;background:#FEE2E2;border-radius:6px;"
                f"padding:5px 8px;font-size:10px;color:#991B1B;'>"
                f"⛔ <b>Zone à éviter</b> — cumuler plusieurs freins critiques rend "
                f"l'investissement à haut risque sans plan de redressement territorial."
                f"</div>"
            )

        dynamic_block = (
            f"<div style='background:#FFF5F5;border:1px solid #FCA5A5;"
            f"border-radius:8px;padding:8px 10px;margin-top:8px;'>"
            f"<div style='font-size:10px;font-weight:800;color:#991B1B;"
            f"text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px;'>{fr_title}</div>"
            f"{fr_rows}{conseil}"
            f"</div>"
        )

    # ── Sparklines prédictions 2026-2030 ─────────────────────────
    sparkline_block = ""
    if df_full is not None:
        try:
            tl = generate_timeline_data(df_full, ville)
            tl_pred = tl[tl["annee"] >= 2025]   # on affiche 2025→2030

            svg_attr = _build_sparkline_svg(tl_pred, "attractivite", "#1A56DB")
            svg_opp  = _build_sparkline_svg(tl_pred, "opportunite",  "#059669")
            svg_cho  = _build_sparkline_svg(tl_pred, "chomage",      "#DC2626")
            svg_rev  = _build_sparkline_svg(tl_pred, "revenu",       "#7C3AED")
            svg_prix = _build_sparkline_svg(tl_pred, "prix_m2",      "#D97706")
            svg_risk = _build_sparkline_svg(tl_pred, "risque",       "#F59E0B")

            # Évolution zone 2026→2030
            zone_evol = " → ".join(
                f"<span style='font-weight:700;color:{"#16A34A" if c=="Zone Prioritaire" else "#1A56DB" if c=="Zone Favorable" else "#D97706" if c=="Zone Possible" else "#64748B"};'>"
                f"{y}</span>"
                for y, c in zip(tl_pred["annee"].tolist()[::2], tl_pred["cat_zone"].tolist()[::2])
            )

            sparkline_block = (
                f"<div style='border-top:1px solid #E2E8F2;margin-top:8px;padding-top:8px;'>"
                f"<div style='font-size:9px;font-weight:800;color:#8895AA;text-transform:uppercase;"
                f"letter-spacing:.7px;margin-bottom:6px;'>📈 Trajectoire 2025 → 2030</div>"
                # Zone timeline
                f"<div style='font-size:9px;color:#64748B;margin-bottom:7px;line-height:1.6;'>"
                f"Zone prévue : {zone_evol}</div>"
                # Grille sparklines 2×3
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;'>"
                # Attractivité
                f"<div><div style='font-size:9px;color:#8895AA;font-weight:600;margin-bottom:2px;'>🎯 Attractivité</div>{svg_attr}</div>"
                # Opportunité ML
                f"<div><div style='font-size:9px;color:#8895AA;font-weight:600;margin-bottom:2px;'>💡 Opportunité</div>{svg_opp}</div>"
                # Chômage
                f"<div><div style='font-size:9px;color:#8895AA;font-weight:600;margin-bottom:2px;'>💼 Chômage %</div>{svg_cho}</div>"
                # Revenu
                f"<div><div style='font-size:9px;color:#8895AA;font-weight:600;margin-bottom:2px;'>💶 Revenu méd.</div>{svg_rev}</div>"
                # Prix m²
                f"<div><div style='font-size:9px;color:#8895AA;font-weight:600;margin-bottom:2px;'>🏠 Prix m²</div>{svg_prix}</div>"
                # Risque
                f"<div><div style='font-size:9px;color:#8895AA;font-weight:600;margin-bottom:2px;'>⚠️ Indice de fragilité</div>{svg_risk}</div>"
                f"</div>"
                f"<div style='font-size:8px;color:#CBD5E1;margin-top:5px;text-align:right;'>"
                f"— — Prévision IA &nbsp; — Historique</div>"
                f"</div>"
            )
        except Exception:
            sparkline_block = ""

    return (
        f"<div style='font-family:Plus Jakarta Sans,sans-serif;min-width:250px;max-width:290px;padding:2px;'>"
        # ── En-tête dégradé ───────────────────────────────────────
        f"<div style='background:linear-gradient(135deg,#0B1F5C,#1A56DB);"
        f"border-radius:10px 10px 0 0;padding:10px 13px;margin:-4px -4px 10px;'>"
        f"<div style='font-size:15px;font-weight:800;color:#fff;letter-spacing:-.3px;'>{ville}</div>"
        f"<div style='margin-top:4px;display:flex;align-items:center;gap:6px;'>"
        f"<span style='background:{bg};color:{fg};padding:2px 8px;border-radius:20px;"
        f"font-size:10px;font-weight:700;'>{dot} {cat}</span>"
        f"<span style='font-size:13px;font-weight:800;color:#fff;'>{score_pct:.0f}%</span>"
        f"</div></div>"
        # ── Cluster ML ────────────────────────────────────────────
        f"<div style='background:#EDE9FE;border-radius:6px;padding:5px 9px;margin-bottom:7px;"
        f"display:flex;align-items:center;gap:6px;'>"
        f"<span style='font-size:11px;'>🤖</span>"
        f"<span style='font-size:11px;font-weight:700;color:#5B21B6;'>Profil ML : {cluster}</span>"
        f"</div>"
        # ── Jauges ML ─────────────────────────────────────────────
        f"<div style='margin-bottom:7px;'>"
        + bar(opp_pct,  "#1A56DB", "🎯 Opportunité ML")
        + bar(pred_pct, "#059669", "🔮 Prédiction 2026")
        + bar(risk_pct, risk_color, "⚠️ Indice de fragilité")
        + f"</div>"
        # ── Section dynamique (secteurs OU freins) ────────────────
        + dynamic_block
        # ── Données territoire ────────────────────────────────────
        + f"<div style='border-top:1px solid #E2E8F2;margin-top:8px;padding-top:7px;"
        f"display:grid;grid-template-columns:1fr 1fr;gap:4px;'>"
        f"<div style='font-size:10px;color:#8895AA;'>👥 Population<br>"
        f"<span style='font-size:12px;font-weight:700;color:#0A0F1E;'>{pop_fmt}</span></div>"
        f"<div style='font-size:10px;color:#8895AA;'>💼 Chômage<br>"
        f"<span style='font-size:12px;font-weight:700;color:#DC2626;'>{cho_fmt}%</span></div>"
        f"<div style='font-size:10px;color:#8895AA;'>💶 Revenu méd.<br>"
        f"<span style='font-size:12px;font-weight:700;color:#0A0F1E;'>{rev_fmt} €</span></div>"
        f"<div style='font-size:10px;color:#8895AA;'>🏠 Prix m²<br>"
        f"<span style='font-size:12px;font-weight:700;color:#0A0F1E;'>{prix_m2} €</span></div>"
        f"</div>"
        # ── Sparklines 2025-2030 ──────────────────────────────────
        + sparkline_block
        + f"</div>"
    )


def _build_sparkline_svg(df_timeline: pd.DataFrame, col: str,
                         color: str, width: int = 220, height: int = 40) -> str:
    """Génère un mini sparkline SVG pour une colonne de timeline."""
    vals = df_timeline[col].tolist()
    years = df_timeline["annee"].tolist()
    types = df_timeline["type"].tolist()
    if not vals or max(vals) == min(vals):
        return ""
    vmin, vmax = min(vals), max(vals)
    pad = 6
    W, H = width - 2*pad, height - 2*pad

    def sx(i): return pad + i * W / max(len(vals)-1, 1)
    def sy(v): return pad + H - (v - vmin) / (vmax - vmin) * H

    # Ligne historique
    hist_pts = [(sx(i), sy(v)) for i, (v, t) in enumerate(zip(vals, types)) if t == "Historique"]
    pred_pts = [(sx(i), sy(v)) for i, (v, t) in enumerate(zip(vals, types)) if t == "Prévision IA"]
    # Point de jonction
    join_idx = next((i for i, t in enumerate(types) if t == "Prévision IA"), len(vals)-1)
    if join_idx > 0:
        pred_pts = [(sx(join_idx-1), sy(vals[join_idx-1]))] + pred_pts

    def pts_to_path(pts):
        if len(pts) < 2: return ""
        d = f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"
        for x, y in pts[1:]:
            d += f" L {x:.1f} {y:.1f}"
        return d

    hist_path = pts_to_path(hist_pts)
    pred_path = pts_to_path(pred_pts)

    # Dernière valeur
    last_val = vals[-1]
    last_lbl = f"{last_val*100:.0f}%" if vmax <= 1 else f"{int(last_val):,}".replace(",", " ")

    svg = (
        f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>"
        f"<rect width='{width}' height='{height}' rx='4' fill='#F8FAFC'/>"
    )
    if hist_path:
        svg += f"<path d='{hist_path}' fill='none' stroke='{color}' stroke-width='1.8' stroke-linecap='round'/>"
    if pred_path:
        svg += (f"<path d='{pred_path}' fill='none' stroke='{color}' stroke-width='1.8' "
                f"stroke-dasharray='4 3' stroke-linecap='round' opacity='0.7'/>")
    # Point final
    if vals:
        lx, ly = sx(len(vals)-1), sy(vals[-1])
        svg += f"<circle cx='{lx:.1f}' cy='{ly:.1f}' r='3' fill='{color}'/>"
        svg += (f"<text x='{lx-2:.1f}' y='{ly-6:.1f}' font-size='8' fill='{color}' "
                f"font-weight='700' text-anchor='end'>{last_lbl}</text>")
    svg += "</svg>"
    return svg


def _build_folium_map(df_m: pd.DataFrame, mode_carte: str, show_zones: list,
                      df_full: pd.DataFrame = None) -> "folium.Map":
    """Construit la carte Folium satellite avec rendu innovant selon le mode."""
    from folium.plugins import HeatMap, MarkerCluster
    import branca.element as be

    lat_c = df_m["latitude"].mean()
    lon_c = df_m["longitude"].mean()

    ZONE_HEX = {
        "Zone Prioritaire": "#16A34A",
        "Zone Favorable":   "#1A56DB",
        "Zone Possible":    "#F59E0B",
        "Non Recommandé":   "#94A3B8",
    }
    PULSE_COLOR = {
        "Zone Prioritaire": "0,200,80",
        "Zone Favorable":   "26,86,219",
        "Zone Possible":    "245,158,11",
        "Non Recommandé":   "148,163,184",
    }

    m = folium.Map(
        location=[lat_c, lon_c],
        zoom_start=11,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        prefer_canvas=True,
        zoom_control=True,
    )

    # ── CSS animations pulsation injecté dans la carte ────────────
    pulse_css = be.Element("""
    <style>
    @keyframes pulse-green  { 0%,100%{box-shadow:0 0 0 0 rgba(0,200,80,.6)}  50%{box-shadow:0 0 0 12px rgba(0,200,80,0)} }
    @keyframes pulse-blue   { 0%,100%{box-shadow:0 0 0 0 rgba(26,86,219,.6)} 50%{box-shadow:0 0 0 12px rgba(26,86,219,0)} }
    @keyframes pulse-amber  { 0%,100%{box-shadow:0 0 0 0 rgba(245,158,11,.6)}50%{box-shadow:0 0 0 10px rgba(245,158,11,0)} }
    .pulse-green  { animation: pulse-green  2.0s infinite; }
    .pulse-blue   { animation: pulse-blue   2.4s infinite; }
    .pulse-amber  { animation: pulse-amber  2.8s infinite; }
    .leaflet-popup-content { margin: 4px 8px !important; }
    .leaflet-popup-content-wrapper { border-radius: 12px !important; padding: 2px !important;
        box-shadow: 0 8px 32px rgba(10,15,30,.20) !important; border: none !important; }
    </style>""")
    m.get_root().html.add_child(pulse_css)

    df_vis = df_m[df_m["cat_custom"].isin(show_zones)] if show_zones else df_m

    if mode_carte == "Opportunités 2027":
        # Points uniquement — toutes les communes visibles, taille proportionnelle à l'opportunité
        opp_col = "opportunity_score" if "opportunity_score" in df_vis.columns else "score_custom"
        for _, row in df_vis.iterrows():
            if pd.isna(row["latitude"]) or pd.isna(row["longitude"]): continue
            cat     = row["cat_custom"]
            hex_col = ZONE_HEX.get(cat, "#94A3B8")
            opp_pct = int(float(row.get(opp_col, row["score_custom"]))*100)
            dot_sz  = max(8, int(opp_pct * 0.18 + 5))
            ville_court = str(row['ville'])[:14] + ("…" if len(str(row['ville'])) > 14 else "")
            icon_html = (
                f"<div style='display:flex;flex-direction:column;align-items:center;gap:2px;'>"
                f"<div style='width:{dot_sz}px;height:{dot_sz}px;border-radius:50%;"
                f"background:{hex_col};border:2px solid #fff;opacity:0.88;cursor:pointer;"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:{max(6,dot_sz//3)}px;font-weight:800;color:#fff;'>"
                f"{opp_pct if dot_sz > 12 else ''}</div>"
                f"<div class='ville-label' style='background:rgba(10,15,30,.75);color:#fff;padding:1px 5px;"
                f"border-radius:4px;font-size:9px;font-weight:700;white-space:nowrap;"
                f"font-family:sans-serif;line-height:1.3;'>{ville_court}</div>"
                f"</div>"
            )
            icon_w = max(dot_sz, len(ville_court)*7)
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=folium.DivIcon(html=icon_html,
                    icon_size=(icon_w, dot_sz+18), icon_anchor=(dot_sz//2, dot_sz//2)),
                popup=folium.Popup(_build_popup_html(row, df_full=df_full), max_width=300, show=False),
                tooltip=folium.Tooltip(
                    f"<b>{row['ville']}</b> — Opportunité 2027 : {opp_pct}%",
                    style="font-family:sans-serif;font-size:12px;font-weight:600;"
                ),
            ).add_to(m)

    elif mode_carte == "Radar Opportunités":
        # Cercles pulsants + rayon d'influence = innovation visuelle clé
        for _, row in df_vis.iterrows():
            cat     = row["cat_custom"]
            hex_col = ZONE_HEX.get(cat, "#94A3B8")
            rgb     = PULSE_COLOR.get(cat, "148,163,184")
            score   = float(row["score_custom"])
            opp     = float(row.get("opportunity_score", score))
            radius_influence = int(opp * 1800 + 300)   # rayon zone d'influence en mètres

            # Cercle d'influence semi-transparent
            if cat in ("Zone Prioritaire", "Zone Favorable"):
                folium.Circle(
                    location=[row["latitude"], row["longitude"]],
                    radius=radius_influence,
                    color=hex_col, fill=True, fill_color=hex_col,
                    fill_opacity=0.07, weight=1, opacity=0.35,
                    dash_array="6 4",
                ).add_to(m)

            # Marqueur pulsant central
            pulse_cls = {
                "Zone Prioritaire": "pulse-green",
                "Zone Favorable":   "pulse-blue",
                "Zone Possible":    "pulse-amber",
            }.get(cat, "")
            dot_size = max(8, int(score * 18 + 6))
            ville_court = str(row['ville'])[:14] + ("…" if len(str(row['ville'])) > 14 else "")
            icon_html = (
                f"<div style='display:flex;flex-direction:column;align-items:center;gap:2px;'>"
                f"<div class='{pulse_cls}' style='width:{dot_size}px;height:{dot_size}px;"
                f"border-radius:50%;background:{hex_col};border:2px solid #fff;"
                f"cursor:pointer;position:relative;display:flex;align-items:center;justify-content:center;'>"
                f"<span style='font-size:{max(7,dot_size//3)}px;font-weight:800;color:#fff;'>"
                f"{row['score_pct']:.0f}</span></div>"
                f"<div class='ville-label' style='background:rgba(10,15,30,.75);color:#fff;padding:1px 5px;"
                f"border-radius:4px;font-size:9px;font-weight:700;white-space:nowrap;"
                f"font-family:sans-serif;line-height:1.3;'>{ville_court}</div>"
                f"</div>"
            )
            icon_w = max(80, len(ville_court) * 7)
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=folium.DivIcon(
                    html=icon_html,
                    icon_size=(icon_w, dot_size + 18),
                    icon_anchor=(dot_size//2, dot_size//2),
                ),
                popup=folium.Popup(_build_popup_html(row, df_full=df_full), max_width=300, show=False),
                tooltip=folium.Tooltip(
                    f"<b>{row['ville']}</b><br>Opportunité ML : {opp*100:.0f}%",
                    style="font-family:sans-serif;font-size:12px;"
                ),
            ).add_to(m)

    else:  # Clusters ML
        cluster_colors = {
            "Territoire dynamique": "#16A34A",
            "Zone de vigilance":    "#DC2626",
            "Désert de services":   "#9333EA",
            "Potentiel émergent":   "#F59E0B",
            "Territoire stable":    "#0891B2",
        }
        for _, row in df_vis.iterrows():
            cl      = str(row.get("ml_cluster", "N/A"))
            hex_col = cluster_colors.get(cl, "#64748B")
            score   = float(row["score_custom"])
            dot_size = max(9, int(score * 16 + 7))
            ville_court = str(row['ville'])[:14] + ("…" if len(str(row['ville'])) > 14 else "")
            icon_html = (
                f"<div style='display:flex;flex-direction:column;align-items:center;gap:2px;'>"
                f"<div style='width:{dot_size}px;height:{dot_size}px;border-radius:50%;"
                f"background:{hex_col};border:2px solid rgba(255,255,255,0.8);"
                f"opacity:0.92;cursor:pointer;'></div>"
                f"<div class='ville-label' style='background:rgba(10,15,30,.75);color:#fff;padding:1px 5px;"
                f"border-radius:4px;font-size:9px;font-weight:700;white-space:nowrap;"
                f"font-family:sans-serif;line-height:1.3;'>{ville_court}</div>"
                f"</div>"
            )
            icon_w = max(80, len(ville_court) * 7)
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=folium.DivIcon(
                    html=icon_html,
                    icon_size=(icon_w, dot_size + 18),
                    icon_anchor=(dot_size//2, dot_size//2),
                ),
                popup=folium.Popup(_build_popup_html(row, df_full=df_full), max_width=300, show=False),
                tooltip=folium.Tooltip(
                    f"<b>{row['ville']}</b><br>🤖 {cl}",
                    style="font-family:sans-serif;font-size:12px;"
                ),
            ).add_to(m)

    # ── CSS : labels visibles seulement au zoom ≥ 13 ─────────────
    import branca.element as be
    zoom_css = be.Element("""
    <style>
    /* Labels masqués par défaut — visibles uniquement quand la carte est zoomée */
    .ville-label { display: none !important; }
    /* Leaflet expose le niveau de zoom via data-zoom sur la carte */
    </style>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
      function applyZoomLabels(map) {
        map.on('zoomend', function() {
          var z = map.getZoom();
          var labels = document.querySelectorAll('.ville-label');
          labels.forEach(function(el) {
            el.style.display = z >= 13 ? 'block' : 'none';
          });
        });
      }
      // Attendre que Leaflet soit prêt
      var interval = setInterval(function() {
        if (window.L) {
          clearInterval(interval);
          document.querySelectorAll('.leaflet-container').forEach(function(container) {
            var mapObj = container._leaflet_map || (container.__proto__ && container._leaflet_id && L.map(container));
            if (container._leaflet_map) applyZoomLabels(container._leaflet_map);
          });
        }
      }, 300);
    });
    </script>""")
    m.get_root().html.add_child(zoom_css)
    if mode_carte == "Clusters ML":
        legend_items = [
            ("#16A34A","Territoire dynamique"), ("#DC2626","Zone de vigilance"),
            ("#9333EA","Désert de services"),   ("#F59E0B","Potentiel émergent"),
            ("#0891B2","Territoire stable"),
        ]
    else:
        legend_items = [
            ("#16A34A","Zone Prioritaire ≥70%"), ("#1A56DB","Zone Favorable ≥50%"),
            ("#F59E0B","Zone Possible ≥30%"),    ("#94A3B8","Non recommandé"),
        ]

    legend_rows = "".join(
        f"<div style='display:flex;align-items:center;gap:6px;margin:3px 0;'>"
        f"<div style='width:10px;height:10px;border-radius:50%;background:{c};flex-shrink:0;'></div>"
        f"<span style='font-size:10px;color:#1E293B;'>{lbl}</span></div>"
        for c, lbl in legend_items
    )
    legend_html = f"""
    <div style='position:fixed;bottom:28px;left:12px;z-index:9999;
         background:rgba(255,255,255,0.94);border-radius:10px;
         padding:10px 13px;box-shadow:0 4px 16px rgba(10,15,30,.18);
         border:1px solid #E2E8F2;backdrop-filter:blur(6px);min-width:170px;'>
      <div style='font-size:9px;font-weight:800;color:#8895AA;text-transform:uppercase;
           letter-spacing:.8px;margin-bottom:6px;'>{"Profil ML" if mode_carte=="Clusters ML" else "Zone d'opportunité"}</div>
      {legend_rows}
    </div>"""
    m.get_root().html.add_child(be.Element(legend_html))

    # ── Badge ICEBERG flottant ─────────────────────────────────────
    badge_html = """
    <div style='position:fixed;top:10px;right:10px;z-index:9999;
         background:linear-gradient(135deg,#0B1F5C,#1A56DB);
         border-radius:10px;padding:7px 12px;
         box-shadow:0 4px 14px rgba(26,86,219,.40);'>
      <span style='font-size:11px;font-weight:800;color:#fff;letter-spacing:.3px;'>
        🧊 ICEBERG · Radar IA</span>
    </div>"""
    m.get_root().html.add_child(be.Element(badge_html))

    return m


def page_attractivite(df: pd.DataFrame):
    # Breadcrumb
    st.markdown("""
    <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#8895AA;
         margin-bottom:14px;font-weight:600;">
      <span>Île-de-France</span><span>›</span><span>91 &amp; 94</span>
      <span>›</span><span style="color:#0A0F1E;">Radar Territoire Vivant</span>
    </div>""", unsafe_allow_html=True)

    page_header("🎯", "Radar Territoire Vivant",
                "Carte satellite IA · Opportunités ML prédictives · 241 communes · Dép. 91 & 94",
                badge="ML Prédictif")

    # ── Enrichissement ML ─────────────────────────────────────────
    df = build_ml_features(df)

    nb_alertes = len(df[df["cat_signal_faible"]=="Signal Fort"])
    n_prio_ml  = len(df[df["opportunity_score"] >= 0.70]) if "opportunity_score" in df.columns else 0

    # ── Bannière alertes ──────────────────────────────────────────
    # n_prio_sidebar = même calcul que le sidebar (cat_attractivite, score brut)
    n_prio_sidebar = len(df[df["cat_attractivite"] == "Zone Prioritaire"])
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-bottom:18px;">
      <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:12px;padding:11px 15px;">
        <div style="font-size:10px;font-weight:700;color:#92400E;text-transform:uppercase;letter-spacing:.6px;">⚡ Signaux forts</div>
        <div style="font-size:22px;font-weight:800;color:#D97706;">{nb_alertes}</div>
      </div>
      <div style="background:#D1FAE5;border:1px solid #6EE7B7;border-radius:12px;padding:11px 15px;">
        <div style="font-size:10px;font-weight:700;color:#065F46;text-transform:uppercase;letter-spacing:.6px;">🟢 Zones prioritaires</div>
        <div style="font-size:22px;font-weight:800;color:#059669;">{n_prio_sidebar}</div>
      </div>
      <div style="background:#EDE9FE;border:1px solid #C4B5FD;border-radius:12px;padding:11px 15px;cursor:help;"
           title="Le Gradient Boosting est un algorithme d'IA qui combine plusieurs modèles de prédiction en cascade. Il analyse 13 indicateurs (chômage, revenus, densité, déserts…) pour calculer un score d'attractivité et prédire l'évolution de chaque commune.">
        <div style="font-size:10px;font-weight:700;color:#4C1D95;text-transform:uppercase;letter-spacing:.6px;">🤖 Modèle IA</div>
        <div style="font-size:11px;font-weight:700;color:#6D28D9;line-height:1.4;">Gradient Boosting<br>
          <span style="font-size:9px;font-weight:500;color:#7C3AED;opacity:.8;">13 indicateurs · 241 communes</span>
        </div>
      </div>
      <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;padding:11px 15px;">
        <div style="font-size:10px;font-weight:700;color:#1E40AF;text-transform:uppercase;letter-spacing:.6px;">📅 Horizon</div>
        <div style="font-size:22px;font-weight:800;color:#1A56DB;">2030</div>
      </div>
    </div>""", unsafe_allow_html=True)

    col_map, col_panel = st.columns([3, 1])

    with col_map:
        # ── Contrôles carte ───────────────────────────────────────
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            type_projet = st.selectbox("", [
                "🛒 Commerce / Grande surface","🏥 Médecin / Cabinet médical",
                "🎓 Centre de formation","🏭 Industrie / Usine",
                "📦 Logistique / Entrepôt","🍔 Restauration","💊 Pharmacie","⚙️ Personnalisé"
            ], label_visibility="collapsed")
        with c2:
            dept_sel = st.selectbox("", ["Toute la zone","Essonne (91)","Val-de-Marne (94)"],
                                    label_visibility="collapsed", key="dept_attr")
        with c3:
            mode_carte = st.radio("", ["Radar Opportunités","Opportunités 2027","Clusters ML"],
                                  horizontal=True, label_visibility="collapsed")

        presets = {
            "🛒 Commerce / Grande surface":(60,40),"🏥 Médecin / Cabinet médical":(25,75),
            "🎓 Centre de formation":(30,70),"🏭 Industrie / Usine":(70,30),
            "📦 Logistique / Entrepôt":(65,35),"🍔 Restauration":(40,60),
            "💊 Pharmacie":(30,70),"⚙️ Personnalisé":(50,50)
        }
        pf, pe = presets.get(type_projet, (50,50))
        if type_projet == "⚙️ Personnalisé":
            pf = st.slider("Poids foncier (%)", 0, 100, 50, 10)
            pe = 100 - pf

        # ── Filtre zones (boutons inline) ─────────────────────────
        all_zones = ["Zone Prioritaire","Zone Favorable","Zone Possible","Non Recommandé"]
        zone_key  = f"zones_filter_{dept_sel}_{type_projet}"
        if zone_key not in st.session_state:
            st.session_state[zone_key] = all_zones.copy()

        st.markdown("<div style='font-size:10px;font-weight:700;color:#8895AA;text-transform:uppercase;"
                    "letter-spacing:.7px;margin:8px 0 4px;'>Filtrer les zones affichées</div>",
                    unsafe_allow_html=True)
        zc1, zc2, zc3, zc4 = st.columns(4)
        zone_btns = {
            "Zone Prioritaire": (zc1, "🟢 Prioritaire", "#16A34A"),
            "Zone Favorable":   (zc2, "🔵 Favorable",   "#1A56DB"),
            "Zone Possible":    (zc3, "🟡 Possible",    "#D97706"),
            "Non Recommandé":   (zc4, "⚪ Non reco.",   "#64748B"),
        }
        for zone, (col, lbl, _) in zone_btns.items():
            with col:
                active = zone in st.session_state[zone_key]
                if st.button(lbl, key=f"btn_{zone}_{zone_key}",
                             use_container_width=True,
                             type="primary" if active else "secondary"):
                    if active:
                        if len(st.session_state[zone_key]) > 1:
                            st.session_state[zone_key].remove(zone)
                    else:
                        st.session_state[zone_key].append(zone)
                    st.rerun()

        show_zones = st.session_state[zone_key]

    # ── Slider temporel ───────────────────────────────────────────
    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:#8895AA;text-transform:uppercase;"
        "letter-spacing:.7px;margin:4px 0 2px;'>🕐 Horizon temporel de projection</div>",
        unsafe_allow_html=True
    )
    sc1, sc2 = st.columns([4, 1])
    with sc1:
        # Année courante = 2026
        if "slider_annee_carte" not in st.session_state:
            st.session_state["slider_annee_carte"] = 2026
        annee_sel = st.slider(
            "", min_value=2026, max_value=2030, step=1,
            label_visibility="collapsed", key="slider_annee_carte"
        )
    with sc2:
        delta_annee = annee_sel - 2026
        badge_color = "#059669" if delta_annee == 0 else ("#1A56DB" if delta_annee <= 2 else "#7C3AED")
        st.markdown(
            f"<div style='background:{badge_color};color:#fff;border-radius:10px;"
            f"padding:8px 0;text-align:center;font-size:15px;font-weight:800;margin-top:2px;'>"
            f"{'Actuel' if delta_annee==0 else f'+{delta_annee} ans'}<br>"
            f"<span style='font-size:20px;font-weight:900;'>{annee_sel}</span></div>",
            unsafe_allow_html=True
        )

    if annee_sel > 2026:
        st.markdown(
            f"<div style='background:linear-gradient(90deg,#EDE9FE,#F5F3FF);border:1px solid #C4B5FD;"
            f"border-radius:9px;padding:7px 14px;margin-bottom:8px;font-size:11px;color:#5B21B6;"
            f"display:flex;align-items:center;gap:7px;'>"
            f"<span style='font-size:14px;'>🔮</span>"
            f"<span><b>Projection IA {annee_sel}</b> — Zones recalculées par Gradient Boosting "
            f"selon les tendances ML de chaque profil. Popups : trajectoire 2025→2030.</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        df_proj = project_df_to_year(df, annee_sel)
    else:
        df_proj = df.copy()

    # ── Calcul scores sur df_proj (AVANT le split col_map/col_panel) ──
    df_proj["score_custom"] = (
        df_proj["score_reindustrialisation"]*(pf/100) +
        df_proj["score_employabilite"]*(pe/100)
    )
    mn, mx = df_proj["score_custom"].min(), df_proj["score_custom"].max()
    if mx > mn: df_proj["score_custom"] = (df_proj["score_custom"]-mn)/(mx-mn)
    df_proj["score_custom"] = df_proj["score_custom"].clip(0,1).fillna(0)
    df_proj["cat_custom"] = df_proj["score_custom"].apply(
        lambda s: "Zone Prioritaire" if s>=.70 else
                  ("Zone Favorable"  if s>=.50 else
                  ("Zone Possible"   if s>=.30 else "Non Recommandé"))
    )

    df_m = df_proj.dropna(subset=["latitude","longitude"]).copy()
    if "91" in dept_sel:  df_m = df_m[df_m["code_dept"].astype(str)=="91"]
    elif "94" in dept_sel: df_m = df_m[df_m["code_dept"].astype(str)=="94"]

    df_m["score_pct"] = (df_m["score_custom"]*100).round(1)
    df_m["pop_fmt"]   = df_m["population"].apply(safe_val)
    df_m["cho_fmt"]   = df_m["taux_chomage"].apply(lambda x: safe_val(x,1))
    df_m["rev_fmt"]   = df_m["revenu_median"].apply(safe_val)

    # ── Rendu carte + panneau ─────────────────────────────────────
    with col_map:
        if _FOLIUM_OK:
            folium_map = _build_folium_map(df_m, mode_carte, show_zones, df_full=df)
            st_folium(folium_map, width=None, height=560,
                      use_container_width=True, returned_objects=[])
        else:
            st.error("⚠️ Folium non installé : `pip install folium streamlit-folium`")

    # ── Panneau latéral ───────────────────────────────────────────
    with col_panel:
        moy_cho = df_proj["taux_chomage"].mean()
        # Même calcul que le sidebar : cat_attractivite sur score brut
        n_prio  = len(df[df["cat_attractivite"] == "Zone Prioritaire"])
        nb_cr   = int(df_proj["nb_entreprises_actives"].sum()*0.08)

        for lbl, val, delta, color in [
            ("Taux chômage",       f"{moy_cho:.1f}%",  "↗ +0,3 pts",   "#DC2626"),
            ("Zones prioritaires", str(n_prio),         "",              "#059669"),
            ("Créations (est.)",   safe_val(nb_cr),     "↗ +12 ce mois","#059669"),
        ]:
            st.markdown(metric_card(lbl, val, delta, color), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Top 5 opportunités ML
        top_opp = df.nlargest(5, "opportunity_score")[
            ["ville","opportunity_score","ml_cluster","pred_attractivite_2026"]
        ].to_dict("records")
        st.markdown(
            '<div style="background:#fff;border:1px solid #E2E8F2;border-radius:14px;'
            'overflow:hidden;margin-top:4px;">'
            '<div style="padding:11px 14px;border-bottom:1px solid #F0F4F8;display:flex;'
            'align-items:center;justify-content:space-between;">'
            '<span style="font-size:12px;font-weight:700;color:#0A0F1E;">🎯 Top Opportunités ML</span>'
            f'<span style="background:#EDE9FE;color:#6D28D9;padding:2px 8px;'
            f'border-radius:20px;font-size:10px;font-weight:700;">2026</span>'
            '</div>',
            unsafe_allow_html=True
        )
        for r in top_opp:
            opp_pct  = int(r["opportunity_score"]*100)
            pred_pct = int(r["pred_attractivite_2026"]*100)
            cl       = r["ml_cluster"]
            cl_color = {"Territoire dynamique":"#059669","Zone de vigilance":"#DC2626",
                        "Potentiel émergent":"#D97706","Territoire stable":"#0891B2",
                        "Désert de services":"#7C3AED"}.get(cl,"#64748B")
            st.markdown(
                f'<div style="padding:9px 12px;border-bottom:1px solid #F8FAFC;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">'
                f'<span style="font-size:12px;font-weight:700;color:#0A0F1E;">{r["ville"]}</span>'
                f'<span style="font-size:13px;font-weight:800;color:#1A56DB;">{opp_pct}%</span>'
                f'</div>'
                f'<div style="display:flex;gap:5px;flex-wrap:wrap;">'
                f'<span style="background:{cl_color}22;color:{cl_color};padding:1px 7px;'
                f'border-radius:20px;font-size:9px;font-weight:700;">🤖 {cl[:18]}</span>'
                f'<span style="background:#D1FAE5;color:#059669;padding:1px 7px;'
                f'border-radius:20px;font-size:9px;font-weight:700;">🔮 {pred_pct}% en 2026</span>'
                f'</div>'
                f'<div style="background:#F0F4F8;border-radius:4px;height:4px;margin-top:5px;overflow:hidden;">'
                f'<div style="width:{opp_pct}%;height:4px;background:linear-gradient(90deg,#0B1F5C,#3B82F6);'
                f'border-radius:4px;"></div></div>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Signaux faibles
        top_sig  = df_proj.nlargest(8,"score_signal_faible")[
            ["ville","cat_signal_faible","score_signal_faible","taux_chomage"]
        ].to_dict("records")
        nb_fort  = len(df_proj[df_proj["cat_signal_faible"]=="Signal Fort"])
        st.markdown(
            f'<div style="background:#fff;border:1px solid #E2E8F2;border-radius:14px;'
            f'overflow:hidden;margin-top:10px;">'
            f'<div style="padding:11px 14px;border-bottom:1px solid #F0F4F8;display:flex;'
            f'align-items:center;justify-content:space-between;">'
            f'<span style="font-size:12px;font-weight:700;color:#0A0F1E;">⚡ Signaux IA</span>'
            f'<span style="background:#FEE2E2;color:#DC2626;padding:2px 8px;'
            f'border-radius:20px;font-size:10px;font-weight:700;">{nb_fort} alertes</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        for s in top_sig:
            clr = "#DC2626" if s["cat_signal_faible"]=="Signal Fort" else "#D97706"
            pct = int(s["score_signal_faible"]*100)
            st.markdown(
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:8px 12px;border-bottom:1px solid #F8FAFC;">'
                f'<div style="display:flex;align-items:center;gap:7px;">'
                f'<div style="width:6px;height:6px;border-radius:50%;background:{clr};flex-shrink:0;"></div>'
                f'<span style="font-size:12px;font-weight:600;color:#0A0F1E;">{s["ville"]}</span></div>'
                f'<span style="background:{"#FEE2E2" if clr=="#DC2626" else "#FEF3C7"};'
                f'color:{clr};padding:1px 7px;border-radius:20px;font-size:10px;font-weight:700;">{pct}%</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tableau Top 10 ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🏆 Top 10 opportunités territoriales — Score ML combiné")
    top10 = df_proj.nlargest(10,"opportunity_score")[
        ["ville","dept_nom","score_custom","cat_custom","opportunity_score",
         "pred_attractivite_2026","risk_score","ml_cluster","taux_chomage","population"]
    ].copy()
    top10.columns = ["Commune","Département","Score","Zone","🎯 Opportunité ML",
                     "🔮 Pred.2026","⚠️ Risque","🤖 Profil ML","Chômage %","Population"]
    for col in ["Score","🎯 Opportunité ML","🔮 Pred.2026","⚠️ Risque"]:
        top10[col] = (top10[col]*100).round(1).astype(str) + "%"
    st.dataframe(top10.reset_index(drop=True), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Top 10 CSV", top10.to_csv(index=False, encoding="utf-8-sig"),
                           "top10_opportunites_ml.csv", "text/csv", use_container_width=True)
    with c2:
        all_s = df_proj[["ville","dept_nom","score_custom","cat_custom","opportunity_score","ml_cluster"]
                   ].sort_values("opportunity_score",ascending=False).copy()
        st.download_button("⬇️ Toutes les communes", all_s.to_csv(index=False,encoding="utf-8-sig"),
                           "toutes_communes_ml.csv", "text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 11. PAGE PRÉDICTIONS ML  (NOUVELLE)
# ══════════════════════════════════════════════════════════════════
def page_predictions_ml(df: pd.DataFrame):
    page_header("🔮", "Prédictions ML 2025–2030",
                "Gradient Boosting · K-Means · Timeline · 241 communes · Dép. 91 & 94",
                badge="scikit-learn" if _ML_OK else "Mode dégradé")

    if not _ML_OK:
        st.warning("⚠️ scikit-learn non installé — prédictions en mode simplifié.")

    df = build_ml_features(df)
    ville_names = sorted(df["ville"].dropna().unique().tolist())

    # ── KPIs banner ───────────────────────────────────────────────
    avg_pred   = df["pred_attractivite_2026"].mean()
    nb_up      = len(df[df["pred_attractivite_2026"] > df["score_attractivite"]])
    nb_risk    = len(df[df["risk_score"] > 0.70])
    nb_opp     = len(df[df["opportunity_score"] > 0.65])
    trend_terr = "+" if avg_pred > df["score_attractivite"].mean() else ""

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;">
      <div style="background:#fff;border:1px solid #E2E8F2;border-radius:16px;padding:18px 20px;
           border-top:3px solid #1A56DB;box-shadow:0 2px 8px rgba(10,15,30,.05);">
        <div style="font-size:10px;font-weight:700;color:#8895AA;text-transform:uppercase;
             letter-spacing:.8px;margin-bottom:8px;">🔮 Attractivité moy. 2026</div>
        <div style="font-size:30px;font-weight:800;color:#1A56DB;letter-spacing:-.8px;">{avg_pred:.0%}</div>
        <div style="font-size:11px;color:#059669;margin-top:4px;font-weight:600;">{trend_terr} tendance territoire</div>
      </div>
      <div style="background:#fff;border:1px solid #E2E8F2;border-radius:16px;padding:18px 20px;
           border-top:3px solid #059669;box-shadow:0 2px 8px rgba(10,15,30,.05);">
        <div style="font-size:10px;font-weight:700;color:#8895AA;text-transform:uppercase;
             letter-spacing:.8px;margin-bottom:8px;">📈 Communes en hausse</div>
        <div style="font-size:30px;font-weight:800;color:#059669;letter-spacing:-.8px;">{nb_up}</div>
        <div style="font-size:11px;color:#8895AA;margin-top:4px;">score prédit > score 2025</div>
      </div>
      <div style="background:#fff;border:1px solid #E2E8F2;border-radius:16px;padding:18px 20px;
           border-top:3px solid #DC2626;box-shadow:0 2px 8px rgba(10,15,30,.05);">
        <div style="font-size:10px;font-weight:700;color:#8895AA;text-transform:uppercase;
             letter-spacing:.8px;margin-bottom:8px;">⚠️ Zones à risque élevé</div>
        <div style="font-size:30px;font-weight:800;color:#DC2626;letter-spacing:-.8px;">{nb_risk}</div>
        <div style="font-size:11px;color:#DC2626;margin-top:4px;font-weight:600;">risk_score &gt; 70%</div>
      </div>
      <div style="background:#fff;border:1px solid #E2E8F2;border-radius:16px;padding:18px 20px;
           border-top:3px solid #6D28D9;box-shadow:0 2px 8px rgba(10,15,30,.05);">
        <div style="font-size:10px;font-weight:700;color:#8895AA;text-transform:uppercase;
             letter-spacing:.8px;margin-bottom:8px;">🎯 Opportunités détectées</div>
        <div style="font-size:30px;font-weight:800;color:#6D28D9;letter-spacing:-.8px;">{nb_opp}</div>
        <div style="font-size:11px;color:#8895AA;margin-top:4px;">opportunité ML &gt; 65%</div>
      </div>
    </div>""", unsafe_allow_html=True)

    tab_overview, tab_timeline, tab_cluster = st.tabs([
        "📊 Vue globale ML", "📈 Timeline & comparaison", "🗂️ Profils territoriaux"
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — VUE GLOBALE
    # ══════════════════════════════════════════════════════════════
    with tab_overview:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("""
            <div style="font-size:13px;font-weight:700;color:#0A0F1E;margin-bottom:12px;">
              ⚡ Score 2025 → Prédiction 2026 — chaque point = 1 commune</div>""",
            unsafe_allow_html=True)
            df_sc = df[["ville","score_attractivite","pred_attractivite_2026",
                        "risk_score","dept_nom"]].copy()
            df_sc["tendance"] = df_sc.apply(
                lambda r: "↗ Hausse" if r["pred_attractivite_2026"] > r["score_attractivite"]+0.02
                else ("↘ Baisse" if r["pred_attractivite_2026"] < r["score_attractivite"]-0.02
                      else "→ Stable"), axis=1)
            base_sc = alt.Chart(df_sc).mark_circle(size=55, opacity=0.75).encode(
                x=alt.X("score_attractivite:Q", title="Score 2025",
                        scale=alt.Scale(domain=[0,1]),
                        axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8",
                                      labelFontSize=10, titleFontSize=11)),
                y=alt.Y("pred_attractivite_2026:Q", title="Prédiction 2026",
                        scale=alt.Scale(domain=[0,1]),
                        axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8",
                                      labelFontSize=10, titleFontSize=11)),
                color=alt.Color("tendance:N",
                    scale=alt.Scale(
                        domain=["↗ Hausse","→ Stable","↘ Baisse"],
                        range=["#059669","#1A56DB","#DC2626"]),
                    legend=alt.Legend(orient="top-right",
                                      titleFontSize=10, labelFontSize=10)),
                tooltip=["ville","dept_nom","score_attractivite",
                         "pred_attractivite_2026","tendance"]
            ).properties(height=300)
            diag = alt.Chart(pd.DataFrame({"x":[0.,1.],"y":[0.,1.]})).mark_line(
                color="#CBD5E1", strokeDash=[4,4], strokeWidth=1.5
            ).encode(x="x:Q", y="y:Q")
            st.altair_chart(
                alt.layer(diag, base_sc).properties(height=300)
                .configure_view(strokeWidth=0)
                .configure_axis(labelFont="Plus Jakarta Sans",
                                titleFont="Plus Jakarta Sans"),
                use_container_width=True)

        with col_right:
            st.markdown("""
            <div style="font-size:13px;font-weight:700;color:#0A0F1E;margin-bottom:12px;">
              🎯 Top 10 opportunités 2026</div>""", unsafe_allow_html=True)
            top10 = df.nlargest(10,"opportunity_score")[
                ["ville","dept_nom","opportunity_score",
                 "pred_attractivite_2026","risk_score","ml_cluster"]].copy()
            for _, r in top10.iterrows():
                opp  = int(r["opportunity_score"]*100)
                pred = int(r["pred_attractivite_2026"]*100)
                risk = int(r["risk_score"]*100)
                cl   = r["ml_cluster"]
                cl_c = {"Territoire dynamique":"#059669","Zone de vigilance":"#DC2626",
                        "Potentiel émergent":"#D97706","Territoire stable":"#0891B2",
                        "Désert de services":"#7C3AED"}.get(cl,"#64748B")
                bar_c = "#059669" if opp>=70 else ("#1A56DB" if opp>=50 else "#D97706")
                st.markdown(
                    f'<div style="background:#fff;border:1px solid #E2E8F2;border-radius:12px;'
                    f'padding:10px 13px;margin-bottom:7px;transition:all .15s ease;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'margin-bottom:5px;">'
                    f'<span style="font-size:13px;font-weight:700;color:#0A0F1E;">{r["ville"]}</span>'
                    f'<span style="font-size:14px;font-weight:800;color:{bar_c};">{opp}%</span></div>'
                    f'<div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:6px;">'
                    f'<span style="background:{cl_c}18;color:{cl_c};padding:1px 7px;border-radius:20px;'
                    f'font-size:9px;font-weight:700;">🤖 {cl[:16]}</span>'
                    f'<span style="background:#EBF1FF;color:#1A56DB;padding:1px 7px;border-radius:20px;'
                    f'font-size:9px;font-weight:700;">🔮 {pred}% en 2026</span>'
                    f'<span style="background:#FEE2E2;color:#DC2626;padding:1px 7px;border-radius:20px;'
                    f'font-size:9px;font-weight:700;">⚠️ {risk}%</span></div>'
                    f'<div style="background:#F0F4F8;border-radius:3px;height:4px;">'
                    f'<div style="width:{opp}%;height:4px;background:linear-gradient(90deg,#0B1F5C,{bar_c});'
                    f'border-radius:3px;"></div></div></div>',
                    unsafe_allow_html=True)

        # Histo distribution risque
        st.markdown("---")
        st.markdown('<div style="font-size:13px;font-weight:700;color:#0A0F1E;margin-bottom:10px;">'
                    '📊 Distribution des scores de risque ML — 241 communes</div>',
                    unsafe_allow_html=True)
        df_rh = df[["ville","risk_score","dept_nom"]].copy()
        df_rh["cat"] = df_rh["risk_score"].apply(
            lambda s: "🔴 Risque élevé (>70%)" if s>.7
            else ("🟡 Risque modéré (40–70%)" if s>.4 else "🟢 Risque faible (<40%)"))
        hist = alt.Chart(df_rh).mark_bar(
            cornerRadiusTopLeft=4, cornerRadiusTopRight=4
        ).encode(
            x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=25), title="Score de risque",
                    axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8", labelFontSize=10)),
            y=alt.Y("count():Q", title="Nb communes",
                    axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8", labelFontSize=10)),
            color=alt.Color("cat:N", scale=alt.Scale(
                domain=["🔴 Risque élevé (>70%)","🟡 Risque modéré (40–70%)","🟢 Risque faible (<40%)"],
                range=["#DC2626","#F59E0B","#059669"]),
                legend=alt.Legend(orient="top-right", titleFontSize=10, labelFontSize=10)),
            tooltip=["cat","count()"]
        ).properties(height=220).configure_view(strokeWidth=0).configure_axis(
            labelFont="Plus Jakarta Sans")
        st.altair_chart(hist, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — TIMELINE
    # ══════════════════════════════════════════════════════════════
    with tab_timeline:
        st.markdown(
            '<div style="background:linear-gradient(135deg,#EDE9FE,#F5F3FF);border:1px solid #C4B5FD;'
            'border-radius:12px;padding:13px 18px;margin-bottom:18px;font-size:12px;color:#5B21B6;">'
            '🔮 <b>Historique 2019–2024</b> (trait plein · bleu) · '
            '<b>Prévision IA 2025–2030</b> (tirets · violet) — modèle Gradient Boosting entraîné en temps réel'
            '</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            ville_tl = st.selectbox("Commune à analyser", ville_names, key="ville_timeline")
        with c2:
            indicateur_tl = st.selectbox("Indicateur", [
                "attractivite","chomage","entreprises","revenu","prix_m2","opportunite","risque"
            ], key="ind_timeline", format_func=lambda x: {
                "attractivite": "⭐ Score attractivité",
                "chomage":      "📉 Taux de chômage (%)",
                "entreprises":  "🏢 Nb entreprises",
                "revenu":       "💶 Revenu médian (€)",
                "prix_m2":      "🏠 Prix m² (€)",
                "opportunite":  "🎯 Opportunité ML",
                "risque":       "⚠️ Indice de fragilité",
            }[x])

        try:
            from app_iceberg_v4 import generate_timeline_data as _gtl
        except Exception:
            pass
        df_tl = generate_timeline_data(df, ville_tl)

        # Score actuel vs prédit
        row_ville = df[df["ville"] == ville_tl].iloc[0]
        pred_2026 = int(row_ville.get("pred_attractivite_2026", row_ville["score_attractivite"])*100)
        opp_score = int(row_ville.get("opportunity_score", 0)*100)
        risk_sc   = int(row_ville.get("risk_score",0)*100)

        st.markdown(f"""
        <div style="display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap;">
          <div style="background:#EBF1FF;border:1px solid #BFDBFE;border-radius:10px;
               padding:8px 14px;font-size:12px;color:#1A56DB;font-weight:600;">
            🔮 Préd. 2026 : <b>{pred_2026}%</b></div>
          <div style="background:#D1FAE5;border:1px solid #6EE7B7;border-radius:10px;
               padding:8px 14px;font-size:12px;color:#059669;font-weight:600;">
            🎯 Opportunité : <b>{opp_score}%</b></div>
          <div style="background:#FEE2E2;border:1px solid #FCA5A5;border-radius:10px;
               padding:8px 14px;font-size:12px;color:#DC2626;font-weight:600;">
            ⚠️ Indice de fragilité : <b>{risk_sc}%</b></div>
          <div style="background:#EDE9FE;border:1px solid #C4B5FD;border-radius:10px;
               padding:8px 14px;font-size:12px;color:#6D28D9;font-weight:600;">
            🤖 Profil : <b>{row_ville.get('ml_cluster','N/A')}</b></div>
        </div>""", unsafe_allow_html=True)

        # Vérifier si la colonne existe dans generate_timeline_data
        if indicateur_tl not in df_tl.columns:
            indicateur_tl = "attractivite"

        base = alt.Chart(df_tl)
        line_hist = base.transform_filter(alt.datum.type == "Historique").mark_line(
            strokeWidth=2.8, point=alt.OverlayMarkDef(filled=True, size=55)
        ).encode(
            x=alt.X("annee:O", title="Année",
                    axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8", labelFontSize=11)),
            y=alt.Y(f"{indicateur_tl}:Q", title="",
                    axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8", labelFontSize=11)),
            color=alt.value("#1A56DB"),
            tooltip=["annee","type",indicateur_tl]
        )
        line_pred = base.transform_filter(alt.datum.type == "Prévision IA").mark_line(
            strokeWidth=2.5, strokeDash=[6,3],
            point=alt.OverlayMarkDef(filled=True, size=55, shape="diamond")
        ).encode(
            x=alt.X("annee:O"),
            y=alt.Y(f"{indicateur_tl}:Q"),
            color=alt.value("#8B5CF6"),
            tooltip=["annee","type",indicateur_tl]
        )
        area = base.transform_filter(alt.datum.type == "Prévision IA").mark_area(
            opacity=0.10, color="#8B5CF6"
        ).encode(x="annee:O", y=f"{indicateur_tl}:Q")

        st.altair_chart(
            alt.layer(line_hist, area, line_pred).properties(
                height=300,
                title=alt.TitleParams(
                    f"{ville_tl} · Évolution {indicateur_tl} 2019–2030",
                    fontSize=13, fontWeight=700, color="#0A0F1E")
            ).configure_view(strokeWidth=0).configure_axis(
                labelFont="Plus Jakarta Sans", titleFont="Plus Jakarta Sans"),
            use_container_width=True)

        # Légende inline
        st.markdown("""
        <div style="display:flex;gap:20px;align-items:center;padding:6px 0 14px;">
          <div style="display:flex;align-items:center;gap:7px;">
            <div style="width:28px;height:3px;background:#1A56DB;border-radius:2px;"></div>
            <span style="font-size:11px;color:#3D4A63;font-weight:500;">Historique 2019–2024</span>
          </div>
          <div style="display:flex;align-items:center;gap:7px;">
            <div style="width:28px;height:3px;background:#8B5CF6;border-radius:2px;
                 border-top:2px dashed #8B5CF6;"></div>
            <span style="font-size:11px;color:#3D4A63;font-weight:500;">Prévision IA 2025–2030</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # Comparaison multi-villes
        st.markdown("---")
        st.markdown('<div style="font-size:13px;font-weight:700;color:#0A0F1E;margin-bottom:10px;">'
                    '🆚 Comparaison de trajectoires</div>', unsafe_allow_html=True)
        villes_comp = st.multiselect("Jusqu'à 5 communes", ville_names,
                                     default=ville_names[:3], max_selections=5, key="villes_comp")
        if villes_comp:
            frames = []
            for v in villes_comp:
                dv = generate_timeline_data(df, v); dv["ville"] = v; frames.append(dv)
            dm = pd.concat(frames, ignore_index=True)
            col_ind = indicateur_tl if indicateur_tl in dm.columns else "attractivite"
            dh = dm[dm["type"]=="Historique"]; dp = dm[dm["type"]=="Prévision IA"]
            enc_x = alt.X("annee:O", title="Année",
                          axis=alt.Axis(labelColor="#8895AA",gridColor="#F0F4F8",labelFontSize=10))
            enc_y = alt.Y(f"{col_ind}:Q", title="",
                          axis=alt.Axis(labelColor="#8895AA",gridColor="#F0F4F8",labelFontSize=10))
            enc_c = alt.Color("ville:N",
                              legend=alt.Legend(orient="top",titleFontSize=10,labelFontSize=10))
            lh = alt.Chart(dh).mark_line(strokeWidth=2,
                point=alt.OverlayMarkDef(filled=True,size=35)
            ).encode(x=enc_x,y=enc_y,color=enc_c,
                     tooltip=["annee","ville","type",col_ind])
            lp = alt.Chart(dp).mark_line(strokeWidth=2,strokeDash=[6,3],
                point=alt.OverlayMarkDef(filled=True,size=35,shape="diamond")
            ).encode(x=enc_x,y=enc_y,color=enc_c,
                     tooltip=["annee","ville","type",col_ind])
            st.altair_chart(
                alt.layer(lh, lp).properties(height=280)
                .configure_view(strokeWidth=0)
                .configure_axis(labelFont="Plus Jakarta Sans"),
                use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — CLUSTERING
    # ══════════════════════════════════════════════════════════════
    with tab_cluster:
        CLUSTER_META = {
            "Territoire dynamique":{"color":"#059669","bg":"#D1FAE5","icon":"🚀",
                "desc":"Fort tissu éco · Emploi solide · Bonne attractivité"},
            "Zone de vigilance":   {"color":"#DC2626","bg":"#FEE2E2","icon":"⚠️",
                "desc":"Chômage élevé · Pauvreté · Signaux de fragilité"},
            "Désert de services":  {"color":"#D97706","bg":"#FEF3C7","icon":"🏚️",
                "desc":"Accès limité médecins/commerces · Mobilité faible"},
            "Potentiel émergent":  {"color":"#1A56DB","bg":"#EBF1FF","icon":"💡",
                "desc":"Indicateurs en amélioration · Terrain d'investissement"},
            "Territoire stable":   {"color":"#6D28D9","bg":"#EDE9FE","icon":"⚖️",
                "desc":"Équilibré · Peu de risques · Croissance modérée"},
        }

        st.markdown(
            '<div style="font-size:13px;font-weight:700;color:#0A0F1E;margin-bottom:16px;">'
            '🗂️ 5 profils territoriaux — K-Means sur 13 indicateurs</div>',
            unsafe_allow_html=True)

        cluster_counts = df.groupby("ml_cluster").agg(
            nb=("ville","count"),
            moy_attr=("score_attractivite","mean"),
            moy_risk=("risk_score","mean"),
            moy_cho=("taux_chomage","mean"),
            moy_opp=("opportunity_score","mean"),
        ).reset_index()

        c_stats = st.columns(min(5, len(cluster_counts)))
        for i, rc in cluster_counts.iterrows():
            meta = CLUSTER_META.get(rc["ml_cluster"],
                                    {"color":"#64748B","bg":"#F1F5F9","icon":"📍","desc":""})
            with c_stats[i % len(c_stats)]:
                st.markdown(
                    f'<div style="background:{meta["bg"]};border:1.5px solid {meta["color"]}33;'
                    f'border-left:4px solid {meta["color"]};border-radius:14px;padding:16px 14px;">'
                    f'<div style="font-size:18px;margin-bottom:6px;">{meta["icon"]}</div>'
                    f'<div style="font-size:11px;font-weight:700;color:{meta["color"]};'
                    f'margin-bottom:6px;line-height:1.3;">{rc["ml_cluster"]}</div>'
                    f'<div style="font-size:26px;font-weight:800;color:#0A0F1E;">{int(rc["nb"])}</div>'
                    f'<div style="font-size:9px;color:#8895AA;margin-bottom:8px;">communes</div>'
                    f'<div style="font-size:10px;color:#3D4A63;line-height:1.5;">{meta["desc"]}</div>'
                    f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid {meta["color"]}22;'
                    f'display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:9px;color:#8895AA;">'
                    f'<span>Attr. <b style="color:{meta["color"]}">{rc["moy_attr"]:.0%}</b></span>'
                    f'<span>Opp. <b style="color:{meta["color"]}">{rc["moy_opp"]:.0%}</b></span>'
                    f'<span>Risque <b>{rc["moy_risk"]:.0%}</b></span>'
                    f'<span>Chôm. <b>{rc["moy_cho"]:.1f}%</b></span>'
                    f'</div></div>',
                    unsafe_allow_html=True)

        # Carte clusters
        st.markdown("---")
        st.markdown('<div style="font-size:13px;font-weight:700;color:#0A0F1E;margin-bottom:12px;">'
                    '🗺️ Carte satellite des profils territoriaux</div>',
                    unsafe_allow_html=True)
        df_cl = df.dropna(subset=["latitude","longitude"]).copy()
        color_map = {
            "Territoire dynamique": [5,150,105,230],
            "Zone de vigilance":    [220,38,38,230],
            "Désert de services":   [217,119,6,230],
            "Potentiel émergent":   [26,86,219,230],
            "Territoire stable":    [109,40,217,230],
        }
        df_cl["cluster_color"] = df_cl["ml_cluster"].apply(
            lambda c: color_map.get(c, [148,163,184,180]))
        df_cl["cluster_desc"] = df_cl["ml_cluster"].map(
            {k: v["desc"] for k,v in CLUSTER_META.items()}).fillna("")
        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer("ScatterplotLayer", data=df_cl,
                get_position="[longitude,latitude]", get_color="cluster_color",
                get_radius=280, pickable=True, auto_highlight=True, opacity=0.88)],
            initial_view_state=pdk.ViewState(
                latitude=df_cl["latitude"].mean(), longitude=df_cl["longitude"].mean(),
                zoom=10, pitch=35),
            tooltip={"html":"<b>{ville}</b><br>"
                     "<span style='color:#ccc;font-size:11px;'>{ml_cluster}</span><br>"
                     "<i style='font-size:10px;color:#aaa;'>{cluster_desc}</i>"},
            map_style="https://server.arcgisonline.com/ArcGIS/rest/services/"
                      "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
            use_container_width=True)

        # Tableau filtrable
        st.markdown("---")
        col_f1, col_f2 = st.columns([2,1])
        with col_f1:
            cluster_filter = st.selectbox(
                "Filtrer par profil",
                ["Tous"] + sorted(df["ml_cluster"].dropna().unique().tolist()))
        with col_f2:
            dept_filter = st.selectbox(
                "Département", ["Tous","Essonne (91)","Val-de-Marne (94)"], key="dept_cl")

        df_show = df.copy()
        if cluster_filter != "Tous":
            df_show = df_show[df_show["ml_cluster"] == cluster_filter]
        if "91" in dept_filter:
            df_show = df_show[df_show["code_dept"].astype(str) == "91"]
        elif "94" in dept_filter:
            df_show = df_show[df_show["code_dept"].astype(str) == "94"]

        top_cl = df_show.nlargest(25,"opportunity_score")[
            ["ville","dept_nom","ml_cluster","opportunity_score",
             "pred_attractivite_2026","risk_score","taux_chomage"]].copy()
        top_cl["opportunity_score"]      = top_cl["opportunity_score"].apply(lambda x: f"{x:.0%}")
        top_cl["pred_attractivite_2026"] = top_cl["pred_attractivite_2026"].apply(lambda x: f"{x:.0%}")
        top_cl["risk_score"]             = top_cl["risk_score"].apply(lambda x: f"{x:.0%}")
        top_cl["taux_chomage"]           = top_cl["taux_chomage"].apply(lambda x: f"{x:.1f}%")
        top_cl.columns = ["Commune","Dép.","Profil ML","🎯 Opportunité",
                          "🔮 Pred. 2026","⚠️ Risque","💼 Chômage"]
        st.dataframe(top_cl.reset_index(drop=True), use_container_width=True, height=380)
def page_deserts(df: pd.DataFrame):
    page_header("🏥", "Carte des Déserts de Services",
                "Médical · Commercial · Mobilité · Scolaire — Cartographie multi-couches")

    # ── FILTRES ──────────────────────────────────────────────────
    fc1,fc2,fc3,fc4,fc5 = st.columns(5)
    with fc1: aff_med = st.checkbox("🔴 Médical", value=True)
    with fc2: aff_com = st.checkbox("🔵 Commercial", value=True)
    with fc3: aff_mob = st.checkbox("🟣 Mobilité", value=True)
    with fc4: aff_sco = st.checkbox("🎓 Scolaire", value=False)
    with fc5: niv     = st.selectbox("Niveau", ["Fort uniquement","Modéré et Fort","Tous"], index=1)
    niveaux_f = {"Fort uniquement":["Fort"],"Modéré et Fort":["Fort","Modéré"],"Tous":["Fort","Modéré","Faible"]}[niv]

    # ── KPIs ──────────────────────────────────────────────────────
    pop = df["population"].clip(lower=1)
    has_bpe     = df["nb_medecin_generaliste"].sum() > 0
    has_bpe_com = df["nb_epicerie"].sum() > 0
    has_bpe_sco = df["nb_ecole_primaire"].sum() > 0

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.metric("🔴 Déserts médicaux forts",    len(df[df["cat_desert_medical"]=="Fort"]))
    with k2: st.metric("🔵 Déserts comm. forts",       len(df[df["cat_desert_commercial"]=="Fort"]))
    with k3: st.metric("🟣 Déserts mobilité forts",    len(df[df["cat_desert_mobilite"]=="Fort"]))
    with k4: st.metric("🎓 Sans école primaire",       int((df["nb_ecole_primaire"]==0).sum()) if has_bpe_sco else "N/A")

    # ── CARTE PYDECK ─────────────────────────────────────────────
    df_m = df.dropna(subset=["latitude","longitude"]).copy()
    layers = []

    def _heat(data, clr_range, radius=55, intensity=1.3):
        return pdk.Layer("HeatmapLayer", data=data,
            get_position="[longitude,latitude]", get_weight="weight",
            radiusPixels=radius, intensity=intensity, threshold=0.07,
            color_range=clr_range)

    def _scatter(data, color, radius=240):
        # S'assurer que les colonnes du tooltip sont présentes
        cols_needed = ["latitude","longitude","ville","dept_nom"]
        if "niveau" in data.columns:
            cols_needed.append("niveau")
        data_clean = data[[c for c in cols_needed if c in data.columns]].copy()
        return pdk.Layer("ScatterplotLayer", data=data_clean,
            get_position="[longitude,latitude]", get_color=color,
            get_radius=radius, pickable=True, auto_highlight=True, opacity=0.88)

    if aff_med:
        d = df_m[df_m["cat_desert_medical"].isin(niveaux_f)].copy()
        d["weight"] = d["score_desert_medical"]
        d["niveau"] = "Désert médical — " + d["cat_desert_medical"]
        layers += [
            _heat(d, [[254,235,200,0],[253,141,60,120],[240,59,32,200],[189,0,38,255]]),
            _scatter(d, [220,38,38,210])
        ]
    if aff_com:
        d = df_m[df_m["cat_desert_commercial"].isin(niveaux_f)].copy()
        d["weight"] = d["score_desert_commercial"]
        d["niveau"] = "Désert commercial — " + d["cat_desert_commercial"]
        layers += [
            _heat(d, [[235,245,255,0],[96,165,250,120],[37,99,235,200],[29,78,216,255]]),
            _scatter(d, [29,78,216,210])
        ]
    if aff_mob:
        d = df_m[df_m["cat_desert_mobilite"].isin(niveaux_f)].copy()
        d["weight"] = d["score_desert_mobilite"]
        d["niveau"] = "Désert mobilité — " + d["cat_desert_mobilite"]
        layers += [
            _heat(d, [[245,243,255,0],[167,139,250,120],[124,58,237,200],[109,40,217,255]]),
            _scatter(d, [124,58,237,210])
        ]
    if aff_sco and has_bpe_sco:
        d = df_m[df_m["cat_desert_primaire"].isin(niveaux_f)].copy()
        d["weight"] = d["score_desert_primaire"]
        d["niveau"] = "Désert scolaire — " + d["cat_desert_primaire"]
        layers += [
            _heat(d, [[240,253,244,0],[134,239,172,100],[22,163,74,200],[15,118,52,255]]),
            _scatter(d, [22,163,74,210])
        ]

    # ── CARTE FOLIUM — Points simples ────────────────────────────
    if not _FOLIUM_OK:
        st.error("⚠️ Folium non installé : `pip install folium streamlit-folium`")
    else:
        DESERT_STYLE = {
            "médical":    ("#DC2626", "#FEE2E2", "🔴 Médical"),
            "commercial": ("#1D4ED8", "#DBEAFE", "🔵 Commercial"),
            "mobilité":   ("#7C3AED", "#EDE9FE", "🟣 Mobilité"),
            "scolaire":   ("#059669", "#D1FAE5", "🎓 Scolaire"),
        }

        m_desert = folium.Map(
            location=[df_m["latitude"].mean(), df_m["longitude"].mean()],
            zoom_start=11,
            tiles=None,
            prefer_canvas=True,
        )
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
            attr="CartoDB Voyager",
        ).add_to(m_desert)

        count = 0
        for _, row in df_m.iterrows():
            if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
                continue

            # Collecter tous les déserts actifs pour cette commune
            deserts = []
            if aff_med and row.get("cat_desert_medical","") in niveaux_f:
                deserts.append(("médical", float(row.get("score_desert_medical", 0)),
                                row.get("cat_desert_medical","")))
            if aff_com and row.get("cat_desert_commercial","") in niveaux_f:
                deserts.append(("commercial", float(row.get("score_desert_commercial", 0)),
                                row.get("cat_desert_commercial","")))
            if aff_mob and row.get("cat_desert_mobilite","") in niveaux_f:
                deserts.append(("mobilité", float(row.get("score_desert_mobilite", 0)),
                                row.get("cat_desert_mobilite","")))
            if aff_sco and has_bpe_sco and row.get("cat_desert_primaire","") in niveaux_f:
                deserts.append(("scolaire", float(row.get("score_desert_primaire", 0)),
                                row.get("cat_desert_primaire","")))

            if not deserts:
                continue

            # Pire désert → couleur dominante
            deserts.sort(key=lambda x: x[1], reverse=True)
            dtype, dscore, dniv = deserts[0]
            color_hex, color_bg, emoji_lbl = DESERT_STYLE[dtype]

            # Rayon proportionnel au score (entre 5 et 10)
            radius = int(5 + dscore * 5)

            # Tooltip simple
            tooltip_lines = "<br>".join(
                f"{DESERT_STYLE[d][2]} : {niv}"
                for d, _, niv in deserts
            )
            tooltip_html = (
                f"<b style='font-size:13px;'>{row.get('ville','')}</b>"
                f"<br><span style='color:#64748B;font-size:11px;'>{row.get('dept_nom','')}</span>"
                f"<br>{tooltip_lines}"
                f"<br><span style='font-size:10px;color:#94A3B8;'>👥 {safe_val(row.get('population',0))} hab.</span>"
            )

            folium.CircleMarker(
                location=[float(row["latitude"]), float(row["longitude"])],
                radius=radius,
                color=color_hex,
                fill=True,
                fill_color=color_hex,
                fill_opacity=0.80,
                weight=1.5,
                tooltip=folium.Tooltip(tooltip_html,
                    style="font-family:Plus Jakarta Sans,sans-serif;font-size:12px;"),
            ).add_to(m_desert)
            count += 1

        if count == 0:
            st.info("Sélectionnez au moins un type de désert.")
        else:
            st_folium(m_desert, width=None, height=520,
                      use_container_width=True, returned_objects=[])

    # ── TABLEAUX ──────────────────────────────────────────────────
    st.markdown("---")
    tabs_d = st.tabs(["🔴 Médical","🔵 Commercial","🟣 Mobilité","🎓 Scolaire"])
    for tab, col, label, lbl_cols in [
        (tabs_d[0], "score_desert_medical",    "Désert médical",    ["Commune","Dép.","Score","Niveau","Population"]),
        (tabs_d[1], "score_desert_commercial", "Désert commercial", ["Commune","Dép.","Score","Niveau","Population"]),
        (tabs_d[2], "score_desert_mobilite",   "Désert mobilité",   ["Commune","Dép.","Score","Niveau","Population"]),
    ]:
        with tab:
            cat_col = col.replace("score_","cat_")
            t = df.nlargest(10,col)[["ville","dept_nom",col,cat_col,"population"]].copy()
            t.columns = lbl_cols
            t["Score"] = t["Score"].round(3)
            st.dataframe(t.reset_index(drop=True), use_container_width=True)

    with tabs_d[3]:
        if not has_bpe_sco:
            st.info("Données scolaires non disponibles.")
        else:
            t = df.nlargest(10,"score_desert_primaire")[
                ["ville","dept_nom","nb_ecole_primaire","score_desert_primaire","cat_desert_primaire","population"]
            ].copy()
            t.columns = ["Commune","Dép.","Nb écoles","Score","Niveau","Population"]
            st.dataframe(t.reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 13. PAGE INDICATEURS (améliorée avec Altair)
# ══════════════════════════════════════════════════════════════════
def page_indicateurs(df: pd.DataFrame):
    page_header("📈", "Indicateurs économiques", "Emploi · Économie · Social — Évolution mensuelle simulée")

    random.seed(42)
    mois = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
    moy_cho = float(df["taux_chomage"].mean())
    moy_rev = float(df["revenu_median"].mean())
    moy_prix = float(df["prix_m2_median"].mean())
    nb_ent  = int(df["nb_entreprises_actives"].sum())

    cho_data  = [round(moy_cho + random.uniform(-0.4,0.4),1) for _ in mois]
    crea_data = [random.randint(90,160) for _ in mois]
    ferm_data = [random.randint(30,70) for _ in mois]
    rev_data  = [round(moy_rev+random.uniform(-200,200)) for _ in mois]

    tab_vue, tab_emp, tab_eco = st.tabs(["Vue d'ensemble","📉 Emploi","💶 Économie"])

    with tab_vue:
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(metric_card("Chômage moyen",   f"{moy_cho:.1f}%",  "↗ +0,3 pts", "#DC2626"), unsafe_allow_html=True)
        with c2: st.markdown(metric_card("Créations 3 mois", str(sum(crea_data[-3:])), "↗ +12", "#059669"), unsafe_allow_html=True)
        with c3: st.markdown(metric_card("Fermetures 3 mois",str(sum(ferm_data[-3:])), "⚠ +8", "#D97706"), unsafe_allow_html=True)
        with c4: st.markdown(metric_card("Entreprises actives", safe_val(nb_ent), "", "#1A56DB"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        df_evol = pd.DataFrame({"Mois":mois,"Créations":crea_data,"Fermetures":ferm_data})
        chart_evol = alt.Chart(df_evol.melt("Mois",var_name="Type",value_name="Nb")).mark_line(
            point=alt.OverlayMarkDef(filled=True,size=50), strokeWidth=2.5
        ).encode(
            x=alt.X("Mois:O", sort=mois, axis=alt.Axis(labelColor="#8895AA",gridColor="#F0F4F8",domainColor="#E2E8F2",labelFontSize=10)),
            y=alt.Y("Nb:Q", axis=alt.Axis(labelColor="#8895AA",gridColor="#F0F4F8",domainColor="transparent",labelFontSize=10), title=""),
            color=alt.Color("Type:N", scale=alt.Scale(domain=["Créations","Fermetures"],range=["#059669","#DC2626"]),
                            legend=alt.Legend(orient="top-right",titleFontSize=10,labelFontSize=10)),
            tooltip=["Mois","Type","Nb"]
        ).properties(height=220, title=alt.TitleParams("Créations vs Fermetures d'entreprises · 12 mois",
            fontSize=13,fontWeight=700,color="#0A0F1E")
        ).configure_view(strokeWidth=0).configure_axis(labelFont="Plus Jakarta Sans")
        st.altair_chart(chart_evol, use_container_width=True)

        solde_data = pd.DataFrame({"Mois":mois,"Solde":[c-f for c,f in zip(crea_data,ferm_data)]})
        solde_data["Couleur"] = solde_data["Solde"].apply(lambda x: "Positif" if x>=0 else "Négatif")
        bar_solde = alt.Chart(solde_data).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X("Mois:O",sort=mois,axis=alt.Axis(labelColor="#8895AA",gridColor="#F0F4F8",domainColor="#E2E8F2",labelFontSize=10)),
            y=alt.Y("Solde:Q",axis=alt.Axis(labelColor="#8895AA",gridColor="#F0F4F8",domainColor="transparent",labelFontSize=10),title=""),
            color=alt.Color("Couleur:N",scale=alt.Scale(domain=["Positif","Négatif"],range=["#059669","#DC2626"]),legend=None),
            tooltip=["Mois","Solde"]
        ).properties(height=180,title=alt.TitleParams("Solde net mensuel (Créations − Fermetures)",
            fontSize=13,fontWeight=700,color="#0A0F1E")
        ).configure_view(strokeWidth=0).configure_axis(labelFont="Plus Jakarta Sans")
        st.altair_chart(bar_solde, use_container_width=True)

    with tab_emp:
        e1,e2,e3 = st.columns(3)
        with e1: st.markdown(metric_card("Chômage moy.", f"{moy_cho:.1f}%", "↗ +0,3 pts vs M-1","#DC2626"), unsafe_allow_html=True)
        with e2: st.markdown(metric_card("Revenu médian", f"{int(moy_rev):,} €".replace(",","\u00a0"), "↗ +1,2% vs N-1","#059669"), unsafe_allow_html=True)
        with e3: st.markdown(metric_card("Taux pauvreté moy.", f"{df['taux_pauvrete'].mean():.1f}%","↗ +0,5 pts","#DC2626"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        df_cho = pd.DataFrame({"Mois":mois,"Chômage (%)":cho_data})
        line_cho = alt.Chart(df_cho).mark_area(
            line={"color":"#DC2626","strokeWidth":2.5},
            color=alt.Gradient(gradient="linear",
                stops=[alt.GradientStop(color="#FEE2E2",offset=1),alt.GradientStop(color="#FEE2E200",offset=0)],
                x1=1,x2=1,y1=1,y2=0),
            point=alt.OverlayMarkDef(filled=True,fill="#DC2626",size=45)
        ).encode(
            x=alt.X("Mois:O",sort=mois,axis=alt.Axis(labelColor="#8895AA",gridColor="#F0F4F8",domainColor="#E2E8F2",labelFontSize=10)),
            y=alt.Y("Chômage (%):Q",scale=alt.Scale(zero=False),axis=alt.Axis(labelColor="#8895AA",gridColor="#F0F4F8",labelFontSize=10),title=""),
            tooltip=["Mois","Chômage (%)"]
        ).properties(height=200,title=alt.TitleParams("Évolution mensuelle du chômage",fontSize=13,fontWeight=700,color="#0A0F1E")
        ).configure_view(strokeWidth=0).configure_axis(labelFont="Plus Jakarta Sans")
        st.altair_chart(line_cho, use_container_width=True)

        top_cho = df.nlargest(10,"taux_chomage")[["ville","dept_nom","taux_chomage","revenu_median","taux_pauvrete"]].copy()
        top_cho.columns = ["Commune","Département","Chômage %","Revenu médian €","Pauvreté %"]
        st.dataframe(top_cho.reset_index(drop=True), use_container_width=True)

    with tab_eco:
        ec1,ec2,ec3 = st.columns(3)
        with ec1: st.markdown(metric_card("Total entreprises", safe_val(nb_ent), "", "#1A56DB"), unsafe_allow_html=True)
        with ec2: st.markdown(metric_card("Prix m² moyen", f"{int(moy_prix):,} €".replace(",","\u00a0"), "↗ +3,1% vs N-1","#D97706"), unsafe_allow_html=True)
        with ec3: st.markdown(metric_card("Pauvreté moy.", f"{df['taux_pauvrete'].mean():.1f}%", "↗ +0,5 pts", "#DC2626"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        top_ent = df.nlargest(10,"entreprises_1000hab")[["ville","dept_nom","entreprises_1000hab","nb_entreprises_actives","prix_m2_median"]].copy()
        top_ent.columns = ["Commune","Dép.","Entr./1000 hab","Nb total","Prix m²"]
        top_ent["Entr./1000 hab"] = top_ent["Entr./1000 hab"].round(1)
        st.dataframe(top_ent.reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 14. PAGE OPPORTUNITÉS IA
# ══════════════════════════════════════════════════════════════════
def page_opportunites(df: pd.DataFrame):
    page_header("💡", "Opportunités IA", "Signaux détectés par le moteur d'intelligence territoriale ICEBERG")
    df = build_ml_features(df)

    def _gen_signaux(df):
        signaux = []
        for col, type_s, titre_s in [
            ("score_signal_faible", "Emploi",  "Fragilité économique"),
            ("score_desert_medical","Médical",  "Désert médical"),
            ("score_attractivite",  "Foncier",  "Opportunité d'investissement"),
        ]:
            row = df.nlargest(1, col).iloc[0]
            score = int(row[col]*100)
            conf  = min(97, int(row[col]*90+7))
            cho = float(row.get("taux_chomage",0)); pauv = float(row.get("taux_pauvrete",0))
            nb_gen = int(row.get("nb_medecin_generaliste",0))
            pop_v  = int(row.get("population",0))
            prix_m2= int(row.get("prix_m2_median",0))
            nb_ent = int(row.get("nb_entreprises_actives",0))
            opp_s  = float(row.get("opportunity_score",row[col]))
            pred_a = float(row.get("pred_attractivite_2026",row.get("score_attractivite",0)))
            risk_s = float(row.get("risk_score",0))

            if type_s == "Emploi":
                signal = f"Chômage {cho:.1f}% · Pauvreté {pauv:.1f}% · Signal fragilité élevé"
                analyse = (f"{row['ville']} présente un profil de fragilité parmi les plus préoccupants. "
                           f"Chômage {cho:.1f}%, pauvreté {pauv:.1f}%. Intervention coordonnée recommandée.")
                urgence = "Haute" if row[col]>=0.75 else "Modérée"
            elif type_s == "Médical":
                signal = f"{nb_gen} généraliste(s) pour {pop_v:,} hab · Zone sous-dotée".replace(",","\u00a0")
                analyse = (f"{row['ville']} est en situation de désert médical avéré. "
                           f"Avec {nb_gen} généraliste(s) pour {pop_v:,} habitants, "
                           f"une maison de santé représente une opportunité prioritaire.".replace(",","\u00a0"))
                urgence = "Haute" if row[col]>=0.75 else "Modérée"
            else:
                signal = f"Score attractivité {score}/100 · {nb_ent} entreprises · Prix m² {prix_m2:,} €".replace(",","\u00a0")
                analyse = (f"{row['ville']} est la commune la plus attractive (score {score}/100). "
                           f"Prédiction 2026 : {pred_a:.0%}. Score opportunité global : {opp_s:.0%}.")
                urgence = "Opportunité"

            signaux.append({
                "ville":row["ville"],"dept":str(row.get("dept_nom","")),"type":type_s,
                "urgence":urgence,"score":score,"confiance":conf,"nb_ent":nb_ent,
                "opp_score":f"{opp_s:.0%}","pred_2026":f"{pred_a:.0%}","risk":f"{risk_s:.0%}",
                "signal":signal,"analyse":analyse,"delai":"Il y a 2h",
                "sources":{"Emploi":["INSEE","SIRENE","DARES"],"Médical":["RPPS","ARS","INSEE"],"Foncier":["DVF","SIRENE","INSEE"]}[type_s],
                "contacts": {
                    "Emploi": [
                        {"ini":"FT","nom":"France Travail","poste":"Agence locale","clr":"#1A56DB","tags":["Reconversion","Emploi"],"note":"Dispositifs de reconversion disponibles"},
                        {"ini":"BP","nom":"BPI France","poste":"Financement PME","clr":"#059669","tags":["Prêts","Urgence"],"note":"Dossiers urgents sous 15 jours"},
                        {"ini":"CC","nom":"CCI Essonne","poste":"Réseau entreprises","clr":"#D97706","tags":["Réseau local"],"note":"600+ entreprises membres"},
                    ],
                    "Médical": [
                        {"ini":"AR","nom":"ARS Île-de-France","poste":"Agence Régionale Santé","clr":"#DC2626","tags":["Zonage","Subventions"],"note":"Valide dossiers MSP et aides installation"},
                        {"ini":"OM","nom":"Ordre des médecins","poste":"CDOM 91","clr":"#6D28D9","tags":["Réseau médecins"],"note":"Mobilise médecins en attente installation"},
                        {"ini":"CP","nom":"CPAM Essonne","poste":"Financement DSP","clr":"#D97706","tags":["DSP","Financement"],"note":"Expert zones sous-dotées"},
                    ],
                    "Foncier": [
                        {"ini":"UR","nom":"Service Urbanisme","poste":f"Mairie {row['ville']}","clr":"#1A56DB","tags":["PLU","Foncier"],"note":"Pour projets > 500m²"},
                        {"ini":"GP","nom":"Grand Paris Invest.","poste":"Accompagnement","clr":"#059669","tags":["Accompagnement gratuit"],"note":"Projets > 5M€"},
                        {"ini":"BP","nom":"BPI France","poste":"Financement","clr":"#6D28D9","tags":["Prêts","Garanties"],"note":"Prêts pour projets innovants"},
                    ],
                }[type_s],
            })
        return signaux

    signaux = _gen_signaux(df)
    col_list, col_detail = st.columns([2,3])

    with col_list:
        st.markdown('<div style="font-size:13px;font-weight:700;color:#0A0F1E;margin-bottom:12px;">⚡ Signaux détectés</div>', unsafe_allow_html=True)
        urg_bg = {"Haute":("#FEE2E2","#DC2626"),"Modérée":("#FEF3C7","#D97706"),"Opportunité":("#D1FAE5","#059669")}
        type_bg = {"Emploi":("#EBF1FF","#1A56DB"),"Médical":("#D1FAE5","#059669"),"Foncier":("#FEF3C7","#D97706")}

        for i, sig in enumerate(signaux):
            bg_u,clr_u = urg_bg.get(sig["urgence"],("#F1F5F9","#64748B"))
            bg_t,clr_t = type_bg.get(sig["type"],("#F1F5F9","#64748B"))
            is_sel = st.session_state.signal_selectionne==i
            st.markdown(
                f'<div style="background:{"#F0F5FF" if is_sel else "#fff"};'
                f'border:{("2px solid #1A56DB" if is_sel else "1px solid #E2E8F2")};'
                f'border-radius:14px;padding:15px;margin-bottom:9px;'
                f'box-shadow:{"0 2px 12px rgba(26,86,219,.12)" if is_sel else "0 1px 3px rgba(10,15,30,.05)"};">'
                f'<div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:7px;">'
                f'<span style="background:{bg_t};color:{clr_t};padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700;">{sig["type"]}</span>'
                f'<span style="background:{bg_u};color:{clr_u};padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700;">{sig["urgence"]}</span>'
                f'<span style="background:#EBF1FF;color:#1344B8;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700;">{sig["score"]}/100</span>'
                f'</div>'
                f'<div style="font-size:16px;font-weight:800;color:#0A0F1E;letter-spacing:-.3px;">{sig["ville"]}</div>'
                f'<div style="font-size:10px;color:#8895AA;margin:2px 0 6px;">📍 {sig["dept"]}</div>'
                f'<div style="font-size:12px;color:#3D4A63;line-height:1.5;">{sig["signal"]}</div>'
                f'<div style="display:flex;gap:14px;margin-top:9px;font-size:10px;color:#8895AA;">'
                f'<span>🕐 {sig["delai"]}</span><span>✅ {sig["confiance"]}%</span>'
                f'<span>🔮 Opp. {sig["opp_score"]}</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )
            if st.button(f"Voir le détail →", key=f"btn_opp_{i}", use_container_width=True):
                st.session_state.signal_selectionne = i
                st.rerun()

    with col_detail:
        sig = signaux[st.session_state.signal_selectionne]
        bg_u,clr_u = urg_bg.get(sig["urgence"],("#F1F5F9","#64748B"))
        bg_t,clr_t = type_bg.get(sig["type"],("#F1F5F9","#64748B"))

        st.markdown(
            f'<div style="background:#fff;border:1px solid #E2E8F2;border-radius:16px;'
            f'padding:20px 24px;margin-bottom:14px;box-shadow:0 1px 3px rgba(10,15,30,.05);">'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;">'
            f'<span style="background:{bg_t};color:{clr_t};padding:4px 12px;border-radius:20px;font-size:12px;font-weight:700;">{sig["type"]}</span>'
            f'<span style="background:{bg_u};color:{clr_u};padding:4px 12px;border-radius:20px;font-size:12px;font-weight:700;">{sig["urgence"]}</span>'
            f'<span style="background:#EBF1FF;color:#1344B8;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:700;">{sig["score"]}/100</span>'
            f'</div>'
            f'<div style="font-size:22px;font-weight:800;color:#0A0F1E;letter-spacing:-.4px;margin-bottom:4px;">{sig["ville"]}</div>'
            f'<div style="font-size:12px;color:#8895AA;margin-bottom:14px;">📍 {sig["dept"]}</div>'
            f'<div style="background:#F8FAFC;border-radius:11px;padding:14px 16px;margin-bottom:12px;">'
            f'<div style="font-size:13px;font-weight:600;color:#0A0F1E;margin-bottom:8px;">{sig["signal"]}</div>'
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
            f'<div style="background:#fff;border:1px solid #E2E8F2;border-radius:9px;padding:8px 13px;text-align:center;">'
            f'<div style="font-size:9px;color:#8895AA;margin-bottom:1px;">Confiance IA</div>'
            f'<div style="font-size:12px;font-weight:700;color:#0A0F1E;">{sig["confiance"]}%</div></div>'
            f'<div style="background:#D1FAE5;border:1px solid #6EE7B7;border-radius:9px;padding:8px 13px;text-align:center;">'
            f'<div style="font-size:9px;color:#059669;margin-bottom:1px;">Opportunité</div>'
            f'<div style="font-size:12px;font-weight:700;color:#059669;">{sig["opp_score"]}</div></div>'
            f'<div style="background:#EBF1FF;border:1px solid #B8CCFF;border-radius:9px;padding:8px 13px;text-align:center;">'
            f'<div style="font-size:9px;color:#1A56DB;margin-bottom:1px;">Pred. 2026</div>'
            f'<div style="font-size:12px;font-weight:700;color:#1A56DB;">{sig["pred_2026"]}</div></div>'
            f'<div style="background:#FEE2E2;border:1px solid #FCA5A5;border-radius:9px;padding:8px 13px;text-align:center;">'
            f'<div style="font-size:9px;color:#DC2626;margin-bottom:1px;">Risque</div>'
            f'<div style="font-size:12px;font-weight:700;color:#DC2626;">{sig["risk"]}</div></div>'
            f'</div></div></div>',
            unsafe_allow_html=True
        )

        # Sources
        sources_html = "".join([
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:7px 0;border-bottom:1px solid #F8FAFC;">'
            f'<span style="font-size:13px;font-weight:600;color:#3D4A63;">{s}</span>'
            f'<span style="background:#D1FAE5;color:#059669;padding:2px 9px;border-radius:20px;font-size:10px;font-weight:700;">Vérifié</span>'
            f'</div>' for s in sig["sources"]
        ])
        st.markdown(
            f'<div style="background:#fff;border:1px solid #E2E8F2;border-radius:14px;'
            f'padding:14px 18px;margin-bottom:12px;">'
            f'<div style="font-size:12px;font-weight:700;color:#0A0F1E;margin-bottom:9px;">Sources</div>'
            f'{sources_html}</div>',
            unsafe_allow_html=True
        )

        # Analyse IA
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#050C1F,#0B1632);border-radius:14px;'
            f'padding:18px 20px;margin-bottom:12px;">'
            f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:9px;">'
            f'<span style="font-size:15px;">🧊</span>'
            f'<span style="font-size:13px;font-weight:700;color:#F1F5F9;">Analyse ICEBERG v5</span>'
            f'<span class="ml-badge" style="font-size:9px;padding:1px 7px;">ML</span></div>'
            f'<div style="font-size:12px;color:#94A3B8;line-height:1.7;">{sig["analyse"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Contacts
        st.markdown(f'<div style="font-size:14px;font-weight:700;color:#0A0F1E;margin-bottom:10px;">👥 {len(sig["contacts"])} contacts recommandés</div>', unsafe_allow_html=True)
        contacts = sig["contacts"]
        for i in range(0, len(contacts), 2):
            c1, c2 = st.columns(2)
            for j, col in enumerate([c1, c2]):
                if i+j < len(contacts):
                    c = contacts[i+j]
                    tags_h = "".join([
                        f'<span style="background:rgba(255,255,255,.1);color:#94A3B8;padding:1px 7px;'
                        f'border-radius:20px;font-size:9px;font-weight:600;margin-right:3px;">{t}</span>'
                        for t in c["tags"]
                    ])
                    with col:
                        st.markdown(
                            f'<div style="background:#0F172A;border:1px solid rgba(255,255,255,.07);'
                            f'border-radius:13px;padding:14px;margin-bottom:9px;">'
                            f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:9px;">'
                            f'<div style="width:38px;height:38px;border-radius:50%;background:{c["clr"]};'
                            f'display:flex;align-items:center;justify-content:center;font-size:12px;'
                            f'font-weight:700;color:#fff;flex-shrink:0;">{c["ini"]}</div>'
                            f'<div><div style="font-size:12px;font-weight:700;color:#F1F5F9;">{c["nom"]}</div>'
                            f'<div style="font-size:10px;color:#475569;">{c["poste"]}</div></div></div>'
                            f'<div style="margin-bottom:7px;">{tags_h}</div>'
                            f'<div style="font-size:10px;color:#475569;background:rgba(26,86,219,.12);'
                            f'border-radius:7px;padding:7px 9px;margin-bottom:9px;">{c["note"]}</div>'
                            f'<div style="display:flex;gap:5px;">'
                            f'<div style="flex:1;background:rgba(255,255,255,.05);border-radius:7px;'
                            f'padding:5px;text-align:center;font-size:10px;color:#94A3B8;font-weight:600;">✉</div>'
                            f'<div style="flex:1;background:rgba(255,255,255,.05);border-radius:7px;'
                            f'padding:5px;text-align:center;font-size:10px;color:#94A3B8;font-weight:600;">📞</div>'
                            f'<div style="flex:1;background:rgba(255,255,255,.05);border-radius:7px;'
                            f'padding:5px;text-align:center;font-size:10px;color:#94A3B8;font-weight:600;">📅</div>'
                            f'</div></div>',
                            unsafe_allow_html=True
                        )


# ══════════════════════════════════════════════════════════════════
# 15. PAGE COMMUNES
# ══════════════════════════════════════════════════════════════════
def page_communes(df: pd.DataFrame):
    page_header("📊", "Communes", "Classement · Comparaison · Export — 241 communes analysées")
    df = build_ml_features(df)
    ville_names = sorted(df["ville"].dropna().unique().tolist())

    tab_rank, tab_comp = st.tabs(["📋 Classement", "📈 Comparaison"])

    with tab_rank:
        COLS_RANK = {
            "score_attractivite":        "⭐ Attractivité globale",
            "opportunity_score":         "🎯 Opportunité ML 2026",
            "pred_attractivite_2026":    "🔮 Prédiction attractivité 2026",
            "risk_score":                "⚠️ Indice de fragilité",
            "score_signal_faible":       "🚨 Signal faible",
            "score_desert_medical":      "🏥 Désert médical",
            "score_desert_commercial":   "🛒 Désert commercial",
            "score_desert_mobilite":     "🚉 Désert mobilité",
            "taux_chomage":              "📉 Taux chômage (%)",
            "taux_pauvrete":             "💸 Taux pauvreté (%)",
            "revenu_median":             "💶 Revenu médian (€)",
            "prix_m2_median":            "🏠 Prix m² (€)",
            "population":                "👥 Population",
        }
        c1,c2 = st.columns([3,1])
        with c1: indicateur = st.selectbox("Indicateur", list(COLS_RANK.keys()), format_func=lambda x: COLS_RANK[x])
        with c2: ordre = st.selectbox("Ordre", ["↓ Du plus élevé","↑ Du plus bas"])
        asc = "plus bas" in ordre

        df_rank = df[["ville","dept_nom","ml_cluster",indicateur]].sort_values(indicateur,ascending=asc).reset_index(drop=True)
        df_rank.index += 1
        df_rank[indicateur] = df_rank[indicateur].round(3)
        df_rank.columns = ["Commune","Département","Profil ML",COLS_RANK[indicateur]]

        # Podium
        st.markdown("#### 🥇 Podium")
        top3 = list(df_rank.head(3).iterrows())
        medals = [("🥇","#D97706",130,True),("🥈","#64748B",90,False),("🥉","#B45309",60,False)]
        podium_order = [1,0,2] if len(top3)>=3 else list(range(len(top3)))
        max_val = float(top3[0][1][COLS_RANK[indicateur]])
        bar_grads = {"🥇":"linear-gradient(180deg,#FDE68A,#F59E0B)","🥈":"linear-gradient(180deg,#E2E8F0,#94A3B8)","🥉":"linear-gradient(180deg,#FED7AA,#B45309)"}
        cols_p = st.columns(3)
        for i, idx in enumerate(podium_order):
            if idx>=len(top3): continue
            _, row = top3[idx]; med,clr,ph,is_win = medals[idx]
            val = row[COLS_RANK[indicateur]]
            try: pct = int(float(val)/max_val*100) if max_val else 0
            except: pct=0
            with cols_p[i]:
                st.markdown(
                    f'<div style="background:#fff;border:1px solid #E2E8F2;border-radius:18px 18px 0 0;'
                    f'padding:20px 14px 14px;text-align:center;margin-top:{30-(ph-60)//3}px;'
                    f'box-shadow:0 2px 8px rgba(10,15,30,.07);">'
                    f'<div style="font-size:{"38px" if is_win else "30px"};margin-bottom:7px;">{med}</div>'
                    f'<div style="font-size:{"15px" if is_win else "13px"};font-weight:800;color:#0A0F1E;">{row["Commune"]}</div>'
                    f'<div style="font-size:10px;color:#8895AA;margin:3px 0 7px;">{row["Département"]}</div>'
                    f'<div style="background:#F0F4F8;border-radius:4px;height:5px;overflow:hidden;margin-bottom:7px;">'
                    f'<div style="width:{pct}%;height:5px;border-radius:4px;background:linear-gradient(90deg,#1A56DB,#3B82F6);"></div></div>'
                    f'<div style="font-size:{"21px" if is_win else "17px"};font-weight:800;color:{clr};">{val}</div>'
                    f'</div>'
                    f'<div style="background:{bar_grads[med]};height:{ph}px;border-radius:0 0 8px 8px;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:24px;font-weight:900;color:rgba(255,255,255,.6);">{["1","2","3"][idx]}</div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.dataframe(df_rank, use_container_width=True, height=430)
        c1,c2 = st.columns(2)
        with c1: st.download_button("⬇️ Classement CSV", df_rank.to_csv(index=True,encoding="utf-8-sig"), "iceberg_classement.csv","text/csv",use_container_width=True)
        with c2: st.download_button("⬇️ Top 10", df_rank.head(10).to_csv(index=True,encoding="utf-8-sig"), "iceberg_top10.csv","text/csv",use_container_width=True)

    with tab_comp:
        c1,c2 = st.columns(2)
        with c1: A = st.selectbox("Commune A", ville_names, key="comp_A")
        with c2: B = st.selectbox("Commune B", ville_names, index=min(1,len(ville_names)-1), key="comp_B")
        rA = df[df["ville"]==A].iloc[0]; rB = df[df["ville"]==B].iloc[0]

        c1,c2 = st.columns(2)
        with c1: st.markdown(f'<div style="background:#EBF1FF;border:2px solid #1A56DB;border-radius:13px;padding:14px 18px;"><div style="font-size:15px;font-weight:700;color:#1344B8;">📍 {A}</div><div style="font-size:11px;color:#64748B;">{rA["dept_nom"]}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div style="background:#F5F0FF;border:2px solid #6D28D9;border-radius:13px;padding:14px 18px;"><div style="font-size:15px;font-weight:700;color:#5B21B6;">📍 {B}</div><div style="font-size:11px;color:#64748B;">{rB["dept_nom"]}</div></div>', unsafe_allow_html=True)

        # Mini carte 2 communes — Folium avec marqueurs pin
        if _FOLIUM_OK:
            try:
                lat_A = float(rA["latitude"]); lon_A = float(rA["longitude"])
                lat_B = float(rB["latitude"]); lon_B = float(rB["longitude"])
                if any(pd.isna([lat_A, lon_A, lat_B, lon_B])):
                    raise ValueError("Coordonnées manquantes")

                lat_mid = (lat_A + lat_B) / 2
                lon_mid = (lon_A + lon_B) / 2
                dist = ((lat_A - lat_B)**2 + (lon_A - lon_B)**2)**0.5
                zoom_lvl = max(8, min(12, int(12 - dist * 18)))

                m_comp = folium.Map(
                    location=[lat_mid, lon_mid],
                    zoom_start=zoom_lvl,
                    tiles=None,
                    prefer_canvas=True,
                )
                folium.TileLayer(
                    tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
                    attr="CartoDB Voyager",
                    name="Carte",
                ).add_to(m_comp)

                for lat, lon, ville_n, dept_n, score_v, color_hex in [
                    (lat_A, lon_A, A, str(rA.get("dept_nom","")),
                     f"{float(rA.get('score_attractivite',0)):.0%}", "#1A56DB"),
                    (lat_B, lon_B, B, str(rB.get("dept_nom","")),
                     f"{float(rB.get('score_attractivite',0)):.0%}", "#6D28D9"),
                ]:
                    pin_html = (
                        f"<div style='display:flex;flex-direction:column;align-items:center;'>"
                        f"<div style='background:{color_hex};color:#fff;padding:5px 10px;"
                        f"border-radius:20px;font-size:12px;font-weight:700;"
                        f"white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.25);"
                        f"font-family:Plus Jakarta Sans,sans-serif;'>"
                        f"📍 {ville_n}</div>"
                        f"<div style='width:2px;height:8px;background:{color_hex};'></div>"
                        f"<div style='width:8px;height:8px;border-radius:50%;background:{color_hex};'></div>"
                        f"</div>"
                    )
                    popup_html = (
                        f"<div style='font-family:sans-serif;min-width:150px;'>"
                        f"<b style='font-size:14px;color:#0A0F1E;'>{ville_n}</b><br>"
                        f"<span style='font-size:12px;color:#64748B;'>{dept_n}</span><br>"
                        f"<span style='font-size:13px;font-weight:700;color:{color_hex};'>⭐ {score_v}</span>"
                        f"</div>"
                    )
                    folium.Marker(
                        location=[lat, lon],
                        icon=folium.DivIcon(
                            html=pin_html,
                            icon_size=(max(120, len(ville_n)*10), 50),
                            icon_anchor=(max(60, len(ville_n)*5), 50),
                        ),
                        popup=folium.Popup(popup_html, max_width=200),
                        tooltip=folium.Tooltip(
                            f"<b>{ville_n}</b> · {dept_n} · {score_v}",
                            style="font-family:sans-serif;font-size:12px;"
                        ),
                    ).add_to(m_comp)

                st_folium(m_comp, width=None, height=420, use_container_width=True, returned_objects=[])
            except Exception:
                st.info(f"📍 **{A}** · {rA.get('dept_nom','')} &nbsp;|&nbsp; 📍 **{B}** · {rB.get('dept_nom','')} — coordonnées non disponibles.")
        else:
            st.info(f"📍 **{A}** · {rA.get('dept_nom','')} &nbsp;|&nbsp; 📍 **{B}** · {rB.get('dept_nom','')} — installez folium pour la carte.")

        # Tableau comparaison
        COLS_COMP = {
            "taux_chomage":"Chômage (%)","taux_pauvrete":"Pauvreté (%)","revenu_median":"Revenu médian (€)",
            "prix_m2_median":"Prix m² (€)","nb_entreprises_actives":"Nb entreprises",
            "score_attractivite":"⭐ Attractivité","opportunity_score":"🎯 Opportunité ML",
            "pred_attractivite_2026":"🔮 Pred. 2026","risk_score":"⚠️ Indice de fragilité",
            "score_signal_faible":"🚨 Signal faible","score_desert_medical":"🏥 Désert médical",
            "ml_cluster":"🗂️ Profil ML",
        }
        rows_c = []
        for col,lbl in COLS_COMP.items():
            try:
                va = round(float(rA[col]),3); vb = round(float(rB[col]),3)
                winner = "🔵 A" if va>vb else ("🔴 B" if vb>va else "═ Égal")
                rows_c.append({"Indicateur":lbl, A:va, B:vb, "Meilleur":winner})
            except:
                rows_c.append({"Indicateur":lbl, A:str(rA.get(col,"N/A")), B:str(rB.get(col,"N/A")), "Meilleur":"—"})
        st.dataframe(pd.DataFrame(rows_c), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 16. PAGE ASSISTANT IA
# ══════════════════════════════════════════════════════════════════
def get_context(df: pd.DataFrame) -> str:
    df = build_ml_features(df)
    t_attr = df.nlargest(5,"score_attractivite")[["ville","score_attractivite","cat_attractivite"]].to_dict("records")
    t_med  = df.nlargest(5,"score_desert_medical")[["ville","score_desert_medical","cat_desert_medical"]].to_dict("records")
    t_opp  = df.nlargest(5,"opportunity_score")[["ville","opportunity_score","ml_cluster"]].to_dict("records")
    return (
        f"Tu es ICEBERG v5, expert en intelligence territoriale (Essonne 91, Val-de-Marne 94, 241 communes). "
        f"Tu intègres un modèle ML (Gradient Boosting) pour prédire l'attractivité 2026. "
        f"Top attractivité: {t_attr}. Top déserts méd: {t_med}. Top opportunités ML: {t_opp}. "
        f"Zones prioritaires: {len(df[df['cat_attractivite']=='Zone Prioritaire'])}. "
        f"Risques élevés (ML): {len(df[df['risk_score']>0.7])}. "
        f"Réponds en français, de façon concise et professionnelle avec des chiffres précis."
    )


def page_assistant(df: pd.DataFrame):
    moteur = "Mistral AI + Groq fallback" if USE_MISTRAL else "Groq · Llama-3.3-70B"
    page_header("💬", "Assistant IA IceCube", f"Propulsé par {moteur} · Intelligence territoriale ML v5")

    st.markdown(
        '<div style="background:#fff;border:1px solid #E2E8F2;border-radius:15px;'
        'padding:14px 18px;margin-bottom:18px;display:flex;align-items:center;gap:12px;">'
        '<div style="width:42px;height:42px;background:linear-gradient(135deg,#1A56DB,#3B82F6);'
        'border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">🤖</div>'
        '<div style="flex:1;">'
        '<div style="font-size:13px;font-weight:700;color:#0A0F1E;">Assistant IceCube v5</div>'
        '<div style="font-size:11px;color:#8895AA;">Analyse + ML prédictif · 241 communes · Dép. 91 &amp; 94</div>'
        '</div><div style="background:#D1FAE5;border:1px solid #6EE7B7;border-radius:20px;'
        'padding:4px 12px;font-size:10px;font-weight:700;color:#059669;">● En ligne</div></div>',
        unsafe_allow_html=True
    )

    for msg in st.session_state.messages:
        if msg["role"]=="user":
            st.markdown(
                f'<div style="display:flex;justify-content:flex-end;margin:7px 0;">'
                f'<div style="background:linear-gradient(135deg,#1A56DB,#2563EB);color:#fff;'
                f'padding:11px 15px;border-radius:16px 16px 4px 16px;max-width:75%;'
                f'font-size:13px;line-height:1.5;box-shadow:0 2px 8px rgba(26,86,219,.25);">'
                f'{msg["content"]}</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="display:flex;justify-content:flex-start;margin:7px 0;">'
                f'<div style="background:#fff;border:1px solid #E2E8F2;color:#1E293B;'
                f'padding:11px 15px;border-radius:4px 16px 16px 16px;max-width:80%;'
                f'font-size:13px;line-height:1.5;box-shadow:0 1px 3px rgba(10,15,30,.05);">'
                f'{msg["content"]}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown('<p style="font-size:10px;font-weight:700;color:#8895AA;text-transform:uppercase;letter-spacing:.9px;margin-bottom:8px;">Suggestions</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    q1 = c1.button("⭐ Meilleures zones",      use_container_width=True)
    q2 = c2.button("🏥 Déserts médicaux",       use_container_width=True)
    q3 = c3.button("🔮 Prédictions 2026",       use_container_width=True)
    q4 = c4.button("⚠️ Zones à risque ML",      use_container_width=True)

    col1,col2 = st.columns([5,1])
    with col1:
        question = st.text_input("", placeholder="Posez n'importe quelle question sur les communes, ML, prédictions...",
                                 label_visibility="collapsed")
    with col2:
        envoyer = st.button("Envoyer", use_container_width=True)

    if q1: question="Quelles sont les meilleures zones pour investir en 2026 ?"; envoyer=True
    if q2: question="Quelles communes sont en désert médical critique ?"; envoyer=True
    if q3: question="Quelles communes auront la plus forte hausse d'attractivité en 2026 selon le modèle ML ?"; envoyer=True
    if q4: question="Quelles communes présentent un risque élevé selon le modèle ML ?"; envoyer=True

    if envoyer and question:
        st.session_state.messages.append({"role":"user","content":question})
        with st.spinner("🧊 Analyse IA en cours..."):
            try:
                df_ml = build_ml_features(df)
                communes_m = [v for v in df_ml["ville"].unique() if v.lower() in question.lower()]
                ctx_communes = ""
                for v in communes_m[:3]:
                    r = df_ml[df_ml["ville"]==v].iloc[0]
                    ctx_communes += (f"\n📌 {v} ({r['code_dept']}) : Pop. {safe_val(r['population'])} · "
                                     f"Chôm. {safe_val(r['taux_chomage'],1)}% · "
                                     f"Opp. {r['opportunity_score']:.0%} · "
                                     f"Pred.2026 {r['pred_attractivite_2026']:.0%} · "
                                     f"Profil: {r['ml_cluster']}\n")

                system_p = get_context(df)
                hist_p = "".join([f"{'U' if m['role']=='user' else 'A'}: {m['content']}\n" for m in st.session_state.messages[-8:]])
                user_p = (
                    f"Contexte: {len(df)} communes · Chôm. moy. {df['taux_chomage'].mean():.1f}% · "
                    f"Revenu moy. {int(df['revenu_median'].mean()):,} €\n{ctx_communes}\n"
                    f"HISTORIQUE:\n{hist_p}\nQUESTION: {question}\n\n"
                    f"Réponds en français, de façon professionnelle. Cite des communes et chiffres ML précis. Propose une action concrète.\nRÉPONSE:"
                )
                msgs_api = [{"role":"system","content":system_p},{"role":"user","content":user_p}]
                reponse = None
                if USE_MISTRAL and MistralClient and MISTRAL_API_KEY:
                    try:
                        client = MistralClient(api_key=MISTRAL_API_KEY)
                        resp = client.chat.complete(model="mistral-small-latest", messages=msgs_api, max_tokens=900, temperature=0.7)
                        reponse = resp.choices[0].message.content
                    except: reponse = None
                if reponse is None:
                    if _GROQ_OK and GROQ_API_KEY:
                        gc = _GroqClient(api_key=GROQ_API_KEY)
                        gr = gc.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs_api, temperature=0.7, max_tokens=900)
                        reponse = gr.choices[0].message.content
                    else:
                        reponse = "❌ Aucune clé API configurée."
            except Exception as e:
                reponse = f"❌ Erreur : {str(e)}"

        st.session_state.messages.append({"role":"assistant","content":reponse})
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# 17. MAIN — ROUTEUR
# ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════
# RAPPORT ZONES PRIORITAIRES — Directeur uniquement
# ══════════════════════════════════════════════════════════════════
def _generate_rapport_html(titre: str, sous_titre: str, intro: str,
                           methodologie: str, rows_html: str,
                           sources: list, date_str: str) -> str:
    sources_html = "".join(
        f"<tr><td style='padding:6px 10px;font-weight:600;color:#1E293B;border-bottom:1px solid #F0F4F8;'>"
        f"[{i+1}]</td>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #F0F4F8;color:#475569;'>{s['nom']}</td>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #F0F4F8;color:#475569;font-style:italic;'>{s['desc']}</td>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #F0F4F8;'>"
        f"<span style='color:#1A56DB;'>{s['url']}</span></td></tr>"
        for i, s in enumerate(sources)
    )
    return f"""<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#fff; color:#1E293B; font-size:13px; }}
  .cover {{ background:linear-gradient(135deg,#0B1F5C,#1A56DB); color:#fff;
            padding:52px 60px 44px; page-break-after:always; }}
  .cover h1 {{ font-size:32px; font-weight:800; letter-spacing:-.5px; margin-bottom:8px; }}
  .cover .sub {{ font-size:15px; opacity:.8; margin-bottom:32px; }}
  .cover .badge {{ display:inline-block; background:rgba(255,255,255,.15);
                   border-radius:20px; padding:4px 16px; font-size:11px;
                   font-weight:700; letter-spacing:.5px; margin-right:8px; }}
  .cover .meta {{ margin-top:36px; font-size:11px; opacity:.6; }}
  .body {{ padding:44px 60px; }}
  h2 {{ font-size:18px; font-weight:800; color:#0B1F5C; margin:28px 0 10px;
        border-bottom:3px solid #1A56DB; padding-bottom:6px; }}
  h3 {{ font-size:14px; font-weight:700; color:#1E293B; margin:18px 0 6px; }}
  p {{ line-height:1.7; color:#475569; margin-bottom:10px; }}
  .intro-box {{ background:#EFF6FF; border-left:4px solid #1A56DB; border-radius:0 8px 8px 0;
                padding:14px 18px; margin:16px 0 22px; }}
  .methodo {{ background:#F8FAFC; border:1px solid #E2E8F2; border-radius:10px;
              padding:16px 20px; margin:16px 0 22px; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:12px; }}
  th {{ background:#0B1F5C; color:#fff; padding:9px 12px; text-align:left;
        font-weight:700; font-size:11px; text-transform:uppercase; letter-spacing:.5px; }}
  td {{ padding:8px 12px; border-bottom:1px solid #F0F4F8; vertical-align:top; }}
  tr:nth-child(even) td {{ background:#F8FAFC; }}
  .score-badge {{ display:inline-block; border-radius:20px; padding:2px 10px;
                  font-size:11px; font-weight:700; }}
  .green {{ background:#D1FAE5; color:#065F46; }}
  .amber {{ background:#FEF3C7; color:#92400E; }}
  .red   {{ background:#FEE2E2; color:#991B1B; }}
  .footer {{ margin-top:40px; padding-top:16px; border-top:2px solid #E2E8F2;
             font-size:10px; color:#94A3B8; display:flex; justify-content:space-between; }}
  .src-table th {{ background:#F1F5F9; color:#0B1F5C; }}
  @media print {{ .cover {{ page-break-after:always; }} }}
</style></head><body>
<div class="cover">
  <div style="font-size:11px;font-weight:700;letter-spacing:2px;opacity:.6;margin-bottom:16px;text-transform:uppercase;">ICEBERG v5.0 · Intelligence Territoriale · Dép. 91 &amp; 94</div>
  <h1>{titre}</h1>
  <div class="sub">{sous_titre}</div>
  <div><span class="badge">CONFIDENTIEL</span><span class="badge">Directeur d'Agence</span><span class="badge">ML Prédictif</span></div>
  <div class="meta">Généré le {date_str} · Modèle Gradient Boosting · 241 communes analysées</div>
</div>
<div class="body">
  <h2>1. Introduction &amp; Contexte</h2>
  <div class="intro-box"><p>{intro}</p></div>
  <h2>2. Méthodologie</h2>
  <div class="methodo">{methodologie}</div>
  <h2>3. Résultats détaillés</h2>
  {rows_html}
  <h2>4. Sources &amp; Références</h2>
  <table class="src-table">
    <tr><th>#</th><th>Source</th><th>Description</th><th>Référence</th></tr>
    {sources_html}
  </table>
  <div class="footer">
    <span>ICEBERG v5.0 · Rapport confidentiel · Usage interne Directeur d'Agence</span>
    <span>Généré le {date_str}</span>
  </div>
</div></body></html>"""


def page_rapport_prioritaires(df: pd.DataFrame):
    page_header("📋", "Rapport Zones Prioritaires",
                "Analyse détaillée · Données ML · Export PDF — Accès Directeur",
                badge="Confidentiel")

    df = build_ml_features(df)
    df_p = df[df["cat_attractivite"] == "Zone Prioritaire"].copy()
    df_p["score_pct"] = (df_p["score_attractivite"]*100).round(1)
    df_p["opp_pct"]   = (df_p["opportunity_score"]*100).round(1)
    df_p["pred_pct"]  = (df_p["pred_attractivite_2026"]*100).round(1)
    df_p = df_p.sort_values("score_attractivite", ascending=False)

    # ── Bannière ──────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#D1FAE5,#ECFDF5);border:1px solid #6EE7B7;
         border-radius:14px;padding:16px 22px;margin-bottom:22px;display:flex;
         align-items:center;gap:16px;">
      <div style="font-size:36px;">🟢</div>
      <div>
        <div style="font-size:20px;font-weight:800;color:#065F46;">{len(df_p)} communes prioritaires</div>
        <div style="font-size:13px;color:#047857;margin-top:2px;">Score attractivité ≥ 70% · Opportunité d'investissement immédiate · Dép. 91 &amp; 94</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Tableau interactif ────────────────────────────────────────
    st.markdown("#### 📊 Tableau complet des zones prioritaires")
    cols_show = ["ville","dept_nom","score_pct","opp_pct","pred_pct",
                 "taux_chomage","revenu_median","population","prix_m2_median","ml_cluster"]
    cols_rename = {"ville":"Commune","dept_nom":"Département","score_pct":"Score %",
                   "opp_pct":"Opportunité ML %","pred_pct":"Prédiction 2026 %",
                   "taux_chomage":"Chômage %","revenu_median":"Revenu méd. €",
                   "population":"Population","prix_m2_median":"Prix m²","ml_cluster":"Profil ML"}
    df_show = df_p[[c for c in cols_show if c in df_p.columns]].rename(columns=cols_rename)
    st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=400)

    # ── Fiches détaillées ─────────────────────────────────────────
    st.markdown("#### 🔍 Fiches analytiques détaillées")
    for _, row in df_p.iterrows():
        secteurs = _compute_secteurs(row)
        top3_sec = secteurs[:3]
        with st.expander(f"🟢 {row['ville']} — {row['score_pct']:.0f}% · {row.get('dept_nom','')}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score attractivité", f"{row['score_pct']:.0f}%")
            c2.metric("Opportunité ML",     f"{row['opp_pct']:.0f}%")
            c3.metric("Prédiction 2026",    f"{row['pred_pct']:.0f}%")
            c4.metric("Profil ML",          str(row.get("ml_cluster","N/A")))
            st.markdown("**Pourquoi cette commune est prioritaire :**")
            reasons = []
            if row["score_attractivite"] >= 0.80: reasons.append("✅ Score attractivité exceptionnel (≥ 80%)")
            elif row["score_attractivite"] >= 0.70: reasons.append("✅ Score attractivité élevé (≥ 70%)")
            if row.get("taux_chomage", 15) < 10: reasons.append(f"✅ Faible chômage ({row['taux_chomage']:.1f}%) — bassin d'emploi sain")
            if row.get("revenu_median", 0) > 28000: reasons.append(f"✅ Revenu médian élevé ({int(row.get('revenu_median',0)):,} €) — pouvoir d'achat fort")
            if row.get("opportunity_score", 0) >= 0.70: reasons.append("✅ Opportunité ML ≥ 70% — modèle prédit forte dynamique")
            if row.get("pred_attractivite_2026", 0) > row["score_attractivite"]: reasons.append("✅ Trajectoire ascendante prévue en 2026")
            for r in reasons:
                st.markdown(f"- {r}")
            st.markdown("**Secteurs recommandés :**")
            sec_cols = st.columns(3)
            for i, (ico, lbl, sc) in enumerate(top3_sec):
                with sec_cols[i]:
                    c = "#16A34A" if sc >= 70 else "#1A56DB" if sc >= 50 else "#D97706"
                    st.markdown(
                        f"<div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;"
                        f"padding:10px;text-align:center;'>"
                        f"<div style='font-size:20px;'>{ico}</div>"
                        f"<div style='font-weight:700;font-size:12px;'>{lbl}</div>"
                        f"<div style='font-size:14px;font-weight:800;color:{c};'>{sc}%</div>"
                        f"</div>", unsafe_allow_html=True)

    # ── Export PDF ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⬇️ Export")

    rows_html_pdf = "<table><tr><th>Commune</th><th>Dép.</th><th>Score</th><th>Opportunité ML</th><th>Préd. 2026</th><th>Chômage</th><th>Revenu méd.</th><th>Profil ML</th><th>Raisons clés</th></tr>"
    for _, row in df_p.iterrows():
        reasons_short = []
        if row["score_attractivite"] >= 0.70: reasons_short.append("Attractivité ≥70%")
        if row.get("taux_chomage",15) < 10: reasons_short.append(f"Chômage bas ({row['taux_chomage']:.1f}%)")
        if row.get("revenu_median",0) > 28000: reasons_short.append("Revenu élevé")
        if row.get("pred_attractivite_2026",0) > row["score_attractivite"]: reasons_short.append("Tendance +2026")
        badge_cls = "green" if row["score_pct"] >= 80 else "amber"
        rows_html_pdf += (
            f"<tr><td><b>{row['ville']}</b></td>"
            f"<td>{row.get('dept_nom','')}</td>"
            f"<td><span class='score-badge {badge_cls}'>{row['score_pct']:.0f}%</span></td>"
            f"<td>{row['opp_pct']:.0f}%</td>"
            f"<td>{row['pred_pct']:.0f}%</td>"
            f"<td>{row.get('taux_chomage',0):.1f}%</td>"
            f"<td>{int(row.get('revenu_median',0)):,} €</td>"
            f"<td>{row.get('ml_cluster','N/A')}</td>"
            f"<td style='font-size:11px;color:#475569;'>{' · '.join(reasons_short)}</td></tr>"
        )
    rows_html_pdf += "</table>"

    sources = [
        {"nom":"INSEE — Fichier Localisation des équipements","desc":"Données socio-économiques communales (population, emploi, revenus)","url":"insee.fr/fr/statistiques"},
        {"nom":"INSEE — Taux de chômage localisés","desc":"Taux de chômage par zone d'emploi et commune","url":"insee.fr/fr/statistiques/2012795"},
        {"nom":"DREES — Atlas de la démographie médicale","desc":"Densité médicale par commune et département","url":"drees.solidarites-sante.gouv.fr"},
        {"nom":"DVF — Demandes de valeurs foncières","desc":"Prix de l'immobilier par commune (transactions notariées)","url":"app.dvf.etalab.gouv.fr"},
        {"nom":"ICEBERG v5.0 — Modèle ML interne","desc":"Score d'attractivité calculé par Gradient Boosting sur 18 indicateurs","url":"Modèle propriétaire ICEBERG"},
        {"nom":"Banque des Territoires — Baromètre des territoires","desc":"Indicateurs de vitalité et fragilité territoriale","url":"banquedesterritoires.fr"},
    ]
    date_str = datetime.datetime.now().strftime("%d/%m/%Y à %H:%M")
    html_content = _generate_rapport_html(
        titre="Rapport — Zones Prioritaires",
        sous_titre=f"{len(df_p)} communes à fort potentiel · Départements 91 & 94",
        intro=(
            f"Ce rapport présente les {len(df_p)} communes des départements de l'Essonne (91) et du "
            f"Val-de-Marne (94) identifiées comme <b>zones prioritaires</b> par le modèle d'intelligence "
            f"artificielle ICEBERG v5.0. Ces communes obtiennent un score d'attractivité territoriale "
            f"supérieur ou égal à 70%, calculé sur la base de 18 indicateurs socio-économiques. "
            f"Elles représentent les meilleures opportunités d'investissement, d'implantation commerciale "
            f"ou de déploiement de services publics et privés sur le territoire."
        ),
        methodologie=(
            "<h3>Indicateurs utilisés (18 variables)</h3>"
            "<p>Le score d'attractivité est calculé par un modèle <b>Gradient Boosting</b> entraîné "
            "sur les données historiques 2019-2024. Les 18 indicateurs incluent :</p>"
            "<ul style='margin:8px 0 8px 20px;color:#475569;line-height:1.8;'>"
            "<li><b>Emploi :</b> taux de chômage, taux d'activité, nb d'entreprises actives</li>"
            "<li><b>Économie :</b> revenu médian, prix foncier m², densité commerciale</li>"
            "<li><b>Démographie :</b> population, densité hab/km², solde migratoire</li>"
            "<li><b>Services :</b> couverture médicale, accès mobilité, équipements scolaires</li>"
            "<li><b>ML prédictif :</b> score opportunité, profil cluster, prédiction 2026</li>"
            "</ul>"
            "<p><b>Seuil zone prioritaire :</b> score ≥ 0.70 (normalisé entre 0 et 1).</p>"
        ),
        rows_html=rows_html_pdf,
        sources=sources,
        date_str=date_str
    )
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Télécharger rapport HTML (imprimable PDF)",
            data=html_content.encode("utf-8"),
            file_name=f"rapport_zones_prioritaires_{datetime.datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html",
            use_container_width=True,
            type="primary"
        )
    with c2:
        csv_data = df_show.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "⬇️ Données CSV",
            data=csv_data,
            file_name="zones_prioritaires.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.caption("💡 Pour générer un PDF : ouvrez le fichier HTML dans votre navigateur → Fichier → Imprimer → Enregistrer en PDF")


# ══════════════════════════════════════════════════════════════════
# RAPPORT SIGNAUX FORTS — Directeur uniquement
# ══════════════════════════════════════════════════════════════════
def page_rapport_signaux(df: pd.DataFrame):
    page_header("🚨", "Rapport Signaux Forts",
                "Territoires en fragilité détectés par l'IA · Export PDF — Accès Directeur",
                badge="Confidentiel")

    df = build_ml_features(df)
    df_s = df[df["cat_signal_faible"] == "Signal Fort"].copy()
    df_s["sig_pct"]  = (df_s["score_signal_faible"]*100).round(1)
    df_s["risk_pct"] = (df_s["risk_score"]*100).round(1)
    df_s = df_s.sort_values("score_signal_faible", ascending=False)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#FEF3C7,#FFFBEB);border:1px solid #FDE68A;
         border-radius:14px;padding:16px 22px;margin-bottom:22px;display:flex;
         align-items:center;gap:16px;">
      <div style="font-size:36px;">⚡</div>
      <div>
        <div style="font-size:20px;font-weight:800;color:#92400E;">{len(df_s)} signaux forts détectés</div>
        <div style="font-size:13px;color:#B45309;margin-top:2px;">
          Territoires en fragilité identifiés par l'IA · Nécessitent un plan d'action prioritaire · Dép. 91 &amp; 94</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        "<div style='background:#FFF1F2;border:1px solid #FECDD3;border-radius:10px;"
        "padding:14px 18px;margin-bottom:20px;'>"
        "<b style='color:#9F1239;'>📖 Qu'est-ce qu'un signal fort ?</b><br>"
        "<span style='color:#475569;font-size:13px;line-height:1.7;'>"
        "Un signal fort est détecté par le modèle ML lorsqu'une commune cumule simultanément "
        "plusieurs facteurs de fragilité : <b>chômage élevé</b> (> 12%), <b>taux de pauvreté élevé</b> "
        "(> 20%), <b>perte de commerces</b> (désert commercial), <b>désertification médicale</b>, "
        "ou <b>baisse démographique</b>. Le modèle Gradient Boosting attribue un score de fragilité "
        "normalisé entre 0 et 1 — au-delà de 0.65, le territoire est classé en signal fort."
        "</span></div>",
        unsafe_allow_html=True
    )

    # Tableau
    st.markdown("#### 📊 Tableau complet des signaux forts")
    cols_s = ["ville","dept_nom","sig_pct","risk_pct","taux_chomage","taux_pauvrete",
              "score_desert_medical","score_desert_commercial","population","ml_cluster"]
    cols_r = {"ville":"Commune","dept_nom":"Département","sig_pct":"Signal %",
              "risk_pct":"Indice de fragilité %","taux_chomage":"Chômage %","taux_pauvrete":"Pauvreté %",
              "score_desert_medical":"Désert méd.","score_desert_commercial":"Désert com.",
              "population":"Population","ml_cluster":"Profil ML"}
    df_show_s = df_s[[c for c in cols_s if c in df_s.columns]].rename(columns=cols_r)
    st.dataframe(df_show_s.reset_index(drop=True), use_container_width=True, height=400)

    # Fiches
    st.markdown("#### 🔍 Analyse commune par commune")
    for _, row in df_s.iterrows():
        sev = "🔴 Critique" if row["sig_pct"] >= 75 else ("🟠 Sévère" if row["sig_pct"] >= 60 else "🟡 Modéré")
        freins = _compute_freins(row)
        with st.expander(f"{sev} — {row['ville']} · Signal {row['sig_pct']:.0f}% · {row.get('dept_nom','')}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score signal",  f"{row['sig_pct']:.0f}%")
            c2.metric("Indice de fragilité",     f"{row['risk_pct']:.0f}%")
            c3.metric("Chômage",       f"{row.get('taux_chomage',0):.1f}%")
            c4.metric("Pauvreté",      f"{row.get('taux_pauvrete',0):.1f}%")
            st.markdown("**Freins identifiés par l'IA :**")
            for ico, desc, sev_val in freins:
                sev_lbl   = "Critique" if sev_val > 0.65 else ("Élevé" if sev_val > 0.40 else "Modéré")
                sev_color = "#DC2626" if sev_val > 0.65 else ("#D97706" if sev_val > 0.40 else "#64748B")
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;padding:5px 0;"
                    f"border-bottom:1px solid #F0F4F8;'>"
                    f"<span style='font-size:16px;'>{ico}</span>"
                    f"<span style='flex:1;font-size:12px;'>{desc}</span>"
                    f"<span style='background:{"#FEE2E2" if sev_val>0.65 else "#FEF3C7"};color:{sev_color};"
                    f"padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;'>{sev_lbl}</span>"
                    f"</div>", unsafe_allow_html=True)
            st.markdown("**Plan d'action recommandé :**")
            actions = []
            if row.get("taux_chomage", 0) > 12: actions.append("🎯 Mise en place d'une cellule emploi locale · partenariat France Travail")
            if row.get("taux_pauvrete", 0) > 20: actions.append("🏠 Renforcement des aides sociales · accès au logement social")
            if row.get("score_desert_medical", 0) > 0.6: actions.append("🏥 Incitation à l'installation médicale · maison de santé pluridisciplinaire")
            if row.get("score_desert_commercial", 0) > 0.6: actions.append("🛒 Soutien au commerce de proximité · épicerie solidaire ou marché itinérant")
            if not actions: actions.append("📊 Suivi renforcé des indicateurs · réévaluation dans 6 mois")
            for a in actions:
                st.markdown(f"- {a}")

    # Export
    st.markdown("---")
    rows_html_s = "<table><tr><th>Commune</th><th>Dép.</th><th>Signal %</th><th>Indice de fragilité</th><th>Chômage</th><th>Pauvreté</th><th>Profil ML</th><th>Freins principaux</th></tr>"
    for _, row in df_s.iterrows():
        freins = _compute_freins(row)
        freins_txt = " · ".join(f"{ico} {d[:30]}" for ico, d, _ in freins[:3])
        badge_cls = "red" if row["sig_pct"] >= 75 else "amber"
        rows_html_s += (
            f"<tr><td><b>{row['ville']}</b></td>"
            f"<td>{row.get('dept_nom','')}</td>"
            f"<td><span class='score-badge {badge_cls}'>{row['sig_pct']:.0f}%</span></td>"
            f"<td>{row['risk_pct']:.0f}%</td>"
            f"<td>{row.get('taux_chomage',0):.1f}%</td>"
            f"<td>{row.get('taux_pauvrete',0):.1f}%</td>"
            f"<td>{row.get('ml_cluster','N/A')}</td>"
            f"<td style='font-size:11px;color:#475569;'>{freins_txt}</td></tr>"
        )
    rows_html_s += "</table>"

    sources_s = [
        {"nom":"INSEE — Taux de pauvreté par commune","desc":"Taux de pauvreté au seuil de 60% du revenu médian","url":"insee.fr/fr/statistiques/6036907"},
        {"nom":"INSEE — Taux de chômage localisés","desc":"Chômage trimestriel par zone d'emploi","url":"insee.fr/fr/statistiques/2012795"},
        {"nom":"DREES — Zones sous-dotées","desc":"Cartographie officielle des déserts médicaux","url":"drees.solidarites-sante.gouv.fr/publications-dossiers-de-presse"},
        {"nom":"Observatoire des territoires — ANCT","desc":"Indicateurs de fragilité et de vulnérabilité territoriale","url":"observatoire.anct.gouv.fr"},
        {"nom":"Banque des Territoires — Baromètre 2024","desc":"Perception et réalité des inégalités territoriales","url":"banquedesterritoires.fr/barometre-des-territoires"},
        {"nom":"ICEBERG v5.0 — Score signal faible","desc":"Détection par Gradient Boosting de 9 indicateurs de fragilité cumulés","url":"Modèle propriétaire ICEBERG"},
    ]
    date_str = datetime.datetime.now().strftime("%d/%m/%Y à %H:%M")
    html_s = _generate_rapport_html(
        titre="Rapport — Signaux Forts",
        sous_titre=f"{len(df_s)} territoires en fragilité détectés · Départements 91 & 94",
        intro=(
            f"Ce rapport recense les {len(df_s)} communes des départements de l'Essonne (91) et du "
            f"Val-de-Marne (94) identifiées en <b>signal fort de fragilité territoriale</b> par le "
            f"modèle d'intelligence artificielle ICEBERG v5.0. "
            f"Ces communes cumulent plusieurs facteurs de vulnérabilité socio-économique détectés "
            f"simultanément, nécessitant une attention prioritaire et un plan d'action adapté. "
            f"Le score de signal faible dépasse 65% pour toutes les communes listées dans ce rapport."
        ),
        methodologie=(
            "<h3>Détection des signaux forts</h3>"
            "<p>Le modèle <b>Gradient Boosting</b> analyse 9 indicateurs de fragilité en combinaison :</p>"
            "<ul style='margin:8px 0 8px 20px;color:#475569;line-height:1.8;'>"
            "<li><b>Emploi :</b> taux de chômage &gt; 12%, faiblesse du tissu entrepreneurial</li>"
            "<li><b>Social :</b> taux de pauvreté &gt; 20%, revenu médian &lt; 18 000 €</li>"
            "<li><b>Santé :</b> densité médicale &lt; 2,5 généralistes / 10 000 hab.</li>"
            "<li><b>Commerce :</b> score désert commercial &gt; 0.60</li>"
            "<li><b>Mobilité :</b> accessibilité transports &lt; seuil acceptable</li>"
            "<li><b>Démographie :</b> population &lt; 1 000 hab. ou baisse tendancielle</li>"
            "<li><b>ML :</b> score risque &gt; 0.65 calculé par le modèle Random Forest</li>"
            "</ul>"
            "<p><b>Seuil signal fort :</b> score fragilité normalisé ≥ 0.65.</p>"
        ),
        rows_html=rows_html_s,
        sources=sources_s,
        date_str=date_str
    )
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Télécharger rapport HTML (imprimable PDF)",
            data=html_s.encode("utf-8"),
            file_name=f"rapport_signaux_forts_{datetime.datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html",
            use_container_width=True,
            type="primary"
        )
    with c2:
        st.download_button(
            "⬇️ Données CSV",
            data=df_show_s.to_csv(index=False, encoding="utf-8-sig"),
            file_name="signaux_forts.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.caption("💡 Pour générer un PDF : ouvrez le fichier HTML dans votre navigateur → Fichier → Imprimer → Enregistrer en PDF")






def main():
    init_session_state()

    # Chargement des données
    with st.spinner("🧊 Chargement des données territoriales..."):
        df = load_data()

    # Routing
    if st.session_state.role is None:
        # Masquer complètement la sidebar sur la page d'accueil
        st.markdown("""<style>
        [data-testid="stSidebar"] { display: none !important; }
        .main .block-container { padding-left: 2rem !important; }
        </style>""", unsafe_allow_html=True)
        page_login(df)
        return

    page = render_sidebar(df)
    if page is None:
        return

    # Dispatch pages
    if "Classement admins" in page:
        page_classement(df)
    elif "Attractivité" in page:
        page_attractivite(df)
    elif "Rapport Prioritaires" in page:
        page_rapport_prioritaires(df)
    elif "Rapport Signaux" in page:
        page_rapport_signaux(df)
    elif "Déserts" in page:
        page_deserts(df)
    elif "Indicateurs" in page:
        page_indicateurs(df)
    elif "Opportunités" in page:
        page_opportunites(df)
    elif "Communes" in page:
        page_communes(df)
    elif "Assistant IA" in page:
        page_assistant(df)
    else:
        st.info("Page non trouvée.")



main()
