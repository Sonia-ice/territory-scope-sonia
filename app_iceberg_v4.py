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
                 font-size:18px;box-shadow:0 4px 14px rgba(26,86,219,.4);">🧊</div>
            <div>
              <div style="font-size:16px;font-weight:800;color:#0A0F1E;letter-spacing:.3px;">ICEBERG <span style="font-size:10px;font-weight:500;color:#94A3B8;">v{APP_VERSION}</span></div>
              <div style="font-size:9px;color:#94A3B8;text-transform:uppercase;letter-spacing:1.8px;">Dép. 91 &amp; 94</div>
            </div>
          </div>

          <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);
               border-radius:14px;padding:14px;margin-bottom:6px;">
            <div style="font-size:9px;font-weight:700;color:#94A3B8;text-transform:uppercase;
                 letter-spacing:1.2px;margin-bottom:10px;">Vue d'ensemble</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
              <div style="background:rgba(255,255,255,.05);border-radius:10px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:800;color:#FFFFFF;">{_n}</div>
                <div style="font-size:10px;color:#CBD5E1;margin-top:1px;">Communes</div>
              </div>
              <div style="background:rgba(22,163,74,.12);border-radius:10px;padding:10px 12px;cursor:pointer;">
                <div style="font-size:20px;font-weight:800;color:#34D399;">{_prio}</div>
                <div style="font-size:10px;color:#CBD5E1;margin-top:1px;">Prioritaires ↗</div>
              </div>
              <div style="background:rgba(251,191,36,.10);border-radius:10px;padding:10px 12px;cursor:pointer;">
                <div style="font-size:20px;font-weight:800;color:#FBBF24;">{_sig}</div>
                <div style="font-size:10px;color:#CBD5E1;margin-top:1px;">Signaux forts ↗</div>
              </div>
              <div style="background:rgba(220,38,38,.10);border-radius:10px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:800;color:#F87171;">{_desert}</div>
                <div style="font-size:10px;color:#CBD5E1;margin-top:1px;">Déserts méd.</div>
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
                             use_container_width=True):
                    st.session_state["_nav_override"] = "📋  Rapport Prioritaires"
                    st.rerun()
            with _bb:
                if st.button("🚨 Rapport Signaux", key="btn_goto_sig",
                             use_container_width=True):
                    st.session_state["_nav_override"] = "🚨  Rapport Signaux"
                    st.rerun()

        if st.session_state.role is not None:
            st.markdown('<p style="font-size:13px;color:#FFFFFF;text-transform:uppercase;'
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
      <img src="data:image/png;base64,/9j/4AAQSkZJRgABAQAASABIAAD/4QBMRXhpZgAATU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAADD6ADAAQAAAABAAACeQAAAAD/7QA4UGhvdG9zaG9wIDMuMAA4QklNBAQAAAAAAAA4QklNBCUAAAAAABDUHYzZjwCyBOmACZjs+EJ+/8AAEQgCeQMPAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/bAEMAAQEBAQEBAgEBAgMCAgIDBAMDAwMEBgQEBAQEBgcGBgYGBgYHBwcHBwcHBwgICAgICAkJCQkJCwsLCwsLCwsLC//bAEMBAgICAwMDBQMDBQsIBggLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLC//dAAQAMf/aAAwDAQACEQMRAD8A/wA/+iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD//0P8AP/ooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACivun/gnn+xXJ+3T8e7j4X6t4kXwZ4a0TRb/AMQeIPEBt4706dY2irHGy2j3Nq1y895NbWqxxSbwZw+CqNjuf2/P+CbvxD/YhudK8W6fqL+L/AGueXb2uviz+wyW2o+XvlsL+182f7Jcja7wgyulxCN8bFknjhnnV+W+pu8NVVFYhxfI3a/S+9j83aKKKowCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP/9H/AD/6KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACvVPgj8Efix+0j8V9D+B3wO0O58SeK/EdwLXT9PtQN8j4LszMxVI4o0VpJZZGWOKNWkkZUVmHldf2c/BX4T+I/8AglR+yJdfsn+Jrmzu/i/42luL/wAV3FrcyTp4bsNQitN+iW7iR4DNP9khfUp7ZUWcpFC7XEdvbyjgzLMaWCoutUfou77H1nBfCGN4kzOnluCWr+J9Ix6tns37Pvwn+F/gC+8E/wDBMv8AZV1O61rwPb6+uo6/rV0zI3iPW5ViS8vzCGZLe3WG3WO2gQkJDGru8srNK2X8QfiHYeKPFvjDwp4s0ey8QeEvEb3NnqmiXwZrO9s2lDbHKurxujIskE8bLNBIqyxsrquPSv8AglPrPhX4e/H9PjN8QrK9u9O0iXS9Jhe0VS8eo+J9StNDszIJHQeWJL5mlOSyxoxVWYBT8wa5Ismv391Gio7zyttX7v3zWfA98XCrjMRrKT09FpY++8eqOFyfFYThzLY2p4eF5d227tvu3a5/Pb+37/wT78Qfsmaha/E/4dy3XiD4U+I7k2+l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cM22s+zvLjVLdtYumYvfvJePu9bhzIc/991oL8vStox0PMrSvO7CimpuanU+VGA7+D8aj3fNtpqsxbaacm5aGBJ+8Rlk2tIF/gXktn2p37V3xsk/YV8f6h8AJfCNjrnxBttN067utQ1c+fp2mHV7H7ZEttaRti5nhjubWUTXDmASrJE9pKgWRvdP2ctK8N3nxQt/Efjqwu9S8M+Ere88Va/DZRJLcNpOgwSX1yqLI8aM8iQeWAzopZhlgOR/Pl8RviB4t+LPxC174qeP7oX2veJtRutV1K5WKOATXd7K00ziOJUjTdI7Haiqi5woAwK8bOMZOFqcXZ/ofVZTg4uHPNGp8Xfi78SPjz8SNW+Lvxd1aXW/EOtyia7u5gq52qEjRI0CxxQxRqscMMarFDEqxxqqKqjziiivmT3gooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//R/wA/+iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA3/CvirxR4E8Uab438EaldaNrWjXUN9p+oWMz291aXVu4kimhljKvHJG6hkdSGVgCCCK/rt+NX/Byp8Gv29P2ebv4JftRfBfQfD3iDSLXT/FWlatfIfE+gX3ivw1NDfx2kukvDBPaWesCGfTXkW8neGG8McrPE0sg/jzoqlJrYyqUYTalJarY/1VP2svCeoeGvjx4n+2RLHHJcLPF/tJIAQa+Wbi1EwVW/i+Wvn/8A4Jc/t9+LP+CpP7EFwnj/AEu6HxG+ANto+ieItZeZ7pNf069jmjsr6SSaR5ze/wCiSC93lldz5yP+9MMP0VdeZDITJ8pr+h+FcwhjcvhJbpWfqj+JPEPIamV5zUhLZvmXo3f8Nj9NPg/4sbxx8PbDVrp991Enk3Df9NIeCfxHNenKrba+Gf2VfFi6f4k1Hwbdv8l/Etxbq38M0fDgfUf+g19zRnc2FO6vHx9D2VWUVsfUZLivrOFjNvXZnE/EbwqvjjwDqXhWP/XTRboP9meP54/1G2vylhkkk+8NhH3l/unuK/ZaPcpVu/8ADX5l/HrwtH4N+J19awriHUMX8H0uCd4/CQOP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CiiigAooooAKKKKACiiigAooooAKKKKACiiigD//1f8AP/ooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP7pv+CE/7dfgP9sj9jXT/wDgn78V9b2/GT4WC6fwYl2ZDLrnhlF877JFPLNJ51zp+ZVW2VYhHYpD5SOsM7R/tDY/ECSb9mTVvD+oTbb+wuIdLRW4dreaTzR+SJJH/uqK/wAxL4E/Hb4vfsyfF7QPj18BdfuvDHi/wxdC803UrMgSRSAFWBVgySRyIzRyxSK0csbNHIrIzKf9Iux+L3wk/bQ/Z78Nfty/s6XFsfD/AI3ih/4SDSbS4Nw/h/xFHGrXenTs0cL74nkO1mijEyFZkBilRj+kcGZwp8uX13s7x+W6+Z+EeKvDM6SnnOEW6tNfgn+SPLmx977tOT7tNb++KN3yk1+0H8sok+VY2xX5Bf8ABV7QXtbX4d+L15LSarp//faQSj/0W1frrMzK1fmZ/wAFWdLNz8E/CWvrx9k8SmI/9t7Kf/4ivnOKI3wEn6fmj7vw/qOGd0Lea+9M/Fpfl/2qKN275qFX+Fa/MWf00TKp/hbFG9ab/uUeZ/n/ACKc46nQQzTLDt5UZbb83Ssr9sH4yT/s+/szwfsyeDJoV134wWttrfjPzIIpprbRLW5W40axRzK728lxLEdRuFaCKR4DYlJGieRX9D8NWPhuPTdb+IXj1ZX8N+CdOk13UhFvUzJEyQW9t5kcM5hN5eTQWiTNE6RGcO42KxH46/Hf4z+MP2h/jB4h+NPjvYmo+ILtrg28DStbWcIASC0thPJLIlrawqlvbRtI3lwxogOFFeJnOL5Y+wg99z3ckw1715fI8looor5o+jCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD/9b/AD/6KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAr9c/8AgjT/AMFD4f2BP2o/K+I7ef8ACz4kR2/h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yKPU+R/+CpN/YfBTwX8Of2J7MTQeJNHtf+Ev8cBJpkhfVNchifTrWW3eNI5JLHT9skdwryjbfyRjy2SQN+N1dv8AEz4jeMvjF8SPEHxc+It2NQ8Q+KdSu9X1S6EUcAnvL6Vpp5BHCqRpvkdm2oioucKAMCuIr4qrVdSbnLdn3tGkqcFBdAooorM0CiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//9H/AD/6KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACt7wt4p8T+BvE+neNfBOo3Wj6zo91DfWF/YzPb3VrdW7iSKaGWMq8ckbqGR1IZWAIIIrBooA/e/4i/tIfCX9s/x94i+PvwV8JDwRd6nqJ+1+Dhqkms3cbSCPZcxzNa2u+G5mcwqqiVonQLIV86EP80/8FI/jfd6EmlfsA+E1+z6N8L9QuJ/EcySXCPqni2aOOK9+0QuI49ulMsmn2pCORi4lSZ47lVT85fh18V/il8INXm8QfCbxLqvhe/uYkgludIvJrKaSJJo7hUZ4WRiqzQxSgE4EkaMPmUEcBXZWx1SpTVOT2OShgaNKTlTVgooorjOsKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA/9L/AD/6KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP//T/wA/+iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD//1P8AP/ooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//9X/AD/6KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP//W/wA/+iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD//2Q=="
           style="width:110px;height:110px;object-fit:contain;margin-bottom:12px;filter:drop-shadow(0 8px 24px rgba(26,86,219,.25));" />
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

    # ── CARTE FOLIUM — Points simples ────────────────────────────
    df_m = df.dropna(subset=["latitude","longitude"]).copy()

    DESERT_STYLE = {
        "médical":    ("#DC2626", "🔴 Médical"),
        "commercial": ("#1D4ED8", "🔵 Commercial"),
        "mobilité":   ("#7C3AED", "🟣 Mobilité"),
        "scolaire":   ("#059669", "🎓 Scolaire"),
    }

    if not _FOLIUM_OK:
        st.error("⚠️ Folium non installé : `pip install folium streamlit-folium`")
    else:
        m_desert = folium.Map(
            location=[df_m["latitude"].mean(), df_m["longitude"].mean()],
            zoom_start=11, tiles=None, prefer_canvas=True,
        )
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
            attr="CartoDB Voyager",
        ).add_to(m_desert)

        count = 0
        for _, row in df_m.iterrows():
            if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
                continue
            deserts = []
            if aff_med and row.get("cat_desert_medical","") in niveaux_f:
                deserts.append(("médical", float(row.get("score_desert_medical",0)), row.get("cat_desert_medical","")))
            if aff_com and row.get("cat_desert_commercial","") in niveaux_f:
                deserts.append(("commercial", float(row.get("score_desert_commercial",0)), row.get("cat_desert_commercial","")))
            if aff_mob and row.get("cat_desert_mobilite","") in niveaux_f:
                deserts.append(("mobilité", float(row.get("score_desert_mobilite",0)), row.get("cat_desert_mobilite","")))
            if aff_sco and has_bpe_sco and row.get("cat_desert_primaire","") in niveaux_f:
                deserts.append(("scolaire", float(row.get("score_desert_primaire",0)), row.get("cat_desert_primaire","")))
            if not deserts:
                continue
            deserts.sort(key=lambda x: x[1], reverse=True)
            dtype, dscore, _ = deserts[0]
            color_hex, _ = DESERT_STYLE[dtype]
            radius = int(5 + dscore * 5)
            tooltip_lines = "<br>".join(
                f"{DESERT_STYLE[d][1]} : {niv}" for d, _, niv in deserts
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
                color=color_hex, fill=True, fill_color=color_hex,
                fill_opacity=0.80, weight=1.5,
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

    tab_comp, = st.tabs(["📈 Comparaison"])

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
            lat_mid = (lat_A + lat_B) / 2; lon_mid = (lon_A + lon_B) / 2
            dist = ((lat_A - lat_B)**2 + (lon_A - lon_B)**2)**0.5
            zoom_lvl = max(8, min(12, int(12 - dist * 18)))
            m_comp = folium.Map(location=[lat_mid, lon_mid], zoom_start=zoom_lvl, tiles=None, prefer_canvas=True)
            folium.TileLayer(
                tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
                attr="CartoDB Voyager").add_to(m_comp)
            for lat, lon, ville_n, dept_n, score_v, color_hex in [
                (lat_A, lon_A, A, str(rA.get("dept_nom","")), f"{float(rA.get('score_attractivite',0)):.0%}", "#1A56DB"),
                (lat_B, lon_B, B, str(rB.get("dept_nom","")), f"{float(rB.get('score_attractivite',0)):.0%}", "#6D28D9"),
            ]:
                pin_html = (
                    f"<div style='display:flex;flex-direction:column;align-items:center;'>"
                    f"<div style='background:{color_hex};color:#fff;padding:5px 10px;"
                    f"border-radius:20px;font-size:12px;font-weight:700;"
                    f"white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.25);"
                    f"font-family:Plus Jakarta Sans,sans-serif;'>📍 {ville_n}</div>"
                    f"<div style='width:2px;height:8px;background:{color_hex};'></div>"
                    f"<div style='width:8px;height:8px;border-radius:50%;background:{color_hex};'></div>"
                    f"</div>"
                )
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.DivIcon(html=pin_html,
                        icon_size=(max(120, len(ville_n)*10), 50),
                        icon_anchor=(max(60, len(ville_n)*5), 50)),
                    popup=folium.Popup(
                        f"<b style='font-size:14px;'>{ville_n}</b><br>"
                        f"<span style='color:#64748B;'>{dept_n}</span><br>"
                        f"<span style='color:{color_hex};font-weight:700;'>⭐ {score_v}</span>",
                        max_width=200),
                    tooltip=folium.Tooltip(f"<b>{ville_n}</b> · {dept_n} · {score_v}",
                        style="font-family:sans-serif;font-size:12px;"),
                ).add_to(m_comp)
            st_folium(m_comp, width=None, height=420, use_container_width=True, returned_objects=[])
        except Exception:
            st.info(f"📍 **{A}** · {rA.get('dept_nom','')} &nbsp;|&nbsp; 📍 **{B}** · {rB.get('dept_nom','')} — coordonnées non disponibles.")
    else:
        st.info(f"📍 **{A}** · {rA.get('dept_nom','')} &nbsp;|&nbsp; 📍 **{B}** · {rB.get('dept_nom','')} — installez folium.")

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
