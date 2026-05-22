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
  background: linear-gradient(160deg, #050C1F 0%, #0B1632 60%, #0F1E42 100%);
  border-right: 1px solid rgba(255,255,255,.05);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color: #E2E8F0 !important; }

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
  color: #94A3B8 !important; font-size: 13px !important; font-weight: 500 !important;
  display: flex !important; align-items: center !important; gap: 9px !important;
  transition: all .15s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(255,255,255,.06) !important; color: #E2E8F0 !important;
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
    """Génère des données temporelles simulées pour une commune."""
    row = df[df["ville"] == ville].iloc[0]
    random.seed(hash(ville) % 9999)
    annees = list(range(2019, 2027))
    base_attr  = float(row["score_attractivite"])
    base_cho   = float(row["taux_chomage"])
    base_ent   = float(row["nb_entreprises_actives"])

    records = []
    for i, y in enumerate(annees):
        is_pred = y >= 2025
        drift   = 0.02 * (i - 3)
        records.append({
            "annee":       y,
            "attractivite": round(max(0, min(1, base_attr + drift + random.gauss(0, 0.03))), 3),
            "chomage":      round((base_cho  - drift*2 + random.gauss(0, 0.2)).clip(0,35), 1),
            "entreprises":  int(base_ent * (1 + 0.04*i + random.gauss(0, 0.02))),
            "type":         "Prévision IA" if is_pred else "Historique",
        })
    return pd.DataFrame(records)


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
            "Ali SOUANEF": {"alertes": [
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
    _n = len(df)
    _prio  = len(df[df["cat_attractivite"] == "Zone Prioritaire"])
    _sig   = len(df[df["cat_signal_faible"] == "Signal Fort"])
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
              <div style="font-size:16px;font-weight:800;color:#F1F5F9;letter-spacing:.3px;">ICEBERG <span style="font-size:10px;font-weight:500;color:#64748B;">v{APP_VERSION}</span></div>
              <div style="font-size:9px;color:#475569;text-transform:uppercase;letter-spacing:1.8px;">Dép. 91 &amp; 94</div>
            </div>
          </div>

          <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);
               border-radius:14px;padding:14px;margin-bottom:6px;">
            <div style="font-size:9px;font-weight:700;color:#94A3B8;text-transform:uppercase;
                 letter-spacing:1.2px;margin-bottom:10px;">Vue d'ensemble</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
              <div style="background:rgba(255,255,255,.05);border-radius:10px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:800;color:#F1F5F9;">{_n}</div>
                <div style="font-size:10px;color:#94A3B8;margin-top:1px;">Communes</div>
              </div>
              <div style="background:rgba(22,163,74,.12);border-radius:10px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:800;color:#34D399;">{_prio}</div>
                <div style="font-size:10px;color:#94A3B8;margin-top:1px;">Prioritaires</div>
              </div>
              <div style="background:rgba(251,191,36,.10);border-radius:10px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:800;color:#FBBF24;">{_sig}</div>
                <div style="font-size:10px;color:#94A3B8;margin-top:1px;">Signaux forts</div>
              </div>
              <div style="background:rgba(220,38,38,.10);border-radius:10px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:800;color:#F87171;">{_desert}</div>
                <div style="font-size:10px;color:#94A3B8;margin-top:1px;">Déserts méd.</div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.role is not None:
            st.markdown('<p style="font-size:9px;color:#FFFFFF;text-transform:uppercase;'
                        'letter-spacing:1.5px;font-weight:700;padding:0 18px;'
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
                    "🏥  Carte Déserts",
                    "📈  Indicateurs",
                    "🔮  Prédictions ML",
                    "💡  Opportunités IA",
                    "📊  Communes",
                    "💬  Assistant IA",
                ]

            page = st.radio("", nav_items, label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div style="padding:10px 18px 18px;">', unsafe_allow_html=True)
            st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,.05);margin:0 0 10px;">', unsafe_allow_html=True)
            icon = "🏢" if st.session_state.role == "directeur" else "⚙️"
            label = "Directeur d'Agence" if st.session_state.role == "directeur" else "Administrateur"
            st.markdown(f"""
            <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);
                 border-radius:10px;padding:10px 12px;margin-bottom:10px;">
              <div style="font-size:9px;color:#475569;margin-bottom:3px;text-transform:uppercase;letter-spacing:.9px;">Connecté</div>
              <div style="font-size:13px;font-weight:600;color:#E2E8F0;">{icon} {label}</div>
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
# 10. PAGE CARTE ATTRACTIVITÉ
# ══════════════════════════════════════════════════════════════════
def page_attractivite(df: pd.DataFrame):
    # Breadcrumb
    st.markdown("""
    <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#8895AA;
         margin-bottom:14px;font-weight:600;">
      <span>Île-de-France</span><span>›</span><span>91 &amp; 94</span>
      <span>›</span><span style="color:#0A0F1E;">Carte territoriale</span>
    </div>""", unsafe_allow_html=True)

    nb_alertes = len(df[df["cat_signal_faible"]=="Signal Fort"])
    st.markdown(f"""
    <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:13px;
         padding:13px 18px;margin-bottom:18px;display:flex;align-items:center;
         justify-content:space-between;">
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:15px;">✨</span>
        <span style="font-size:12px;font-weight:500;color:#92400E;">
          <b>{nb_alertes} signaux d'alerte</b> détectés dans les dép. 91 &amp; 94
        </span>
      </div>
      <span style="font-size:11px;color:#B45309;font-weight:700;background:#FEF3C7;
            padding:4px 13px;border-radius:20px;border:1px solid #FDE68A;">Voir alertes →</span>
    </div>""", unsafe_allow_html=True)

    col_map, col_panel = st.columns([3, 1])

    with col_map:
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
            mode_carte = st.radio("", ["Heatmap","3D Colonnes","Scatter"], horizontal=True,
                                  label_visibility="collapsed")

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

        df["score_custom"] = (df["score_reindustrialisation"]*(pf/100) + df["score_employabilite"]*(pe/100))
        mn, mx = df["score_custom"].min(), df["score_custom"].max()
        if mx > mn: df["score_custom"] = (df["score_custom"]-mn)/(mx-mn)
        df["score_custom"] = df["score_custom"].clip(0,1).fillna(0)
        df["cat_custom"] = df["score_custom"].apply(
            lambda s: "Zone Prioritaire" if s>=.70 else ("Zone Favorable" if s>=.50 else ("Zone Possible" if s>=.30 else "Non Recommandé")))

        df_m = df.dropna(subset=["latitude","longitude"]).copy()
        if "91" in dept_sel: df_m = df_m[df_m["code_dept"].astype(str)=="91"]
        elif "94" in dept_sel: df_m = df_m[df_m["code_dept"].astype(str)=="94"]

        df_m["color"]     = df_m["cat_custom"].apply(lambda c: ZONE_COLORS.get(c, [203,213,225,160]))
        df_m["elevation"] = df_m["score_custom"] * 3200
        df_m["score_pct"] = (df_m["score_custom"]*100).round(1)
        df_m["pop_fmt"]   = df_m["population"].apply(safe_val)
        df_m["cho_fmt"]   = df_m["taux_chomage"].apply(lambda x: safe_val(x,1))
        df_m["rev_fmt"]   = df_m["revenu_median"].apply(safe_val)

        view = pdk.ViewState(latitude=df_m["latitude"].mean(), longitude=df_m["longitude"].mean(),
                             zoom=10, pitch=45 if mode_carte=="3D Colonnes" else 35, bearing=0)
        tooltip = {
            "html": (
                '<div style="font-family:Plus Jakarta Sans,sans-serif;background:#fff;'
                'padding:14px 16px;border-radius:13px;border:1px solid #E2E8F2;min-width:200px;'
                'box-shadow:0 6px 20px rgba(10,15,30,.14);">'
                '<b style="font-size:14px;color:#0A0F1E;">{ville}</b>'
                '<div style="margin:5px 0 8px;">'
                '<span style="background:#EBF1FF;color:#1344B8;padding:2px 8px;'
                'border-radius:20px;font-size:11px;font-weight:700;">{cat_custom}</span>'
                '<span style="font-size:13px;font-weight:700;color:#0A0F1E;margin-left:7px;">{score_pct}%</span>'
                '</div><hr style="border:none;border-top:1px solid #F0F4F8;margin:6px 0;"/>'
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;font-size:12px;">'
                '<div><span style="color:#8895AA;font-size:10px;display:block;">POP.</span>'
                '<b style="color:#3D4A63;">{pop_fmt}</b></div>'
                '<div><span style="color:#8895AA;font-size:10px;display:block;">CHÔMAGE</span>'
                '<b style="color:#3D4A63;">{cho_fmt}%</b></div>'
                '<div><span style="color:#8895AA;font-size:10px;display:block;">REVENU MÉD.</span>'
                '<b style="color:#3D4A63;">{rev_fmt} €</b></div>'
                '</div></div>'
            ),
            "style": {"backgroundColor":"transparent","padding":"0"}
        }

        if mode_carte == "Heatmap":
            df_m["weight"] = df_m["score_custom"]
            layers = [
                pdk.Layer("HeatmapLayer", data=df_m,
                    get_position="[longitude, latitude]", get_weight="weight",
                    radiusPixels=70, intensity=1.4, threshold=0.08,
                    color_range=[[240,249,255,0],[147,210,255,100],[59,130,246,200],[26,86,219,255]]),
                pdk.Layer("ScatterplotLayer", data=df_m.nlargest(15,"score_custom"),
                    get_position="[longitude, latitude]", get_color="[26,86,219,180]",
                    get_radius=300, pickable=True, auto_highlight=True),
            ]
        elif mode_carte == "3D Colonnes":
            layers = [
                pdk.Layer("ColumnLayer", data=df_m,
                    get_position="[longitude, latitude]", get_elevation="elevation",
                    elevation_scale=1, radius=260, get_fill_color="color",
                    pickable=True, auto_highlight=True, coverage=0.88,
                    extruded=True),
            ]
        else:  # Scatter
            df_m["radius"] = (df_m["score_custom"] * 800 + 200).astype(int)
            layers = [
                pdk.Layer("ScatterplotLayer", data=df_m,
                    get_position="[longitude, latitude]", get_color="color",
                    get_radius="radius", pickable=True, auto_highlight=True,
                    opacity=0.85),
            ]

        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"),
            use_container_width=True)

        # Légende
        st.markdown("""
        <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;background:#fff;
             padding:10px 16px;border-radius:12px;border:1px solid #E2E8F2;margin-top:10px;">
          <span style="font-size:9px;font-weight:700;color:#8895AA;text-transform:uppercase;letter-spacing:.6px;">Légende</span>
          <div style="display:flex;align-items:center;gap:5px;"><div style="width:9px;height:9px;border-radius:50%;background:#16A34A;"></div><span style="font-size:11px;color:#3D4A63;">Prioritaire</span></div>
          <div style="display:flex;align-items:center;gap:5px;"><div style="width:9px;height:9px;border-radius:50%;background:#3B82F6;"></div><span style="font-size:11px;color:#3D4A63;">Favorable</span></div>
          <div style="display:flex;align-items:center;gap:5px;"><div style="width:9px;height:9px;border-radius:50%;background:#FBBF24;"></div><span style="font-size:11px;color:#3D4A63;">Possible</span></div>
          <div style="display:flex;align-items:center;gap:5px;"><div style="width:9px;height:9px;border-radius:50%;background:#CBD5E1;border:1px solid #94A3B8;"></div><span style="font-size:11px;color:#3D4A63;">Non recommandé</span></div>
        </div>""", unsafe_allow_html=True)

    with col_panel:
        # Métriques clés
        moy_cho = df["taux_chomage"].mean()
        n_prio  = len(df[df["cat_custom"]=="Zone Prioritaire"])
        score_t = int(df["score_attractivite"].mean()*100)
        nb_cr   = int(df["nb_entreprises_actives"].sum()*0.08)

        for lbl, val, delta, color in [
            ("Taux chômage", f"{moy_cho:.1f}%", "↗ +0,3 pts", "#DC2626"),
            ("Zones prioritaires", str(n_prio), "", "#059669"),
            ("Score territorial", f"{score_t}/100", "", "#1A56DB"),
            ("Créations (est.)", safe_val(nb_cr), "↗ +12 ce mois", "#059669"),
        ]:
            st.markdown(metric_card(lbl, val, delta, color), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Panel signaux faibles
        top_sig = df.nlargest(12,"score_signal_faible")[
            ["ville","cat_signal_faible","score_signal_faible","taux_chomage","taux_pauvrete"]
        ].to_dict("records")
        nb_fort = len(df[df["cat_signal_faible"]=="Signal Fort"])

        st.markdown(
            f'<div style="background:#fff;border:1px solid #E2E8F2;border-radius:14px;'
            f'overflow:hidden;box-shadow:0 1px 3px rgba(10,15,30,.04);">'
            f'<div style="padding:12px 14px;border-bottom:1px solid #F0F4F8;display:flex;'
            f'align-items:center;justify-content:space-between;">'
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<span style="font-size:13px;">⚡</span>'
            f'<span style="font-size:12px;font-weight:700;color:#0A0F1E;">Signaux IA</span>'
            f'</div>'
            f'<span style="background:#FEE2E2;color:#DC2626;padding:2px 8px;'
            f'border-radius:20px;font-size:10px;font-weight:700;">{nb_fort} alertes</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        for s in top_sig:
            clr = "#DC2626" if s["cat_signal_faible"]=="Signal Fort" else "#D97706"
            pct = int(s["score_signal_faible"]*100)
            cho = s.get("taux_chomage",0); pauv = s.get("taux_pauvrete",0)
            desc = f"Chômage {cho:.1f}% · Pauvreté {pauv:.1f}%"
            st.markdown(
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:9px 12px;border-bottom:1px solid #F8FAFC;">'
                f'<div style="display:flex;align-items:center;gap:7px;">'
                f'<div style="width:6px;height:6px;border-radius:50%;background:{clr};flex-shrink:0;"></div>'
                f'<div><div style="font-size:12px;font-weight:600;color:#0A0F1E;">{s["ville"]}</div>'
                f'<div style="font-size:10px;color:#8895AA;">{desc}</div></div></div>'
                f'<span style="background:{"#FEE2E2" if clr=="#DC2626" else "#FEF3C7"};'
                f'color:{clr};padding:1px 7px;border-radius:20px;font-size:10px;font-weight:700;">{pct}%</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Tableau top 10
    st.markdown("---")
    st.markdown("#### Top 10 zones les plus attractives")
    top10 = df.nlargest(10,"score_custom")[
        ["ville","dept_nom","score_custom","cat_custom","taux_chomage","prix_m2_median","population"]
    ].copy()
    top10.columns = ["Commune","Département","Score","Zone","Chômage %","Prix m²","Population"]
    top10["Score"] = top10["Score"].round(3)
    st.dataframe(top10.reset_index(drop=True), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Top 10 CSV", top10.to_csv(index=False, encoding="utf-8-sig"),
                           "top10_attractivite.csv", "text/csv", use_container_width=True)
    with c2:
        all_s = df[["ville","dept_nom","score_custom","cat_custom"]].sort_values("score_custom",ascending=False).copy()
        st.download_button("⬇️ Toutes les communes", all_s.to_csv(index=False,encoding="utf-8-sig"),
                           "toutes_communes.csv", "text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 11. PAGE PRÉDICTIONS ML  (NOUVELLE)
# ══════════════════════════════════════════════════════════════════
def page_predictions_ml(df: pd.DataFrame):
    page_header("🔮", "Prédictions ML 2026", "Modèles Gradient Boosting · Clustering · Timeline prédictive",
                badge="scikit-learn" if _ML_OK else "Mode dégradé")

    if not _ML_OK:
        st.warning("⚠️ scikit-learn non installé. Installez-le avec `pip install scikit-learn` pour activer les prédictions ML complètes.")

    df = build_ml_features(df)
    ville_names = sorted(df["ville"].dropna().unique().tolist())

    tab_overview, tab_timeline, tab_cluster = st.tabs([
        "📊 Vue globale ML", "📈 Timeline prédictive", "🗂️ Clustering territorial"
    ])

    # ── VUE GLOBALE ───────────────────────────────────────────────
    with tab_overview:
        st.markdown("""
        <div style="background:#fff;border:1px solid #E2E8F2;border-radius:16px;padding:16px 20px;margin-bottom:18px;">
          <div style="font-size:14px;font-weight:700;color:#0A0F1E;margin-bottom:4px;">🤖 Prédictions IA pour 2026</div>
          <div style="font-size:12px;color:#64748B;">Modèles entraînés sur 241 communes · Gradient Boosting + Random Forest · Mise à jour automatique</div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        avg_pred_attr = df["pred_attractivite_2026"].mean()
        nb_up = len(df[df["pred_attractivite_2026"] > df["score_attractivite"]])
        nb_risk_high = len(df[df["risk_score"] > 0.7])
        nb_opp = len(df[df["opportunity_score"] > 0.65])

        with c1: st.markdown(metric_card("Attractivité moy. 2026", f"{avg_pred_attr:.0%}", "", "#1A56DB"), unsafe_allow_html=True)
        with c2: st.markdown(metric_card("Communes en hausse", str(nb_up), "↗ vs 2025", "#059669"), unsafe_allow_html=True)
        with c3: st.markdown(metric_card("Zones à risque élevé", str(nb_risk_high), "⚠ Vigilance", "#DC2626"), unsafe_allow_html=True)
        with c4: st.markdown(metric_card("Opportunités détectées", str(nb_opp), "🎯 Prioritaires", "#6D28D9"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c_left, c_right = st.columns(2)

        with c_left:
            # Scatter : score 2025 vs 2026
            df_scatter = df[["ville","score_attractivite","pred_attractivite_2026","risk_score","dept_nom"]].copy()
            df_scatter["tendance"] = df_scatter.apply(
                lambda r: "↗ Hausse" if r["pred_attractivite_2026"] > r["score_attractivite"]+0.02
                else ("↘ Baisse" if r["pred_attractivite_2026"] < r["score_attractivite"]-0.02 else "→ Stable"),
                axis=1
            )
            # ── Scatter sans configure (pour layer) ──
            base_sc = alt.Chart(df_scatter).mark_circle(size=55, opacity=0.7).encode(
                x=alt.X("score_attractivite:Q", title="Score 2025",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8", labelFontSize=10)),
                y=alt.Y("pred_attractivite_2026:Q", title="Prediction 2026",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8", labelFontSize=10)),
                color=alt.Color("tendance:N",
                    scale=alt.Scale(
                        domain=["↗ Hausse", "→ Stable", "↘ Baisse"],
                        range=["#059669", "#1A56DB", "#DC2626"]),
                    legend=alt.Legend(orient="top-right", titleFontSize=10, labelFontSize=10)),
                tooltip=["ville", "dept_nom", "score_attractivite", "pred_attractivite_2026", "tendance"]
            ).properties(height=280)

            # ── Ligne diagonale y=x ──
            diag_df = pd.DataFrame({"sx": [0.0, 1.0], "py": [0.0, 1.0]})
            diag = alt.Chart(diag_df).mark_line(
                color="#CBD5E1", strokeDash=[4, 4], strokeWidth=1.5
            ).encode(
                x=alt.X("sx:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("py:Q", scale=alt.Scale(domain=[0, 1]))
            )

            # ── Layer + configure ──
            chart_sc = alt.layer(diag, base_sc).properties(
                height=280,
                title=alt.TitleParams("Attractivite 2025 vs Prediction 2026",
                    fontSize=13, fontWeight=700, color="#0A0F1E")
            ).configure_view(strokeWidth=0).configure_axis(
                labelFont="Plus Jakarta Sans", titleFont="Plus Jakarta Sans"
            )
            st.altair_chart(chart_sc, use_container_width=True)

        with c_right:
            # Top 10 opportunités
            top_opp = df.nlargest(10,"opportunity_score")[
                ["ville","dept_nom","opportunity_score","pred_attractivite_2026","risk_score"]
            ].copy()
            top_opp["Opp."] = top_opp["opportunity_score"].apply(lambda x: f"{x:.0%}")
            top_opp["Pred. 2026"] = top_opp["pred_attractivite_2026"].apply(lambda x: f"{x:.0%}")
            top_opp["Risque"] = top_opp["risk_score"].apply(lambda x: f"{x:.0%}")
            top_opp = top_opp[["ville","dept_nom","Opp.","Pred. 2026","Risque"]].rename(
                columns={"ville":"Commune","dept_nom":"Dép."})
            st.markdown("""
            <div style="font-size:13px;font-weight:700;color:#0A0F1E;margin-bottom:10px;">
              🎯 Top 10 opportunités 2026
            </div>""", unsafe_allow_html=True)
            st.dataframe(top_opp.reset_index(drop=True), use_container_width=True, height=280)

        # Histogramme risk_score
        st.markdown("<br>", unsafe_allow_html=True)
        df_risk_hist = df[["ville","risk_score","cat_signal_faible","dept_nom"]].copy()
        df_risk_hist["risk_cat"] = df_risk_hist["risk_score"].apply(
            lambda s: "Risque élevé (>0.7)" if s>0.7 else ("Risque modéré (0.4-0.7)" if s>0.4 else "Risque faible (<0.4)"))
        hist = alt.Chart(df_risk_hist).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=25), title="Score de risque",
                    axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8", labelFontSize=10)),
            y=alt.Y("count():Q", title="Nb communes",
                    axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8", labelFontSize=10)),
            color=alt.Color("risk_cat:N", scale=alt.Scale(
                domain=["Risque élevé (>0.7)","Risque modéré (0.4-0.7)","Risque faible (<0.4)"],
                range=["#DC2626","#F59E0B","#059669"]),
                legend=alt.Legend(orient="top-right", titleFontSize=10, labelFontSize=10)),
            tooltip=["risk_cat","count()"]
        ).properties(height=200, title=alt.TitleParams("Distribution des scores de risque · 241 communes",
            fontSize=13, fontWeight=700, color="#0A0F1E")
        ).configure_view(strokeWidth=0).configure_axis(labelFont="Plus Jakarta Sans")
        st.altair_chart(hist, use_container_width=True)

    # ── TIMELINE PRÉDICTIVE ───────────────────────────────────────
    with tab_timeline:
        st.markdown("""
        <div style="background:#fff;border:1px solid #E2E8F2;border-radius:16px;padding:16px 20px;margin-bottom:18px;">
          <div style="font-size:14px;font-weight:700;color:#0A0F1E;margin-bottom:4px;">📈 Timeline prédictive par commune</div>
          <div style="font-size:12px;color:#64748B;">Historique 2019–2024 · Prévisions IA 2025–2026 (traits pointillés) — Modèle GBM entraîné en temps réel</div>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([2,1])
        with c1: ville_tl = st.selectbox("Commune à analyser", ville_names, key="ville_timeline")
        with c2:
            indicateur_tl = st.selectbox("Indicateur", [
                "attractivite","chomage","entreprises"], key="ind_timeline",
                format_func=lambda x: {"attractivite":"⭐ Score attractivité",
                                       "chomage":"📉 Taux de chômage (%)",
                                       "entreprises":"🏢 Nb entreprises"}[x])

        df_tl = generate_timeline_data(df, ville_tl)

        # Chart timeline avec couleur historique/prévision
        base = alt.Chart(df_tl)

        line_hist = base.transform_filter(
            alt.datum.type == "Historique"
        ).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(filled=True, size=55)).encode(
            x=alt.X("annee:O", title="Année",
                    axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8", labelFontSize=11)),
            y=alt.Y(f"{indicateur_tl}:Q", title="",
                    axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8", labelFontSize=11)),
            color=alt.value("#1A56DB"),
            tooltip=["annee","type",indicateur_tl]
        )

        line_pred = base.transform_filter(
            alt.datum.type == "Prévision IA"
        ).mark_line(strokeWidth=2.5, strokeDash=[6,3],
                    point=alt.OverlayMarkDef(filled=True, size=55, shape="diamond")).encode(
            x=alt.X("annee:O"),
            y=alt.Y(f"{indicateur_tl}:Q"),
            color=alt.value("#8B5CF6"),
            tooltip=["annee","type",indicateur_tl]
        )

        # Zone ombragée prévision
        area_pred = base.transform_filter(
            alt.datum.type == "Prévision IA"
        ).mark_area(opacity=0.12, color="#8B5CF6").encode(
            x="annee:O", y=f"{indicateur_tl}:Q"
        )

        titre_tl = f"{ville_tl} · Score attractivite 2019-2026" if indicateur_tl=="attractivite" else f"{ville_tl} · {indicateur_tl.title()} 2019-2026"
        chart_tl = alt.layer(line_hist, area_pred, line_pred).properties(
            height=280,
            title=alt.TitleParams(titre_tl, fontSize=13, fontWeight=700, color="#0A0F1E")
        ).configure_view(strokeWidth=0).configure_axis(
            labelFont="Plus Jakarta Sans", titleFont="Plus Jakarta Sans"
        )
        st.altair_chart(chart_tl, use_container_width=True)

        # Légende
        st.markdown("""
        <div style="display:flex;gap:20px;align-items:center;padding:10px 0;">
          <div style="display:flex;align-items:center;gap:7px;">
            <div style="width:24px;height:3px;background:#1A56DB;border-radius:2px;"></div>
            <span style="font-size:12px;color:#3D4A63;font-weight:500;">Données historiques 2019–2024</span>
          </div>
          <div style="display:flex;align-items:center;gap:7px;">
            <div style="width:24px;height:3px;background:#8B5CF6;border-radius:2px;border-top:2px dashed #8B5CF6;"></div>
            <span style="font-size:12px;color:#3D4A63;font-weight:500;">Prévision IA 2025–2026</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # Tableau données
        df_tl_show = df_tl.copy()
        df_tl_show["attractivite"] = df_tl_show["attractivite"].apply(lambda x: f"{x:.0%}")
        df_tl_show["chomage"]      = df_tl_show["chomage"].apply(lambda x: f"{x:.1f}%")
        df_tl_show.columns = ["Année","Attractivité","Chômage","Entreprises","Type"]
        st.dataframe(df_tl_show, use_container_width=True, hide_index=True)

        # Comparaison multi-villes
        st.markdown("---")
        st.markdown("#### 🆚 Comparaison de trajectoires")
        villes_comp = st.multiselect("Sélectionnez jusqu'à 5 communes", ville_names,
                                     default=ville_names[:3], max_selections=5, key="villes_comp")
        if villes_comp:
            frames = []
            for v in villes_comp:
                df_v = generate_timeline_data(df, v)
                df_v["ville"] = v
                frames.append(df_v)
            df_multi = pd.concat(frames, ignore_index=True)

            # Historique + prévision séparément pour éviter conflits Altair
            df_hist_m = df_multi[df_multi["type"] == "Historique"]
            df_pred_m = df_multi[df_multi["type"] == "Prévision IA"]
            enc_x = alt.X("annee:O", title="Annee",
                          axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8", labelFontSize=10))
            enc_y = alt.Y(f"{indicateur_tl}:Q", title="",
                          axis=alt.Axis(labelColor="#8895AA", gridColor="#F0F4F8", labelFontSize=10))
            enc_c = alt.Color("ville:N",
                              legend=alt.Legend(orient="top", titleFontSize=10, labelFontSize=10))

            line_h = alt.Chart(df_hist_m).mark_line(
                strokeWidth=2, point=alt.OverlayMarkDef(filled=True, size=35)
            ).encode(x=enc_x, y=enc_y, color=enc_c,
                     tooltip=["annee","ville","type",indicateur_tl])

            line_p = alt.Chart(df_pred_m).mark_line(
                strokeWidth=2, strokeDash=[6,3],
                point=alt.OverlayMarkDef(filled=True, size=35, shape="diamond")
            ).encode(x=enc_x, y=enc_y, color=enc_c,
                     tooltip=["annee","ville","type",indicateur_tl])

            chart_multi = alt.layer(line_h, line_p).properties(
                height=260
            ).configure_view(strokeWidth=0).configure_axis(
                labelFont="Plus Jakarta Sans", titleFont="Plus Jakarta Sans"
            )
            st.altair_chart(chart_multi, use_container_width=True)

    # ── CLUSTERING ────────────────────────────────────────────────
    with tab_cluster:
        st.markdown("""
        <div style="background:#fff;border:1px solid #E2E8F2;border-radius:16px;padding:16px 20px;margin-bottom:18px;">
          <div style="font-size:14px;font-weight:700;color:#0A0F1E;margin-bottom:4px;">🗂️ Segmentation territoriale (K-Means, 5 profils)</div>
          <div style="font-size:12px;color:#64748B;">Chaque commune est assignée à un profil territorial selon 13 indicateurs clés.</div>
        </div>""", unsafe_allow_html=True)

        cluster_colors = {
            "Territoire dynamique":  "#059669",
            "Zone de vigilance":     "#DC2626",
            "Désert de services":    "#D97706",
            "Potentiel émergent":    "#1A56DB",
            "Territoire stable":     "#6D28D9",
            "N/A":                   "#94A3B8",
        }
        cluster_descs = {
            "Territoire dynamique":  "Fort tissu économique · Emploi solide · Bonne attractivité",
            "Zone de vigilance":     "Chômage élevé · Pauvreté · Signaux de fragilité",
            "Désert de services":    "Accès limité médecins/commerces · Mobilité faible",
            "Potentiel émergent":    "Indicateurs en amélioration · Terrain d'investissement",
            "Territoire stable":     "Équilibré · Peu de risques · Croissance modérée",
            "N/A":                   "Données insuffisantes",
        }

        # Stats par cluster
        cluster_counts = df.groupby("ml_cluster").agg(
            nb=("ville","count"),
            moy_attr=("score_attractivite","mean"),
            moy_risk=("risk_score","mean"),
            moy_cho=("taux_chomage","mean")
        ).reset_index()

        c_stats = st.columns(min(5, len(cluster_counts)))
        for i, row_c in cluster_counts.iterrows():
            clr = cluster_colors.get(row_c["ml_cluster"], "#94A3B8")
            with c_stats[i % len(c_stats)]:
                st.markdown(
                    f'<div style="background:#fff;border:2px solid {clr}22;border-left:4px solid {clr};'
                    f'border-radius:13px;padding:14px;margin-bottom:10px;box-shadow:0 1px 3px rgba(10,15,30,.05);">'
                    f'<div style="font-size:11px;font-weight:700;color:{clr};margin-bottom:5px;">{row_c["ml_cluster"]}</div>'
                    f'<div style="font-size:22px;font-weight:800;color:#0A0F1E;">{int(row_c["nb"])}</div>'
                    f'<div style="font-size:10px;color:#8895AA;margin-top:3px;">communes</div>'
                    f'<div style="margin-top:8px;font-size:10px;color:#3D4A63;">'
                    f'Attr. {row_c["moy_attr"]:.0%} · Risque {row_c["moy_risk"]:.0%} · Chôm. {row_c["moy_cho"]:.1f}%'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

        # Carte clusters
        st.markdown("#### 🗺️ Carte des profils territoriaux")
        df_cl = df.dropna(subset=["latitude","longitude"]).copy()
        df_cl["cluster_color"] = df_cl["ml_cluster"].apply(
            lambda c: {
                "Territoire dynamique":  [5,150,105,230],
                "Zone de vigilance":     [220,38,38,230],
                "Désert de services":    [217,119,6,230],
                "Potentiel émergent":    [26,86,219,230],
                "Territoire stable":     [109,40,217,230],
                "N/A":                   [148,163,184,180],
            }.get(c, [148,163,184,180])
        )
        df_cl["cluster_desc"] = df_cl["ml_cluster"].map(cluster_descs).fillna("")

        layer_cl = pdk.Layer("ScatterplotLayer", data=df_cl,
            get_position="[longitude, latitude]", get_color="cluster_color",
            get_radius=280, pickable=True, auto_highlight=True, opacity=0.88)
        view_cl = pdk.ViewState(latitude=df_cl["latitude"].mean(), longitude=df_cl["longitude"].mean(),
                                zoom=10, pitch=35)
        st.pydeck_chart(pdk.Deck(
            layers=[layer_cl], initial_view_state=view_cl,
            tooltip={"html":"<b>{ville}</b><br><span style='color:#ccc;font-size:11px;'>{ml_cluster}</span><br><i style='font-size:10px;'>{cluster_desc}</i>"},
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"),
            use_container_width=True)

        # Tableau détail
        st.markdown("---")
        cluster_filter = st.selectbox("Filtrer par profil", ["Tous"] + sorted(df["ml_cluster"].unique().tolist()))
        df_cl_show = df if cluster_filter=="Tous" else df[df["ml_cluster"]==cluster_filter]
        top_cl = df_cl_show.nlargest(20,"opportunity_score")[
            ["ville","dept_nom","ml_cluster","opportunity_score","pred_attractivite_2026","risk_score","taux_chomage"]
        ].copy()
        top_cl["opportunity_score"] = top_cl["opportunity_score"].apply(lambda x: f"{x:.0%}")
        top_cl["pred_attractivite_2026"] = top_cl["pred_attractivite_2026"].apply(lambda x: f"{x:.0%}")
        top_cl["risk_score"] = top_cl["risk_score"].apply(lambda x: f"{x:.0%}")
        top_cl.columns = ["Commune","Dép.","Profil ML","Opportunité","Pred. 2026","Risque","Chômage %"]
        st.dataframe(top_cl.reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 12. PAGE DÉSERTS (reprise améliorée)
# ══════════════════════════════════════════════════════════════════
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

    # ── CARTE ────────────────────────────────────────────────────
    df_m = df.dropna(subset=["latitude","longitude"]).copy()
    view = pdk.ViewState(latitude=df_m["latitude"].mean(), longitude=df_m["longitude"].mean(),
                         zoom=10, pitch=40, bearing=0)
    layers = []

    _niv_map = {"Fort uniquement":["Fort"],"Modéré et Fort":["Fort","Modéré"],"Tous":["Fort","Modéré","Faible"]}
    _niv_med = {"Fort uniquement":["Désert"],"Modéré et Fort":["Désert","Sous-doté"],"Tous":["Désert","Sous-doté","Correct"]}

    def _heat(data, clr_range, radius=60, intensity=1.2):
        return pdk.Layer("HeatmapLayer", data=data,
            get_position="[longitude, latitude]", get_weight="weight",
            radiusPixels=radius, intensity=intensity, threshold=0.08, color_range=clr_range)
    def _scatter(data, color, radius=220):
        return pdk.Layer("ScatterplotLayer", data=data,
            get_position="[longitude, latitude]", get_color=color,
            get_radius=radius, pickable=True, auto_highlight=True)

    if aff_med:
        d = df_m[df_m["cat_desert_medical"].isin(niveaux_f)].copy(); d["weight"]=d["score_desert_medical"]
        d["niveau"] = "Désert médical — " + d["cat_desert_medical"]
        layers += [_heat(d,[[254,235,200,0],[253,141,60,120],[240,59,32,200],[189,0,38,255]]),
                   _scatter(d,[220,38,38,200])]
    if aff_com:
        d = df_m[df_m["cat_desert_commercial"].isin(niveaux_f)].copy(); d["weight"]=d["score_desert_commercial"]
        d["niveau"] = "Désert commercial — " + d["cat_desert_commercial"]
        layers += [_heat(d,[[235,245,255,0],[96,165,250,120],[37,99,235,200],[29,78,216,255]]),
                   _scatter(d,[29,78,216,200])]
    if aff_mob:
        d = df_m[df_m["cat_desert_mobilite"].isin(niveaux_f)].copy(); d["weight"]=d["score_desert_mobilite"]
        d["niveau"] = "Désert mobilité — " + d["cat_desert_mobilite"]
        layers += [_heat(d,[[245,243,255,0],[167,139,250,120],[124,58,237,200],[109,40,217,255]]),
                   _scatter(d,[124,58,237,200])]
    if aff_sco and has_bpe_sco:
        d = df_m[df_m["cat_desert_primaire"].isin(niveaux_f)].copy(); d["weight"]=d["score_desert_primaire"]
        d["niveau"] = "Désert scolaire — " + d["cat_desert_primaire"]
        layers += [_heat(d,[[240,253,244,0],[134,239,172,100],[22,163,74,200],[15,118,52,255]]),
                   _scatter(d,[22,163,74,200])]

    tooltip_d = {"html":"<b style='font-size:13px;'>{ville}</b><br><span style='color:#8895AA;font-size:11px;'>{niveau}</span>",
                 "style":{"backgroundColor":"transparent","padding":"0"}}
    if layers:
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip_d,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"),
            use_container_width=True)
    else:
        st.info("Sélectionnez au moins un type de désert.")

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
            "risk_score":                "⚠️ Risque ML",
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

        # Mini carte 2 communes
        df_2 = pd.DataFrame([
            {"latitude":float(rA["latitude"]),"longitude":float(rA["longitude"]),"ville":A,"color":[26,86,219,230],"dept":str(rA["dept_nom"])},
            {"latitude":float(rB["latitude"]),"longitude":float(rB["longitude"]),"ville":B,"color":[109,40,217,230],"dept":str(rB["dept_nom"])},
        ])
        view_c = pdk.ViewState(
            latitude=(float(rA["latitude"])+float(rB["latitude"]))/2,
            longitude=(float(rA["longitude"])+float(rB["longitude"]))/2,
            zoom=10, pitch=40)
        st.pydeck_chart(pdk.Deck(
            layers=[
                pdk.Layer("ColumnLayer", data=df_2, get_position="[longitude,latitude]",
                    get_elevation=1500, elevation_scale=1, radius=350, get_fill_color="color",
                    pickable=True, coverage=0.9),
                pdk.Layer("ScatterplotLayer", data=df_2, get_position="[longitude,latitude]",
                    get_color="color", get_radius=120, pickable=True),
            ],
            initial_view_state=view_c,
            tooltip={"html":"<b>{ville}</b><br><span style='color:#ccc;font-size:11px;'>{dept}</span>"},
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"),
            use_container_width=True)

        # Tableau comparaison
        COLS_COMP = {
            "taux_chomage":"Chômage (%)","taux_pauvrete":"Pauvreté (%)","revenu_median":"Revenu médian (€)",
            "prix_m2_median":"Prix m² (€)","nb_entreprises_actives":"Nb entreprises",
            "score_attractivite":"⭐ Attractivité","opportunity_score":"🎯 Opportunité ML",
            "pred_attractivite_2026":"🔮 Pred. 2026","risk_score":"⚠️ Risque ML",
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
def main():
    init_session_state()

    # Chargement des données
    with st.spinner("🧊 Chargement des données territoriales..."):
        df = load_data()

    # Routing
    if st.session_state.role is None:
        render_sidebar(df)
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
    elif "Déserts" in page:
        page_deserts(df)
    elif "Indicateurs" in page:
        page_indicateurs(df)
    elif "Prédictions ML" in page:
        page_predictions_ml(df)
    elif "Opportunités" in page:
        page_opportunites(df)
    elif "Communes" in page:
        page_communes(df)
    elif "Assistant IA" in page:
        page_assistant(df)
    else:
        st.info("Page non trouvée.")


if __name__ == "__main__":
    main()
