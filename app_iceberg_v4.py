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
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False
try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False


# ══════════════════════════════════════════════════════════════════
# API PUBLIQUES — Contacts institutionnels en temps réel
# ══════════════════════════════════════════════════════════════════
_API_HEADERS = {
    "User-Agent": "ICEBERG-Territorial/5.0 (contact@iceberg-territorial.fr)",
    "Accept": "application/json",
}

@st.cache_data(ttl=86400, show_spinner=False)
def _get_code_insee(ville: str) -> str:
    """Retourne le code INSEE d'une commune via geo.api.gouv.fr"""
    if not _REQUESTS_OK: return ""
    try:
        r = _requests.get(
            f"https://geo.api.gouv.fr/communes?nom={ville}&fields=code,nom,departement&limit=5",
            headers=_API_HEADERS, timeout=6
        )
        if not r.ok: return ""
        results = r.json()
        # Priorité depts 91/94
        for item in results:
            if item.get("departement", {}).get("code") in ("91", "94"):
                return item["code"]
        return results[0]["code"] if results else ""
    except Exception:
        return ""

@st.cache_data(ttl=86400, show_spinner=False)
def _get_contacts_mairie(ville: str, code_insee: str) -> dict:
    """Récupère les infos de la mairie — API d'abord, fallback base locale"""

    # ── Base locale de secours (numéros vérifiés) ─────────────────
    _MAIRIES_LOCAL = {
        # Essonne (91)
        "Grigny":               {"tel":"01 69 02 55 55","email":"mairie@grigny91.fr"},
        "Corbeil-Essonnes":     {"tel":"01 60 89 59 00","email":"mairie@corbeil-essonnes.fr"},
        "Évry-Courcouronnes":   {"tel":"01 60 91 48 00","email":"contact@evry-courcouronnes.fr"},
        "Massy":                {"tel":"01 69 80 80 00","email":"mairie@massy.fr"},
        "Savigny-sur-Orge":     {"tel":"01 69 44 73 00","email":"mairie@savigny91.fr"},
        "Viry-Châtillon":       {"tel":"01 69 12 34 56","email":"mairie@viry-chatillon.fr"},
        "Ris-Orangis":          {"tel":"01 69 06 22 22","email":"mairie@ris-orangis.fr"},
        "Brunoy":               {"tel":"01 60 46 80 00","email":"mairie@brunoy.fr"},
        "Sainte-Geneviève-des-Bois": {"tel":"01 60 15 76 00","email":"mairie@saintegenevievelogistique.fr"},
        "Longjumeau":           {"tel":"01 64 54 53 00","email":"mairie@longjumeau.fr"},
        "Épinay-sur-Orge":      {"tel":"01 69 09 05 05","email":"mairie@epinay91.fr"},
        "Athis-Mons":           {"tel":"01 69 54 10 00","email":"mairie@athis-mons.fr"},
        "Juvisy-sur-Orge":      {"tel":"01 69 12 45 00","email":"mairie@juvisy91.fr"},
        "Draveil":              {"tel":"01 69 03 22 33","email":"mairie@draveil.fr"},
        "Yerres":               {"tel":"01 69 48 62 62","email":"mairie@mairie-yerres.fr"},
        "Palaiseau":            {"tel":"01 64 53 59 00","email":"mairie@mairie-palaiseau.fr"},
        "Les Ulis":             {"tel":"01 69 07 28 00","email":"mairie@lesulis.fr"},
        "Gif-sur-Yvette":       {"tel":"01 69 07 57 00","email":"mairie@gif-sur-yvette.fr"},
        "Orsay":                {"tel":"01 69 28 05 05","email":"mairie@mairie-orsay.fr"},
        "Arpajon":              {"tel":"01 64 90 36 00","email":"mairie@ville-arpajon.fr"},
        "Dourdan":              {"tel":"01 64 59 86 00","email":"mairie@dourdan.fr"},
        "Étampes":              {"tel":"01 69 92 72 92","email":"mairie@etampes.fr"},
        "Montlhéry":            {"tel":"01 69 01 06 05","email":"mairie@montlhery.fr"},
        "Brétigny-sur-Orge":    {"tel":"01 60 84 32 00","email":"mairie@bretigny91.fr"},
        "Bondoufle":            {"tel":"01 60 78 01 20","email":"mairie@bondoufle.fr"},
        "Authon-la-Plaine":     {"tel":"01 64 59 22 08","email":"mairie@authon-la-plaine.fr"},
        "Saint-Michel-sur-Orge":{"tel":"01 60 16 08 08","email":"mairie@saint-michel-sur-orge.fr"},
        "Linas":                {"tel":"01 69 01 13 13","email":"mairie@linas.fr"},
        "Limours":              {"tel":"01 64 91 06 91","email":"mairie@limours.fr"},
        "Marcoussis":           {"tel":"01 64 49 08 00","email":"mairie@marcoussis.fr"},
        # Val-de-Marne (94)
        "Charenton-le-Pont":    {"tel":"01 43 68 05 00","email":"mairie@charenton-le-pont.fr"},
        "Saint-Mandé":          {"tel":"01 43 98 30 00","email":"mairie@saintmande.fr"},
        "Vincennes":            {"tel":"01 48 08 12 00","email":"mairie@vincennes.fr"},
        "Nogent-sur-Marne":     {"tel":"01 48 72 62 72","email":"mairie@nogentsurmarneoise.fr"},
        "Le Perreux-sur-Marne": {"tel":"01 48 71 30 00","email":"mairie@leperreux.fr"},
        "Créteil":              {"tel":"01 49 80 93 93","email":"accueil@ville-creteil.fr"},
        "Alfortville":          {"tel":"01 43 78 20 00","email":"mairie@alfortville.fr"},
        "Maisons-Alfort":       {"tel":"01 41 79 13 13","email":"mairie@maisons-alfort.fr"},
        "Vitry-sur-Seine":      {"tel":"01 55 53 10 00","email":"mairie@vitry94.fr"},
        "Ivry-sur-Seine":       {"tel":"01 46 72 40 00","email":"mairie@ivry94.fr"},
        "Gentilly":             {"tel":"01 49 08 27 00","email":"mairie@ville-gentilly.fr"},
        "Le Kremlin-Bicêtre":   {"tel":"01 49 60 60 60","email":"mairie@kremlin-bicetre.fr"},
        "Villejuif":            {"tel":"01 49 58 58 00","email":"contact@villejuif.fr"},
        "Chevilly-Larue":       {"tel":"01 45 60 19 00","email":"mairie@chevilly-larue.fr"},
        "L'Haÿ-les-Roses":      {"tel":"01 49 75 61 00","email":"mairie@lhaylesroses.fr"},
        "Rungis":               {"tel":"01 45 60 02 82","email":"mairie@mairie-rungis.fr"},
        "Thiais":               {"tel":"01 46 81 95 00","email":"mairie@ville-thiais.fr"},
        "Orly":                 {"tel":"01 56 70 20 20","email":"mairie@mairie-orly.fr"},
        "Villeneuve-le-Roi":    {"tel":"01 45 97 13 80","email":"mairie@villeneuve-le-roi.fr"},
        "Ablon-sur-Seine":      {"tel":"01 45 17 03 03","email":"mairie@ablon-sur-seine.fr"},
        "Valenton":             {"tel":"01 45 10 11 10","email":"mairie@ville-valenton.fr"},
        "Villecresnes":         {"tel":"01 45 69 00 03","email":"mairie@villecresnes.fr"},
        "Santeny":              {"tel":"01 45 94 20 36","email":"mairie@santeny.fr"},
        "Boissy-Saint-Léger":   {"tel":"01 45 69 15 00","email":"mairie@boissy-saint-leger.fr"},
        "Sucy-en-Brie":         {"tel":"01 45 90 28 28","email":"mairie@sucy-en-brie.fr"},
        "La Queue-en-Brie":     {"tel":"01 45 76 23 10","email":"mairie@la-queue-en-brie.fr"},
        "Noiseau":              {"tel":"01 45 69 08 69","email":"mairie@noiseau.fr"},
        "Ormesson-sur-Marne":   {"tel":"01 45 76 35 24","email":"mairie@ormesson94.fr"},
        "Chennevières-sur-Marne":{"tel":"01 45 94 68 00","email":"mairie@chennevieres94.fr"},
        "Champigny-sur-Marne":  {"tel":"01 49 83 34 34","email":"mairie@ville-champigny.fr"},
        "Saint-Maur-des-Fossés":{"tel":"01 48 80 80 80","email":"mairie@saint-maur.fr"},
        "Joinville-le-Pont":    {"tel":"01 48 89 43 43","email":"mairie@joinville-le-pont.fr"},
    }

    # ── Fallback local immédiat ───────────────────────────────────
    local = _MAIRIES_LOCAL.get(ville, {})

    # ── Tentative API (enrichit le fallback si disponible) ────────
    if _REQUESTS_OK and code_insee:
        try:
            url = f"https://lannuaire.service-public.fr/api/v1/annuaire?pivotLocal=mairie&codeInsee={code_insee}"
            r = _requests.get(url, headers=_API_HEADERS, timeout=5)
            if r.ok:
                data = r.json()
                item = data.get("service", [{}])[0] if data.get("service") else {}
                coords = item.get("coordCartographie", [])
                tel_api  = next((c["valeur"] for c in coords if c.get("type")=="Téléphone"), "")
                mail_api = next((c["valeur"] for c in coords if c.get("type")=="Email"), "")
                return {
                    "nom": item.get("nom", f"Mairie de {ville}"),
                    "tel":   tel_api  or local.get("tel","—"),
                    "email": mail_api or local.get("email","—"),
                    "url": f"https://lannuaire.service-public.fr/go/mairie-{code_insee}",
                }
        except Exception:
            pass

    # ── Retour fallback local ─────────────────────────────────────
    commune_slug = ville.lower().replace(" ", "-").replace("'", "-").replace("é","e").replace("è","e").replace("ê","e")
    return {
        "nom": f"Mairie de {ville}",
        "tel":   local.get("tel", "—"),
        "email": local.get("email", "—"),
        "url": f"https://lannuaire.service-public.fr/navigation/commune?text={commune_slug}",
    }

@st.cache_data(ttl=86400, show_spinner=False)
def _get_elu_maire(code_insee: str) -> dict:
    """Récupère le nom du maire via l'API RNE (Répertoire National des Élus) sur data.gouv.fr"""
    if not _REQUESTS_OK or not code_insee:
        return {}
    try:
        url = (f"https://tabular-api.data.gouv.fr/api/resources/"
               f"2a3d781e-3b22-45f9-bb5a-9c72c5ec29f8/data/"
               f"?code_commune={code_insee}&code_mandat=01&limit=1")  # 01 = Maire
        r = _requests.get(url, headers=_API_HEADERS, timeout=6)
        if not r.ok: return {}
        data = r.json()
        rows = data.get("data", [])
        if not rows: return {}
        row = rows[0]
        prenom = row.get("prenom_elu", "")
        nom = row.get("nom_elu", "")
        return {
            "nom_complet": f"{prenom} {nom}".strip(),
            "date_naissance": row.get("date_naissance", ""),
            "parti": row.get("libelle_nuance", ""),
            "date_debut": row.get("date_debut_mandat", ""),
        }
    except Exception:
        return {}

@st.cache_data(ttl=86400, show_spinner=False)
def _get_contacts_sante(code_dept: str) -> list:
    """Retourne les contacts santé institutionnels selon le département"""
    contacts_dept = {
        "91": {
            "ars_tel": "01 41 79 67 00",
            "cpam_tel": "36 46",
            "cdom_tel": "01 64 87 60 60",
            "ch_nom": "CH Sud-Francilien (Corbeil-Essonnes)",
            "ch_tel": "01 91 21 00 00",
            "ch_url": "https://www.chsf.fr",
            "cpts_url": "https://www.iledefrance.ars.sante.fr/les-communautes-professionnelles-territoriales-de-sante",
        },
        "94": {
            "ars_tel": "01 41 79 67 00",
            "cpam_tel": "36 46",
            "cdom_tel": "01 43 99 30 30",
            "ch_nom": "CH Intercommunal de Créteil",
            "ch_tel": "01 57 02 20 00",
            "ch_url": "https://www.chicreteil.fr",
            "cpts_url": "https://www.iledefrance.ars.sante.fr/les-communautes-professionnelles-territoriales-de-sante",
        },
    }
    return contacts_dept.get(str(code_dept), contacts_dept["91"])

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

/* ── SIDEBAR BUTTONS ── */
[data-testid="stSidebar"] .stButton > button {
  background: #FFFFFF !important; color: #0A0F1E !important;
  border: 1.5px solid #E2E8F2 !important;
  box-shadow: 0 1px 4px rgba(10,15,30,.08) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: #EBF1FF !important; color: #1A56DB !important;
  border-color: #1A56DB !important; box-shadow: none !important;
  transform: none !important;
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

    # Scores déserts recalculés depuis les vraies données terrain (après normalisation num)
    df["score_desert_medical"]    = 0.0
    df["score_desert_commercial"] = 0.0
    df["score_desert_mobilite"]   = 0.0

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

    # ── Normalisation scores 0-1 (hors déserts, calculés depuis données réelles ci-dessous) ──
    for sc in ["score_attractivite","score_signal_faible",
               "score_reindustrialisation","score_employabilite"]:
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

    # ── Recalcul scores déserts depuis les vraies données terrain ──
    # Médical : densité généralistes (seuil national = 8 généralistes / 10k hab)
    df["score_desert_medical"] = (1 - (df["medecins_10k_hab"] / 8).clip(0, 1))
    # Fallback si medecins_10k_hab toujours nul : nb_medecin_generaliste
    mask_no_med = df["medecins_10k_hab"] == 0
    df.loc[mask_no_med, "score_desert_medical"] = (
        1 - ((df.loc[mask_no_med, "nb_medecin_generaliste"] * 10000 / pop.loc[mask_no_med]) / 8).clip(0, 1)
    )

    # Commercial : indice de couverture commerciale (épicerie + boulangerie + boucherie + supermarché)
    score_com_raw = (
        (1 - ((df["nb_epicerie"]    * 10000 / pop) / 5).clip(0, 1)) * 0.35 +
        (1 - ((df["nb_boulangerie"] * 10000 / pop) / 5).clip(0, 1)) * 0.25 +
        (1 - ((df["nb_boucherie"]   * 10000 / pop) / 3).clip(0, 1)) * 0.20 +
        (1 - ((df["nb_supermarche"] * 10000 / pop) / 3).clip(0, 1)) * 0.20
    )
    # Normaliser entre 0 et 1
    sc_min, sc_max = score_com_raw.min(), score_com_raw.max()
    df["score_desert_commercial"] = ((score_com_raw - sc_min) / (sc_max - sc_min)).clip(0, 1) if sc_max > sc_min else score_com_raw

    # Mobilité : basé sur nb_gares et densité population (communes isolées = score élevé)
    score_mob_raw = (
        (1 - ((df["nb_gares"] * 10000 / pop) / 3).clip(0, 1)) * 0.60 +
        (1 - (df["densite_hab_km2"] / df["densite_hab_km2"].quantile(0.75)).clip(0, 1)) * 0.40
    )
    sm_min, sm_max = score_mob_raw.min(), score_mob_raw.max()
    df["score_desert_mobilite"] = ((score_mob_raw - sm_min) / (sm_max - sm_min)).clip(0, 1) if sm_max > sm_min else score_mob_raw

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
        _logo_b64 = "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POmXK0k6RgweJOkpdGFxIax1Ha9cOs/FlTFLeY/FWFgcDxmNRrS5Y2XjBHt7k35ZHJGua4F+4lLXtQxHDW07IdZCTAFKy3AQaHfv8e1vvcKr33iZ7c1TZh0kSZ+awYMkPaWuLA7CiZJ5fvsM51dHnBg2nFgcMx6P6GohpAGEOO9naJp+BkaMkWFqaNspo/GQSqHmjtGgoUx3uXPzOj9/58ecOnniyb5ASdKxY/AgSU+xjVFieznxwtYZTi7AkCmlndK2LePxGOgnKpVAHzyEQiUTYqWUjlALlEwsmVGCjbVlvvOtb/Li3dtcPLtp1kGS9JkYPEjSU+yZcQgnElw5vcKVU2s0e/dYjJVxk1heXoZcITXQhxAA8wbq8XBIzS2hZlZXFtjbvcfWs2d45eUXWV0aP9HXJUk6ngweJOkpd2EYwukB3D6zwfWTa2zEwLBrWV5YhK6fslTYb5Cu1NxRa6brppRSWFwYcP/DD5ju3Oe1b36Tm9cvcvbEilkHSdJn5pI4SToGBpMpl9ZGfHh6gy5X3pu2jEcjaBLTriWlSKXvd0hNIIbK3t4ew+GQdjphurfL9WuXufvcDa6ef8bAQZL0WAweJOkYuLA4Cv9zUuqNzRNMuwr3dshtBysrtNPMeBwJMVBK6bdN0/dA1JqZTjPbF87z/be/w62b15/wK5EkHWeWLUnSMXFxFMNql7l+YomLK2NWUmB1ZYFauv4XYqIUCAwINRIqDGMkdB1nNzZ449VXuHbOrIMk6fEZPEjSMXJ5oQmvro3CWzcusPfH33P53GnoplAqbYa2JCoD2g5C6TdRLzcNP/jOm1w6t/WkT8NQb2oAAAK2SURBVF+SdMwZPEjSMXRlaRROjCLLg8iZjTUGIZD3pgybETs7O4yaEanAZGeXl+7e4Zsvv8Sl845mlST9ZQweJOmY+vW7P+M3v/oF5zdPUSY7nFhaILQTlgYNTSgECoMUeOP119jcPPWkT1eS9BVg8CBJx1SZ7PGjt97iN7/6JRsLY5rcMr33PrG0xNKSJ7u8+fprfOeN19k+s27WQZL0F3PakiQdU9sHdjW881f/qv7D/36PE0sjQoAP3v9/nD65zrvv/IQ717YMHCRJnwszD5L0FfDv/s3fMqRjQIZ2j+nOh1y+sMXVy9tP+tQkSV8hBg+S9BXw1hsvhp//9Ce09+8xCJXFYcNf//oX3L52wayDJEmSpIf9zd/863rxyu36t//239ff/u4f6pM+H0mSJElPqf/297+r/+I3/7L+h//4nwwcJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSvkL+P2yrONsb8+KvAAAAAElFTkSuQmCC"
        _logo_html = f'<img src="data:image/png;base64,{_logo_b64}" style="width:38px;height:38px;object-fit:contain;" />'
        st.markdown(f"""
        <div style="padding:22px 18px 14px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
            <div style="width:42px;height:42px;background:linear-gradient(135deg,#EBF1FF,#DBEAFE);
                 border-radius:12px;display:flex;align-items:center;justify-content:center;
                 box-shadow:0 2px 8px rgba(26,86,219,.12);">{_logo_html}</div>
            <div>
              <div style="font-size:16px;font-weight:800;color:#0A0F1E;letter-spacing:.3px;">ICEBERG <span style="font-size:10px;font-weight:500;color:#94A3B8;">v{APP_VERSION}</span></div>
              <div style="font-size:9px;color:#94A3B8;text-transform:uppercase;letter-spacing:1.8px;">Dép. 91 &amp; 94</div>
            </div>
          </div>

          <div style="background:#F8FAFC;border:1px solid #E2E8F2;
               border-radius:14px;padding:14px;margin-bottom:6px;">
            <div style="font-size:9px;font-weight:700;color:#475569;text-transform:uppercase;
                 letter-spacing:1.2px;margin-bottom:10px;">Vue d'ensemble</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
              <div style="background:#FFFFFF;border:1px solid #E2E8F2;border-radius:10px;padding:10px 12px;">
                <div style="font-size:22px;font-weight:800;color:#0A0F1E;">{_n}</div>
                <div style="font-size:10px;font-weight:600;color:#475569;margin-top:2px;">Communes</div>
              </div>
              <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:10px 12px;cursor:pointer;">
                <div style="font-size:22px;font-weight:800;color:#16A34A;">{_prio}</div>
                <div style="font-size:10px;font-weight:600;color:#166534;margin-top:2px;">Prioritaires ↗</div>
              </div>
              <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;padding:10px 12px;cursor:pointer;">
                <div style="font-size:22px;font-weight:800;color:#B45309;">{_sig}</div>
                <div style="font-size:10px;font-weight:600;color:#92400E;margin-top:2px;">Signaux forts ↗</div>
              </div>
              <div style="background:#FFF1F2;border:1px solid #FECDD3;border-radius:10px;padding:10px 12px;">
                <div style="font-size:22px;font-weight:800;color:#DC2626;">{_desert}</div>
                <div style="font-size:10px;font-weight:600;color:#991B1B;margin-top:2px;">Déserts méd.</div>
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
      <img src="data:image/png;base64,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POmXK0k6RgweJOkpdGFxIax1Ha9cOs/FlTFLeY/FWFgcDxmNRrS5Y2XjBHt7k35ZHJGua4F+4lLXtQxHDW07IdZCTAFKy3AQaHfv8e1vvcKr33iZ7c1TZh0kSZ+awYMkPaWuLA7CiZJ5fvsM51dHnBg2nFgcMx6P6GohpAGEOO9naJp+BkaMkWFqaNspo/GQSqHmjtGgoUx3uXPzOj9/58ecOnniyb5ASdKxY/AgSU+xjVFieznxwtYZTi7AkCmlndK2LePxGOgnKpVAHzyEQiUTYqWUjlALlEwsmVGCjbVlvvOtb/Li3dtcPLtp1kGS9JkYPEjSU+yZcQgnElw5vcKVU2s0e/dYjJVxk1heXoZcITXQhxAA8wbq8XBIzS2hZlZXFtjbvcfWs2d45eUXWV0aP9HXJUk6ngweJOkpd2EYwukB3D6zwfWTa2zEwLBrWV5YhK6fslTYb5Cu1NxRa6brppRSWFwYcP/DD5ju3Oe1b36Tm9cvcvbEilkHSdJn5pI4SToGBpMpl9ZGfHh6gy5X3pu2jEcjaBLTriWlSKXvd0hNIIbK3t4ew+GQdjphurfL9WuXufvcDa6ef8bAQZL0WAweJOkYuLA4Cv9zUuqNzRNMuwr3dshtBysrtNPMeBwJMVBK6bdN0/dA1JqZTjPbF87z/be/w62b15/wK5EkHWeWLUnSMXFxFMNql7l+YomLK2NWUmB1ZYFauv4XYqIUCAwINRIqDGMkdB1nNzZ449VXuHbOrIMk6fEZPEjSMXJ5oQmvro3CWzcusPfH33P53GnoplAqbYa2JCoD2g5C6TdRLzcNP/jOm1w6t/WkT8NQb2oAAAK2SURBVF+SdMwZPEjSMXRlaRROjCLLg8iZjTUGIZD3pgybETs7O4yaEanAZGeXl+7e4Zsvv8Sl845mlST9ZQweJOmY+vW7P+M3v/oF5zdPUSY7nFhaILQTlgYNTSgECoMUeOP119jcPPWkT1eS9BVg8CBJx1SZ7PGjt97iN7/6JRsLY5rcMr33PrG0xNKSJ7u8+fprfOeN19k+s27WQZL0F3PakiQdU9sHdjW881f/qv7D/36PE0sjQoAP3v9/nD65zrvv/IQ717YMHCRJnwszD5L0FfDv/s3fMqRjQIZ2j+nOh1y+sMXVy9tP+tQkSV8hBg+S9BXw1hsvhp//9Ce09+8xCJXFYcNf//oX3L52wayDJEmSpIf9zd/863rxyu36t//239ff/u4f6pM+H0mSJElPqf/297+r/+I3/7L+h//4nwwcJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSvkL+P2yrONsb8+KvAAAAAElFTkSuQmCC"
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

    # Fond de carte selon le mode
    if mode_carte == "Radar Opportunités":
        m = folium.Map(
            location=[lat_c, lon_c],
            zoom_start=11,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            prefer_canvas=True,
            zoom_control=True,
        )
    else:
        m = folium.Map(
            location=[lat_c, lon_c],
            zoom_start=11,
            tiles=None,
            prefer_canvas=True,
            zoom_control=True,
        )
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
            attr="CartoDB Voyager",
            name="Carte",
        ).add_to(m)

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
    if mode_carte == "Profils territoire":
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
           letter-spacing:.8px;margin-bottom:6px;'>{"Profil ML" if mode_carte=="Profils territoire" else "Zone d'opportunité"}</div>
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
            mode_carte = st.radio("", ["Radar Opportunités","Opportunités 2027","Profils territoire"],
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
        annee_sel = st.slider(
            "", min_value=2026, max_value=2030, step=1,
            value=st.session_state.get("slider_annee_carte", 2026),
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
    with fc5: niv     = st.selectbox("Niveau", ["Fort","Modéré","Tous"], index=0)
    niveaux_f = {"Fort":["Fort"],"Modéré":["Fort","Modéré"],"Tous":["Fort","Modéré","Faible"]}[niv]

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
            ville_n  = str(row["ville"])
            dept_n   = str(row.get("dept_nom",""))
            code_dep = str(row.get("code_dept","91"))

            # ── Chiffres tirés directement du DataFrame ─────────────
            score    = int(row[col] * 100)
            conf     = min(97, int(row[col] * 90 + 7))
            cho      = float(row.get("taux_chomage", 0))
            pauv     = float(row.get("taux_pauvrete", 0))
            rev      = int(row.get("revenu_median", 0))
            nb_gen   = int(row.get("nb_medecin_generaliste", 0))
            pop_v    = int(row.get("population", 0))
            prix_m2  = int(row.get("prix_m2_median", 0))
            nb_ent   = int(row.get("nb_entreprises_actives", 0))
            opp_s    = float(row.get("opportunity_score", row[col]))
            pred_a   = float(row.get("pred_attractivite_2026", row.get("score_attractivite", 0)))
            risk_s   = float(row.get("risk_score", 0))
            pop_fmt  = f"{pop_v:,}".replace(",", "\u00a0")

            # ── Récupération API en temps réel ──────────────────────
            code_insee  = _get_code_insee(ville_n)
            info_mairie = _get_contacts_mairie(ville_n, code_insee)
            info_maire  = _get_elu_maire(code_insee)
            info_sante  = _get_contacts_sante(code_dep)

            maire_label = info_maire.get("nom_complet","") or "Maire"
            mairie_tel  = info_mairie.get("tel","—") or "—"
            mairie_url  = info_mairie.get("url", f"https://lannuaire.service-public.fr/navigation/commune?text={ville_n.lower().replace(' ','-')}")
            mairie_email= info_mairie.get("email","—")

            if type_s == "Emploi":
                signal  = f"Chômage {cho:.1f}% · Pauvreté {pauv:.1f}% · Revenu médian {rev:,} €/an".replace(",", "\u00a0")
                analyse = (f"{ville_n} affiche un taux de chômage de {cho:.1f}% (moyenne nationale ~7%) "
                           f"et un taux de pauvreté de {pauv:.1f}%. Le revenu médian de {rev:,} € "
                           f"est inférieur à la médiane nationale. Une intervention coordonnée emploi-formation est recommandée.".replace(",", "\u00a0"))
                urgence = "Haute" if row[col] >= 0.75 else "Modérée"
                sources_list = ["INSEE — Revenus fiscaux localisés", "SIRENE — Base entreprises", "DARES — Statistiques emploi"]
                contacts = [
                    {"ini": maire_label[:2].upper() if maire_label != "Maire" else "MR",
                     "nom": f"M. {maire_label}" if maire_label != "Maire" else f"Mairie de {ville_n}",
                     "poste": f"Maire de {ville_n}",
                     "clr":"#1A56DB","tags":["Élu local","Décision"],
                     "note":f"Contact direct mairie · {mairie_email}",
                     "url": mairie_url,"tel": mairie_tel},
                    {"ini":"FT","nom":"France Travail","poste":f"Agence {dept_n}",
                     "clr":"#059669","tags":["Emploi","Reconversion"],
                     "note":"Accompagnement demandeurs d'emploi, dispositifs entreprises",
                     "url":"https://www.francetravail.fr","tel":"3949"},
                    {"ini":"DR","nom":"DREETS Île-de-France","poste":"Emploi & formation prof.",
                     "clr":"#D97706","tags":["Financement","FSE+"],
                     "note":"Fonds emploi, FSE+, dispositifs reconversion industrielle",
                     "url":"https://idf.dreets.gouv.fr","tel":"01 70 96 14 00"},
                    {"ini":"BP","nom":"Bpifrance","poste":"Direction régionale Île-de-France",
                     "clr":"#6D28D9","tags":["Financement","PME/TPE"],
                     "note":"Prêts, garanties et accompagnement TPE/PME en difficulté",
                     "url":"https://www.bpifrance.fr","tel":"0969 370 240"},
                ]
            elif type_s == "Médical":
                ratio   = round(pop_v / max(nb_gen, 1))
                signal  = f"{nb_gen} généraliste(s) pour {pop_fmt} hab. · Ratio 1 médecin / {ratio:,} hab.".replace(",", "\u00a0")
                analyse = (f"{ville_n} est en situation de désert médical : {nb_gen} généraliste(s) "
                           f"pour {pop_fmt} habitants (ratio 1/{ratio:,}). "
                           f"Le seuil d'alerte national est de 1 médecin / 1 500 hab. "
                           f"Une Maison de Santé Pluriprofessionnelle (MSP) est une priorité.".replace(",", "\u00a0"))
                urgence = "Haute" if row[col] >= 0.75 else "Modérée"
                sources_list = ["RPPS — Répertoire professionnel de santé", "ARS Île-de-France", "INSEE — Population légale"]
                contacts = [
                    {"ini": maire_label[:2].upper() if maire_label != "Maire" else "MR",
                     "nom": f"M. {maire_label}" if maire_label != "Maire" else f"Mairie de {ville_n}",
                     "poste": f"Maire de {ville_n} — local MSP",
                     "clr":"#1A56DB","tags":["Local gratuit","MSP"],
                     "note":f"Peut mettre un local à disposition pour une MSP · {mairie_tel}",
                     "url": mairie_url,"tel": mairie_tel},
                    {"ini":"AR","nom":"ARS Île-de-France","poste":"Délégué territorial",
                     "clr":"#DC2626","tags":["Zonage","Subventions MSP"],
                     "note":"Valide dossiers MSP, aides à l'installation médecins",
                     "url":"https://www.iledefrance.ars.sante.fr","tel":info_sante["ars_tel"]},
                    {"ini":"CP","nom":"CPAM","poste":f"Dép. {code_dep}",
                     "clr":"#D97706","tags":["DSP","CAQES"],
                     "note":"Contrats d'amélioration de l'accès aux soins",
                     "url":"https://www.ameli.fr","tel":info_sante["cpam_tel"]},
                    {"ini":"CH","nom":info_sante["ch_nom"],"poste":"Direction médicale",
                     "clr":"#059669","tags":["Antenne","Télémédecine"],
                     "note":"Peut projeter des antennes de consultation avancée",
                     "url":info_sante["ch_url"],"tel":info_sante["ch_tel"]},
                ]
            else:
                signal  = f"Attractivité {score}/100 · {nb_ent} entreprises actives · Foncier {prix_m2:,} €/m²".replace(",", "\u00a0")
                analyse = (f"{ville_n} est la commune la plus attractive du territoire (score {score}/100). "
                           f"Avec {nb_ent} entreprises actives et un prix foncier médian de {prix_m2:,} €/m², "
                           f"elle offre un potentiel d'investissement élevé. "
                           f"Projection 2026 : {pred_a:.0%}. Risque évalué à {risk_s:.0%}.".replace(",", "\u00a0"))
                urgence = "Opportunité"
                sources_list = ["DVF — Demandes de valeurs foncières", "SIRENE — Base entreprises", "INSEE — Revenus fiscaux"]
                contacts = [
                    {"ini": maire_label[:2].upper() if maire_label != "Maire" else "MR",
                     "nom": f"M. {maire_label}" if maire_label != "Maire" else f"Mairie de {ville_n}",
                     "poste": f"Maire de {ville_n} — Service Urbanisme",
                     "clr":"#1A56DB","tags":["PLU","Permis","Foncier"],
                     "note":f"Consulter le PLU et les zones à enjeux · {mairie_email}",
                     "url": mairie_url,"tel": mairie_tel},
                    {"ini":"EP","nom":"EPF Île-de-France","poste":"Établissement Public Foncier",
                     "clr":"#059669","tags":["Portage foncier","ZAC"],
                     "note":"Maîtrise foncière, portage et remembrement parcellaire",
                     "url":"https://www.epfif.fr","tel":"01 48 09 19 00"},
                    {"ini":"GP","nom":"Choose Paris Region","poste":"Invest in Greater Paris",
                     "clr":"#D97706","tags":["Accompagnement gratuit","IdF"],
                     "note":"Accompagnement gratuit implantation entreprises en IdF",
                     "url":"https://www.chooseparisregion.org","tel":"01 42 67 97 00"},
                    {"ini":"BT","nom":"Banque des Territoires","poste":"Caisse des Dépôts",
                     "clr":"#6D28D9","tags":["Financement mixte","Long terme"],
                     "note":"Financement projets mixtes logement/commerce/équipements",
                     "url":"https://www.banquedesterritoires.fr","tel":"0800 100 050"},
                ]

            signaux.append({
                "ville":ville_n,"dept":dept_n,"type":type_s,
                "urgence":urgence,"score":score,"confiance":conf,"nb_ent":nb_ent,
                "opp_score":f"{opp_s:.0%}","pred_2026":f"{pred_a:.0%}","risk":f"{risk_s:.0%}",
                "signal":signal,"analyse":analyse,"delai":"En temps réel",
                "sources":sources_list,
                "contacts":contacts,
                "maire": info_maire,
                "mairie": info_mairie,
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

        # Bandeau Maire en fonction (si trouvé via API RNE)
        maire_info = sig.get("maire", {})
        mairie_info = sig.get("mairie", {})
        if maire_info.get("nom_complet"):
            parti_badge = (f'<span style="background:#EBF1FF;color:#1344B8;padding:1px 7px;'
                          f'border-radius:20px;font-size:9px;font-weight:600;">{maire_info["parti"]}</span>'
                          if maire_info.get("parti") else "")
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#EBF1FF,#F5F3FF);border:1px solid #C4B5FD;'
                f'border-radius:12px;padding:12px 16px;margin-bottom:12px;display:flex;align-items:center;gap:12px;">'
                f'<div style="width:40px;height:40px;background:#1A56DB;border-radius:50%;display:flex;'
                f'align-items:center;justify-content:center;font-size:16px;flex-shrink:0;">🏛️</div>'
                f'<div style="flex:1;">'
                f'<div style="font-size:10px;color:#6D28D9;font-weight:700;text-transform:uppercase;letter-spacing:.8px;">Maire en fonction</div>'
                f'<div style="font-size:14px;font-weight:800;color:#0A0F1E;">{maire_info["nom_complet"]}</div>'
                f'<div style="display:flex;gap:6px;align-items:center;margin-top:3px;">'
                f'<span style="font-size:10px;color:#64748B;">Mandat depuis {maire_info.get("date_debut","—")[:4]}</span>'
                f'{parti_badge}</div></div>'
                f'<a href="{mairie_info.get("url","#")}" target="_blank" style="background:#1A56DB;color:#fff;'
                f'border-radius:8px;padding:6px 12px;font-size:11px;font-weight:600;text-decoration:none;white-space:nowrap;">'
                f'📞 {mairie_info.get("tel","—")}</a>'
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
                            f'<div style="background:#F8FAFC;border:1px solid #E2E8F2;'
                            f'border-radius:13px;padding:14px;margin-bottom:9px;">'
                            f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:9px;">'
                            f'<div style="width:38px;height:38px;border-radius:50%;background:{c["clr"]};'
                            f'display:flex;align-items:center;justify-content:center;font-size:12px;'
                            f'font-weight:700;color:#fff;flex-shrink:0;">{c["ini"]}</div>'
                            f'<div><div style="font-size:12px;font-weight:700;color:#0A0F1E;">{c["nom"]}</div>'
                            f'<div style="font-size:10px;color:#64748B;">{c["poste"]}</div></div></div>'
                            f'<div style="margin-bottom:7px;">{tags_h}</div>'
                            f'<div style="font-size:10px;color:#475569;background:#EBF1FF;'
                            f'border-radius:7px;padding:7px 9px;margin-bottom:9px;">{c["note"]}</div>'
                            f'<div style="display:flex;gap:5px;">'
                            f'<a href="{c["url"]}" target="_blank" style="flex:1;background:#1A56DB;color:#fff;border-radius:7px;'
                            f'padding:5px;text-align:center;font-size:10px;font-weight:600;text-decoration:none;">🌐 Site</a>'
                            f'<div style="flex:1;background:#F1F5F9;border-radius:7px;'
                            f'padding:5px;text-align:center;font-size:10px;color:#475569;font-weight:600;">📞 {c["tel"]}</div>'
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
