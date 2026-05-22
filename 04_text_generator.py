"""
ICEBERG - MODÈLE 4 : GÉNÉRATION DE TEXTE
Version finale avec clé API intégrée
"""

import os
import pandas as pd
import numpy as np
import joblib
from mistralai import Mistral

# ============================================================
# 🔑 CLÉ API MISTRAL (NE PAS PARTAGER)
# ============================================================
MISTRAL_API_KEY = "Ajouter la clée api"

print("=" * 60)
print("🧊 ICEBERG - MODÈLE 4 : GÉNÉRATION DE TEXTE")
print("   Rapports intelligents avec Mistral AI")
print("=" * 60)

# ============================================================
# 1. CHARGEMENT DES MODÈLES ML
# ============================================================
print("\n📂 1. Chargement des modèles ML...")

models = {
    'classifier': joblib.load("models/classifier_risque_pro.pkl"),
    'predictor': joblib.load("models/predictor_risk_score.pkl"),
    'scaler_classifier': joblib.load("models/scaler_pro.pkl"),
    'scaler_predictor': joblib.load("models/scaler_predictor.pkl"),
    'label_encoder': joblib.load("models/label_encoder_pro.pkl"),
    'features_classifier': joblib.load("models/features_pro.pkl"),
    'features_predictor': joblib.load("models/features_predictor.pkl")
}

df = pd.read_csv("iceberg final v3.csv", encoding="utf-8")
print(f"   ✅ {len(df)} communes chargées")

# ============================================================
# 2. CONNEXION À MISTRAL AI
# ============================================================
print("\n🤖 2. Connexion à Mistral AI...")
try:
    client = Mistral(api_key=MISTRAL_API_KEY)
    print("   ✅ Client Mistral connecté")
except Exception as e:
    print(f"   ❌ Erreur : {e}")
    exit()

# ============================================================
# 3. FONCTION DE PRÉDICTION
# ============================================================
def predict_commune(ville):
    """Prédit le risque d'une commune"""
    commune = df[df['ville'] == ville]
    if len(commune) == 0:
        return None
    
    # Classification
    features_classif = {}
    for f in models['features_classifier']:
        if f in commune.columns:
            features_classif[f] = commune[f].values[0]
    
    X_classif = np.array([list(features_classif.values())])
    X_classif_scaled = models['scaler_classifier'].transform(X_classif)
    classe_code = models['classifier'].predict(X_classif_scaled)[0]
    classe = models['label_encoder'].inverse_transform([classe_code])[0]
    
    # Prédiction score
    features_pred = {}
    for f in models['features_predictor']:
        if f in commune.columns:
            features_pred[f] = commune[f].values[0]
    
    X_pred = np.array([list(features_pred.values())])
    X_pred_scaled = models['scaler_predictor'].transform(X_pred)
    score = models['predictor'].predict(X_pred_scaled)[0]
    
    return {
        'ville': ville,
        'code_dept': commune['code_dept'].values[0],
        'classe': classe,
        'score': score,
        'population': commune['population'].values[0],
        'taux_chomage': commune['taux_chomage'].values[0],
        'revenu_median': commune['revenu_median'].values[0],
        'nb_entreprises': commune['nb_entreprises_actives'].values[0]
    }

# ============================================================
# 4. FONCTIONS DE GÉNÉRATION DE TEXTE
# ============================================================
def generer_rapport_commune(ville):
    """Génère un rapport complet sur une commune"""
    data = predict_commune(ville)
    if not data:
        return f"❌ Commune '{ville}' non trouvée."
    
    emoji = "🔴" if data['classe'] == 'CRITIQUE' else ("🟠" if data['classe'] == 'ELEVE' else "🟢")
    
    prompt = f"""
Génère un RAPPORT TERRITORIAL PROFESSIONNEL pour la commune suivante :

📊 DONNÉES COMMUNE {data['ville']} ({data['code_dept']}) :
- Classe de risque : {data['classe']} {emoji}
- Score de risque : {data['score']:.3f}/1.00
- Population : {data['population']:,} habitants
- Taux de chômage : {data['taux_chomage']:.1f}%
- Revenu médian : {data['revenu_median']:,} €
- Nombre d'entreprises : {data['nb_entreprises']}

Le rapport doit contenir :
1. Résumé exécutif (2-3 phrases)
2. Analyse du risque (pourquoi cette classe ?)
3. Recommandation pour un investisseur
4. Verdict final

Style : professionnel, concis, actionable.
"""
    
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur Mistral : {str(e)}"

def generer_comparaison(ville1, ville2):
    """Génère une comparaison entre deux communes"""
    data1 = predict_commune(ville1)
    data2 = predict_commune(ville2)
    
    if not data1 or not data2:
        return "Une ou deux communes non trouvées."
    
    prompt = f"""
Compare ces deux communes pour un investisseur :

COMMUNE A : {ville1}
- Classe : {data1['classe']} | Score: {data1['score']:.3f}
- Chômage : {data1['taux_chomage']}% | Revenu : {data1['revenu_median']:,}€
- Entreprises : {data1['nb_entreprises']}

COMMUNE B : {ville2}
- Classe : {data2['classe']} | Score: {data2['score']:.3f}
- Chômage : {data2['taux_chomage']}% | Revenu : {data2['revenu_median']:,}€
- Entreprises : {data2['nb_entreprises']}

Génère une comparaison professionnelle avec :
1. Forces et faiblesses de chaque commune
2. Recommandation finale (laquelle choisir pour investir)
3. Justification basée sur les données
"""
    
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur Mistral : {str(e)}"

def generer_alerte_auto():
    """Génère une alerte automatique sur les zones à risque"""
    communes_critiques = df[df['classe_risque'] == 'CRITIQUE'].nlargest(5, 'risque_credit_local')
    
    liste_communes = ""
    for _, c in communes_critiques.iterrows():
        liste_communes += f"- {c['ville']} ({c['code_dept']}) : score {c['risque_credit_local']:.3f}\n"
    
    prompt = f"""
Génère un message d'ALERTE TERRITORIALE pour un directeur d'agence.

Communes en zone CRITIQUE détectées :
{liste_communes}

Le message doit contenir :
1. Un titre accrocheur
2. La liste des communes à surveiller
3. Des actions concrètes recommandées
4. Niveau d'urgence (ÉLEVÉ)
"""
    
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur Mistral : {str(e)}"

# ============================================================
# 5. TESTS
# ============================================================
print("\n🧪 5. Tests de génération de texte...\n")
print("-" * 60)

# Test 1 : Rapport pour Massy
print("\n📄 TEST 1 : Rapport pour Massy")
print("-" * 40)
print(generer_rapport_commune("Massy"))

print("\n" + "=" * 60)
print("🎉 MODÈLE 4 PRÊT !")
print("=" * 60)

print("\n📋 FONCTIONS DISPONIBLES :")
print("   - generer_rapport_commune('Massy') → Rapport détaillé")
print("   - generer_comparaison('Massy', 'Créteil') → Comparaison")
print("   - generer_alerte_auto() → Alertes automatiques")
