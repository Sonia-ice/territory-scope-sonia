"""
ICEBERG - MODÈLE 3 : RECOMMANDATION DE COMMUNES SIMILAIRES
Trouve les communes qui ressemblent le plus à une commune donnée
Pour l'aide à la décision d'investissement
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob

print("=" * 60)
print("🧊 ICEBERG - MODÈLE 3 : RECOMMANDATION")
print("   Trouver des communes similaires pour l'investissement")
print("=" * 60)

# -------------------------------------------------------------------
# 1. CHARGER LES DONNÉES
# -------------------------------------------------------------------
print("\n📂 1. Chargement des données...")

csv_files = glob.glob("data-iceberg-v4_rows*.csv") or glob.glob("*.csv")
fichier_csv = csv_files[0] if csv_files else None
if not fichier_csv:
    print("   ❌ Aucun fichier CSV trouvé !")
    exit()

df = pd.read_csv(fichier_csv, encoding="utf-8")
print(f"   ✅ {len(df)} communes chargées")

# -------------------------------------------------------------------
# 2. FEATURES POUR LA SIMILARITÉ
# -------------------------------------------------------------------
print("\n🔧 2. Préparation des features pour similarité...")

features_similarite = [
    "population",
    "densite_hab_km2",
    "taux_chomage",
    "revenu_median",
    "taux_pauvrete",
    "entreprises_1000hab",
    "nb_gares",
    "score_fragilite",
    "score_momentum",
    "potentiel_investissement"
]

features_disponibles = [f for f in features_similarite if f in df.columns]
print(f"   ✅ {len(features_disponibles)} features pour la similarité")

# Nettoyer
df_clean = df.dropna(subset=features_disponibles)
print(f"   📊 {len(df_clean)} communes analysables")

X = df_clean[features_disponibles]

# -------------------------------------------------------------------
# 3. NORMALISATION
# -------------------------------------------------------------------
print("\n📐 3. Normalisation des données...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   ✅ Normalisation terminée")

# -------------------------------------------------------------------
# 4. MODÈLE KNN (K-plus proches voisins)
# -------------------------------------------------------------------
print("\n🤖 4. Entraînement du modèle de similarité...")
knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn.fit(X_scaled)
print("   ✅ Modèle KNN entraîné (trouve les 5 communes les plus similaires)")

# -------------------------------------------------------------------
# 5. SAUVEGARDE
# -------------------------------------------------------------------
print("\n💾 5. Sauvegarde...")
os.makedirs("models", exist_ok=True)

joblib.dump(knn, "models/recommender_knn.pkl")
joblib.dump(scaler, "models/scaler_recommender.pkl")
joblib.dump(features_disponibles, "models/features_recommender.pkl")

# Sauvegarde des infos communes (sans score_risque qui n'existe pas)
colonnes_a_garder = ["ville", "code_dept", "classe_risque"]
if "risque_credit_local" in df_clean.columns:
    colonnes_a_garder.append("risque_credit_local")

communes_ref = df_clean[colonnes_a_garder].copy()
joblib.dump(communes_ref, "models/communes_reference.pkl")

print("   ✅ Modèle sauvegardé : models/recommender_knn.pkl")

# -------------------------------------------------------------------
# 6. TESTS DE RECOMMANDATION
# -------------------------------------------------------------------
print("\n🧪 6. Tests de recommandation...")

def get_similar_communes(ville, df_clean, knn, scaler, features):
    """Trouve les communes similaires à une ville donnée"""
    # Trouver la commune
    commune = df_clean[df_clean['ville'] == ville]
    if len(commune) == 0:
        return None, None
    
    # Préparer les features
    X_commune = commune[features].values.reshape(1, -1)
    X_commune_scaled = scaler.transform(X_commune)
    
    # Trouver les voisins
    distances, indices = knn.kneighbors(X_commune_scaled)
    
    # Récupérer les communes similaires (sans la commune elle-même)
    similaires = df_clean.iloc[indices[0][1:]]
    distances_sim = distances[0][1:]
    
    return similaires, distances_sim

# Tester sur quelques communes
communes_test = ["Massy", "Évry-Courcouronnes", "Créteil", "Vitry-sur-Seine"]

print("\n   🔍 Recommandations d'investissement :")

for ville_test in communes_test:
    if ville_test in df_clean['ville'].values:
        similaires, distances = get_similar_communes(ville_test, df_clean, knn, scaler, features_disponibles)
        
        print(f"\n   📍 {ville_test} :")
        print(f"      🔎 Communes similaires pour comparaison :")
        
        for i, (idx, commune) in enumerate(similaires.iterrows()):
            classe_risque = commune.get('classe_risque', 'N/A')
            if classe_risque == 'MODERE':
                emoji = "🟢"
            elif classe_risque == 'ELEVE':
                emoji = "🟠"
            elif classe_risque == 'CRITIQUE':
                emoji = "🔴"
            else:
                emoji = "⚪"
            
            similarite_pct = max(0, 100 - (distances[i] * 100))
            print(f"         {i+1}. {commune['ville']} ({commune['code_dept']}) {emoji} - similarité {similarite_pct:.0f}%")
    else:
        print(f"\n   📍 {ville_test} : Commune non trouvée")

# -------------------------------------------------------------------
# 7. RECOMMANDATION POUR INVESTISSEMENT
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("💡 7. Recommandation stratégique pour investissement")
print("=" * 60)

# Trouver les communes avec meilleur potentiel d'investissement
if 'potentiel_investissement' in df_clean.columns and 'classe_risque' in df_clean.columns:
    meilleures = df_clean.nlargest(10, 'potentiel_investissement')
    meilleures_modere = meilleures[meilleures['classe_risque'] == 'MODERE']
    
    if len(meilleures_modere) > 0:
        print("\n   🟢 TOP 5 COMMUNES PRIORITAIRES POUR INVESTISSEMENT :")
        for i, (idx, commune) in enumerate(meilleures_modere.head(5).iterrows()):
            # Score de risque (si disponible)
            score_risque = commune.get('risque_credit_local', None)
            score_aff = f"{score_risque:.3f}" if score_risque is not None else "N/A"
            
            print(f"      {i+1}. {commune['ville']} ({commune['code_dept']})")
            print(f"         💰 Potentiel : {commune['potentiel_investissement']:.2f}/1.00")
            print(f"         📊 Score risque : {score_aff}")
    else:
        print("\n   ⚠️ Aucune commune MODERE trouvée avec fort potentiel")

print("\n" + "=" * 60)
print("🎉 MODÈLE 3 TERMINÉ !")
print("=" * 60)

print("\n🤖 CE MODÈLE PERMET AU CHATBOT DE :")
print("   ✅ Trouver des communes similaires pour comparaison")
print("   ✅ Recommander des zones d'investissement similaires")
print("   ✅ Identifier des benchmarks territoriaux")
print("   ✅ Proposer des alternatives aux investisseurs")