"""
ICEBERG - MODÈLE 1 : CLASSIFICATION DU RISQUE (VERSION PRO)
Avec indicateurs stratégiques pour l'aide à la décision
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import glob

print("=" * 60)
print("🧊 ICEBERG - MODÈLE 1 : CLASSIFICATION DU RISQUE")
print("   Version Pro - Indicateurs stratégiques")
print("=" * 60)

# -------------------------------------------------------------------
# 1. CHARGER LES DONNÉES
# -------------------------------------------------------------------
print("\n📂 1. Chargement des données...")

csv_files = glob.glob("*.csv")
fichier_csv = csv_files[0] if csv_files else None
if not fichier_csv:
    print("   ❌ Aucun fichier CSV trouvé !")
    exit()

df = pd.read_csv(fichier_csv, encoding="utf-8")
print(f"   ✅ {len(df)} communes chargées")

# -------------------------------------------------------------------
# 2. INDICATEURS STRATÉGIQUES POUR LE PROJET
# -------------------------------------------------------------------
print("\n🎯 2. Indicateurs stratégiques ICEBERG...")

# Vérifier les indicateurs disponibles
indicateurs_cles = {
    "potentiel_investissement": "Score d'attractivité (0-1)",
    "dynamique_eco": "Croissance économique (%)",
    "taux_survie_entreprises": "Pérennité des entreprises",
    "score_momentum": "Dynamique territoriale",
    "risque_credit_local": "Risque financier (cible)",
    "score_freins_invisibles": "Freins structurels",
    "entreprises_1000hab": "Vitalité entrepreneuriale"
}

print("\n   📋 Indicateurs stratégiques disponibles :")
for col, desc in indicateurs_cles.items():
    if col in df.columns:
        print(f"      ✅ {col} : {desc}")
    else:
        print(f"      ❌ {col} : {desc} (à créer)")

# -------------------------------------------------------------------
# 3. PRÉPARER LES FEATURES (TOUS LES INDICATEURS)
# -------------------------------------------------------------------
print("\n🔧 3. Préparation des features stratégiques...")

features = [
    # Données démographiques
    "population",
    "densite_hab_km2",
    
    # Données économiques
    "taux_chomage",
    "revenu_median",
    "taux_pauvrete",
    
    # Données entrepreneuriales
    "nb_entreprises_actives",
    "entreprises_1000hab",
    "taux_survie_entreprises",
    "dynamique_eco",
    
    # Scores ICEBERG
    "score_fragilite",
    "score_emergence",
    "score_momentum",
    "score_freins_invisibles",
    "potentiel_investissement",
    
    # Infrastructures
    "nb_gares",
    "prix_m2_median"
]

features_disponibles = [f for f in features if f in df.columns]
print(f"   ✅ {len(features_disponibles)} indicateurs stratégiques chargés")

# Afficher les indicateurs manquants
manquants = [f for f in features if f not in df.columns]
if manquants:
    print(f"   ⚠️ Indicateurs manquants : {manquants}")

# -------------------------------------------------------------------
# 4. CIBLES : CLASSE DE RISQUE
# -------------------------------------------------------------------
print("\n🎯 4. Cible : classe_risque (MODERE/ELEVE/CRITIQUE)")

colonne_cible = 'classe_risque'
if colonne_cible not in df.columns:
    print(f"   ❌ Colonne '{colonne_cible}' non trouvée !")
    exit()

print(f"   ✅ Cible : {colonne_cible}")

# Distribution
print(f"\n   📊 Distribution des classes :")
for valeur in sorted(df[colonne_cible].unique()):
    count = sum(df[colonne_cible] == valeur)
    pct = count / len(df) * 100
    print(f"      {valeur} : {count} communes ({pct:.1f}%)")

# Encodage
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df[colonne_cible])

# -------------------------------------------------------------------
# 5. NETTOYAGE
# -------------------------------------------------------------------
print("\n🧹 5. Nettoyage des données...")
df_clean = df.dropna(subset=features_disponibles + [colonne_cible])
y_clean = y_encoded[df.index.isin(df_clean.index)]
print(f"   📊 {len(df_clean)} communes analysables")

X = df_clean[features_disponibles]
y = y_clean

# -------------------------------------------------------------------
# 6. NORMALISATION
# -------------------------------------------------------------------
print("\n📐 6. Normalisation...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   ✅ Normalisation terminée")

# -------------------------------------------------------------------
# 7. SÉPARATION TRAIN/TEST
# -------------------------------------------------------------------
print("\n✂️ 7. Séparation entraînement/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   📚 Entraînement : {len(X_train)} communes")
print(f"   🧪 Test : {len(X_test)} communes")

# -------------------------------------------------------------------
# 8. ENTRAÎNEMENT
# -------------------------------------------------------------------
print("\n🤖 8. Entraînement du modèle...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)
print("   ✅ Modèle entraîné !")

# -------------------------------------------------------------------
# 9. ÉVALUATION
# -------------------------------------------------------------------
print("\n📊 9. Évaluation du modèle...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   🎯 Précision globale : {accuracy * 100:.1f}%")

print(f"\n   📋 Rapport par classe :")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# -------------------------------------------------------------------
# 10. ANALYSE STRATÉGIQUE POUR LE CHATBOT
# -------------------------------------------------------------------
print("\n💡 10. Analyse stratégique pour le chatbot...")

# Top 5 features les plus importantes
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\n   🔑 Indicateurs clés pour la décision :")
for i in range(min(5, len(features_disponibles))):
    idx = sorted_idx[i]
    print(f"      {i+1}. {features_disponibles[idx]} : {importances[idx]*100:.1f}%")

# -------------------------------------------------------------------
# 11. SAUVEGARDE
# -------------------------------------------------------------------
print("\n💾 11. Sauvegarde des modèles...")
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/classifier_risque_pro.pkl")
joblib.dump(scaler, "models/scaler_pro.pkl")
joblib.dump(label_encoder, "models/label_encoder_pro.pkl")
joblib.dump(features_disponibles, "models/features_pro.pkl")

print("   ✅ Modèle sauvegardé : models/classifier_risque_pro.pkl")

# -------------------------------------------------------------------
# 12. TEST DÉCISIONNEL
# -------------------------------------------------------------------
print("\n🧪 12. Simulation de décision d'investissement :")

# Prendre des communes de chaque classe
for classe in label_encoder.classes_:
    communes_classe = df_clean[df_clean[colonne_cible] == classe]
    if len(communes_classe) > 0:
        exemple = communes_classe.iloc[0]
        print(f"\n   📍 {exemple['ville']} ({classe}) :")
        
        # Indicateurs stratégiques
        if 'potentiel_investissement' in features_disponibles:
            potentiel = exemple['potentiel_investissement']
            print(f"      💰 Potentiel investissement : {potentiel:.2f}/1.00")
        
        if 'dynamique_eco' in features_disponibles:
            dynamique = exemple['dynamique_eco']
            print(f"      📈 Dynamique économique : {dynamique:.1f}%")
        
        if 'score_momentum' in features_disponibles:
            momentum = exemple['score_momentum']
            print(f"      🚀 Momentum : {momentum:.2f}/1.00")
        
        # Recommandation
        if classe == "CRITIQUE":
            print(f"      ⚠️ RECO : Surveillance renforcée / Financement cautionné")
        elif classe == "ELEVE":
            print(f"      🟠 RECO : Analyse approfondie avant financement")
        else:
            print(f"      🟢 RECO : Zone prioritaire pour l'investissement")

print("\n" + "=" * 60)
print("🎉 MODÈLE 1 PRO TERMINÉ !")
print("=" * 60)

print("\n📊 RÉCAPITULATIF POUR LE CHATBOT :")
print("   - 3 niveaux de risque : CRITIQUE / ELEVE / MODERE")
print("   - Précision globale : {:.1f}%".format(accuracy * 100))
print("   - Indicateurs clés : potentiel_investissement, dynamique_eco, score_momentum")
print("\n🤖 LE CHATBOT POURRA DÉCIDER :")
print("   - Où orienter les financements en priorité")
print("   - Quelles zones nécessitent une surveillance renforcée")
print("   - Quelles sont les communes à fort potentiel")