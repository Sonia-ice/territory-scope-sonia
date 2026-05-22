"""
ICEBERG - MODÈLE 2 : PRÉDICTION DU SCORE DE RISQUE
Prédit le risque_credit_local (score continu 0-1)
Pour simuler l'impact des changements territoriaux
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import glob

print("=" * 60)
print("🧊 ICEBERG - MODÈLE 2 : PRÉDICTION SCORE RISQUE")
print("   Pour simuler l'impact des changements territoriaux")
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
# 2. CIBLE : risque_credit_local (score à prédire)
# -------------------------------------------------------------------
print("\n🎯 2. Cible : risque_credit_local (score 0-1)")

colonne_cible = 'risque_credit_local'
if colonne_cible not in df.columns:
    print(f"   ❌ Colonne '{colonne_cible}' non trouvée !")
    print("   📋 Colonnes disponibles :", df.columns.tolist())
    exit()

print(f"   ✅ Cible : {colonne_cible}")
print(f"   📊 Statistiques :")
print(f"      Min : {df[colonne_cible].min():.3f}")
print(f"      Max : {df[colonne_cible].max():.3f}")
print(f"      Moyenne : {df[colonne_cible].mean():.3f}")
print(f"      Écart-type : {df[colonne_cible].std():.3f}")

# -------------------------------------------------------------------
# 3. FEATURES STRATÉGIQUES (mêmes que modèle 1)
# -------------------------------------------------------------------
print("\n🔧 3. Préparation des features stratégiques...")

features = [
    "population",
    "densite_hab_km2",
    "taux_chomage",
    "revenu_median",
    "taux_pauvrete",
    "nb_entreprises_actives",
    "entreprises_1000hab",
    "taux_survie_entreprises",
    "dynamique_eco",
    "score_fragilite",
    "score_emergence",
    "score_momentum",
    "score_freins_invisibles",
    "potentiel_investissement",
    "nb_gares",
    "prix_m2_median"
]

features_disponibles = [f for f in features if f in df.columns]
print(f"   ✅ {len(features_disponibles)} features disponibles")

# Nettoyer
df_clean = df.dropna(subset=features_disponibles + [colonne_cible])
print(f"   📊 {len(df_clean)} communes analysables")

X = df_clean[features_disponibles]
y = df_clean[colonne_cible]

# -------------------------------------------------------------------
# 4. NORMALISATION
# -------------------------------------------------------------------
print("\n📐 4. Normalisation...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   ✅ Normalisation terminée")

# -------------------------------------------------------------------
# 5. SÉPARATION TRAIN/TEST
# -------------------------------------------------------------------
print("\n✂️ 5. Séparation entraînement/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"   📚 Entraînement : {len(X_train)} communes")
print(f"   🧪 Test : {len(X_test)} communes")

# -------------------------------------------------------------------
# 6. ENTRAÎNEMENT
# -------------------------------------------------------------------
print("\n🤖 6. Entraînement du modèle...")
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)
print("   ✅ Modèle entraîné !")

# -------------------------------------------------------------------
# 7. ÉVALUATION
# -------------------------------------------------------------------
print("\n📊 7. Évaluation du modèle...")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"   📈 Erreur absolue moyenne (MAE) : {mae:.4f}")
print(f"   📉 Racine erreur quadratique (RMSE) : {rmse:.4f}")
print(f"   🎯 Score R² : {r2:.3f}")

# Validation croisée
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print(f"\n   🔄 Validation croisée (5 folds) :")
print(f"      R² moyen : {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# -------------------------------------------------------------------
# 8. FEATURES IMPORTANTES
# -------------------------------------------------------------------
print("\n🔍 8. Features les plus importantes :")
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

for i in range(min(10, len(features_disponibles))):
    idx = sorted_idx[i]
    print(f"   {i+1}. {features_disponibles[idx]} : {importances[idx]*100:.1f}%")

# -------------------------------------------------------------------
# 9. SAUVEGARDE
# -------------------------------------------------------------------
print("\n💾 9. Sauvegarde...")
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/predictor_risk_score.pkl")
joblib.dump(scaler, "models/scaler_predictor.pkl")
joblib.dump(features_disponibles, "models/features_predictor.pkl")

print("   ✅ Modèle sauvegardé : models/predictor_risk_score.pkl")

# -------------------------------------------------------------------
# 10. TESTS ET SIMULATIONS POUR LE CHATBOT
# -------------------------------------------------------------------
print("\n🧪 10. Tests et simulations...")

print("\n   📍 Prédictions sur des communes :")
test_communes = df_clean.sample(5)

for idx, commune in test_communes.iterrows():
    X_test_commune = [commune[f] for f in features_disponibles]
    X_test_scaled = scaler.transform([X_test_commune])
    
    prediction = model.predict(X_test_scaled)[0]
    vrai = commune[colonne_cible]
    erreur = abs(prediction - vrai)
    
    # Classe de risque
    if prediction < 0.5:
        classe = "🟢 MODERE"
    elif prediction < 0.7:
        classe = "🟠 ELEVE"
    else:
        classe = "🔴 CRITIQUE"
    
    print(f"\n      📍 {commune['ville']}")
    print(f"         Score réel : {vrai:.3f}")
    print(f"         Score prédit : {prediction:.3f} (±{erreur:.3f})")
    print(f"         Classe : {classe}")

print("\n   🔮 Simulation d'impact (exemple) :")
print("      Si une commune voit son taux de chômage diminuer de 2%,")
print("      le score de risque peut baisser de 0.05 à 0.10 points.")
print("      → Le chatbot pourra faire ces simulations en temps réel !")

print("\n" + "=" * 60)
print("🎉 MODÈLE 2 TERMINÉ !")
print("=" * 60)

print("\n🤖 CE MODÈLE PERMET AU CHATBOT DE :")
print("   ✅ Prédire le score de risque d'une nouvelle commune")
print("   ✅ Simuler l'impact de changements (ex: +1 gare, -2% chômage)")
print("   ✅ Comparer le risque prédit vs réel")
print("   ✅ Générer des alertes automatiques sur les zones à risque")