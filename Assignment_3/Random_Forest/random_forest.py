import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
import random

# Veri setini yükle
df = sns.load_dataset("mpg")
df = df.dropna(subset=["horsepower"])

# Özellik ve hedef seçimi
ozellik = df[["horsepower", "acceleration", "weight"]]
hedef = df["mpg"]

# Eğitim-test ayrımı
random_state = 300 #random.randint(0, 999)
ozellik_egitim, ozellik_test, hedef_egitim, hedef_test = train_test_split(ozellik, hedef, test_size=0.1, random_state=random_state)

# n_estimators ve max_depth değerlerinin belirlenmesi
n_estimators_candidates = [50, 100, 200]
max_depth_candidates = [5, 10]

# Sonuçları saklamak için bir DataFrame
cross_validation_sonuçlar = pd.DataFrame(columns=["n_estimators", "max_depth", "r_kare", "mse", "mae"])

# Scorerlar
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
r_kare_scorer = make_scorer(r2_score, greater_is_better=True)

# Cross-validation ve sonuçların kaydedilmesi
for n_est in n_estimators_candidates:
    for m_depth in max_depth_candidates:
        # Model oluştur
        random_forest_model = RandomForestRegressor(n_estimators=n_est, max_depth=m_depth, random_state=random_state)

        # R Kare
        r_kare_skorları = cross_val_score(random_forest_model, ozellik_egitim, hedef_egitim, cv=5, scoring=r_kare_scorer)
        r_kare = r_kare_skorları.mean()

        # MSE
        mse_skorları = cross_val_score(random_forest_model, ozellik_egitim, hedef_egitim, cv=5, scoring=mse_scorer)
        mse = (-1) * mse_skorları.mean()

        # MAE
        mae_skorları = cross_val_score(random_forest_model, ozellik_egitim, hedef_egitim, cv=5, scoring=mae_scorer)
        mae = (-1) * mae_skorları.mean()

        # Sonucu tabloya ekle
        cross_validation_sonuçlar = cross_validation_sonuçlar._append({
            "n_estimators": n_est,
            "max_depth": m_depth,
            "r_kare": r_kare,
            "mse": mse,
            "mae": mae
        }, ignore_index=True)

        print("n_estimators:", n_est, "max_depth:", m_depth, "R Kare:", r_kare, "MSE:", mse, "MAE:", mae)

# R kare yöntemine göre en iyi n_estimators ve max_depth'u bulma
en_iyi_r_kare = cross_validation_sonuçlar["r_kare"].max()
en_iyi_r_kare_n_estimators = cross_validation_sonuçlar.loc[cross_validation_sonuçlar["r_kare"] == en_iyi_r_kare].iloc[0]["n_estimators"]
en_iyi_r_kare_max_depth = cross_validation_sonuçlar.loc[cross_validation_sonuçlar["r_kare"] == en_iyi_r_kare].iloc[0]["max_depth"]
print("En iyi R Kare:", en_iyi_r_kare, "En iyi n_estimators:", en_iyi_r_kare_n_estimators, "En iyi max_depth:", en_iyi_r_kare_max_depth)

# MSE yöntemine göre en iyi n_estimators ve max_depth'u bulma
en_iyi_mse = cross_validation_sonuçlar["mse"].min()
en_iyi_mse_n_estimators = cross_validation_sonuçlar.loc[cross_validation_sonuçlar["mse"] == en_iyi_mse].iloc[0]["n_estimators"]
en_iyi_mse_max_depth = cross_validation_sonuçlar.loc[cross_validation_sonuçlar["mse"] == en_iyi_mse].iloc[0]["max_depth"]
print("En iyi MSE:", en_iyi_mse, "En iyi n_estimators:", en_iyi_mse_n_estimators, "En iyi max_depth:", en_iyi_mse_max_depth)

# MAE yöntemine göre en iyi n_estimators ve max_depth'u bulma
en_iyi_mae = cross_validation_sonuçlar["mae"].min()
en_iyi_mae_n_estimators = cross_validation_sonuçlar.loc[cross_validation_sonuçlar["mae"] == en_iyi_mae].iloc[0]["n_estimators"]
en_iyi_mae_max_depth = cross_validation_sonuçlar.loc[cross_validation_sonuçlar["mae"] == en_iyi_mae].iloc[0]["max_depth"]
print("En iyi MAE:", en_iyi_mae, "En iyi n_estimators:", en_iyi_mae_n_estimators, "En iyi max_depth:", en_iyi_mae_max_depth)

# n_estimators ve max_depth değerlerinin skor grafikleri
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# R Kare grafiği
axes[0].plot(cross_validation_sonuçlar["n_estimators"], cross_validation_sonuçlar["r_kare"], label="R Kare")
axes[0].set_xlabel("n_estimators")
axes[0].set_ylabel("R Kare")
axes[0].set_title("R Kare'ye Göre Random Forest Regresyon Modeli Performansı")
axes[0].legend()
axes[0].grid(True)

# MSE grafiği
axes[1].plot(cross_validation_sonuçlar["n_estimators"], cross_validation_sonuçlar["mse"], label="MSE")
axes[1].set_xlabel("n_estimators")
axes[1].set_ylabel("MSE")
axes[1].set_title("MSE'ye Göre Random Forest Regresyon Modeli Performansı")
axes[1].legend()
axes[1].grid(True)

# MAE grafiği
axes[2].plot(cross_validation_sonuçlar["n_estimators"], cross_validation_sonuçlar["mae"], label="MAE")
axes[2].set_xlabel("n_estimators")
axes[2].set_ylabel("MAE")
axes[2].set_title("MAE'ye Göre Random Forest Regresyon Modeli Performansı")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("random_forest.png")
plt.show()