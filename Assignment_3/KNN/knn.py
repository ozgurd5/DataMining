import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
import random

# Hedefler
hedef_horsepower = 130
hedef_acceleration = 13
hedef_weight = 3500

# Veri setini yükle
df = sns.load_dataset("mpg")
df = df.dropna(subset=["horsepower"])  # eksik değerleri at

# Özellik ve hedef seçimi
ozellik = df[["horsepower", "acceleration", "weight"]]
hedef = df["mpg"]

# Eğitim-test ayrımı
random_state = random.randint(0, 999)
print("Random state:", random_state)
ozellik_egitim, ozellik_test, hedef_egitim, hedef_test = train_test_split(ozellik, hedef, test_size=0.1, random_state=random_state)

# Normalizasyon
min_max_normalizator = MinMaxScaler()
ozellik_egitim_normalize = min_max_normalizator.fit_transform(ozellik_egitim)
ozellik_test_normalize = min_max_normalizator.transform(ozellik_test)

# MSE = Mean Squared Error
# MAE = Mean Absolute Error

# K sayısının belirlenmesi
k_degerleri = range(1, 100, 2) # Sadece tek sayılar
cross_validation_katları = range(2, 100)
cross_validation_df = pd.DataFrame(columns=["k", "n", "r_kare", "mse", "mae"])

for k in k_degerleri:
    for n in cross_validation_katları:
        # Mesafeye göre ağırlıklı KNN regresyon modeli
        knn_modeli = KNeighborsRegressor(n_neighbors=k, weights='distance')

        # Cross-validation ile her k için ortalama R kare skoru
        r_kare_skorları = cross_val_score(knn_modeli, ozellik_egitim_normalize, hedef_egitim, cv=n, scoring=make_scorer(r2_score, greater_is_better=True))
        r_kare = r_kare_skorları.mean()

        # Cross-validation ile her k için ortalama MSE skoru
        mse_skorları = cross_val_score(knn_modeli, ozellik_egitim_normalize, hedef_egitim, cv=n, scoring=make_scorer(mean_squared_error, greater_is_better=False))
        mse = -1 * mse_skorları.mean()

        # Cross-validation ile her k için ortalama MAE skoru
        mae_skorları = cross_val_score(knn_modeli, ozellik_egitim_normalize, hedef_egitim, cv=n, scoring=make_scorer(mean_absolute_error, greater_is_better=False))
        mae = -1 * mae_skorları.mean()

        # Sonuçları dataframe'e ekle
        cross_validation_df = cross_validation_df._append({"k": k, "n": n, "r_kare": r_kare, "mse": mse, "mae": mae}, ignore_index=True)
        print("k:", k, "n:", n, "R Kare:", r_kare, "MSE:", mse, "MAE:", mae)

# Farklı cross validation katları ve R kare yöntemine göre en iyi k'yı bulma
en_iyi_r_kare = cross_validation_df["r_kare"].max()
en_iyi_r_kare_k = cross_validation_df[cross_validation_df["r_kare"] == en_iyi_r_kare]["k"].values[0]
en_iyi_r_kare_n = cross_validation_df[cross_validation_df["r_kare"] == en_iyi_r_kare]["n"].values[0]
print("En iyi R Kare:", en_iyi_r_kare, "En iyi k:", en_iyi_r_kare_k, "En iyi n:", en_iyi_r_kare_n)

# Farklı cross validation katları ve MSE yöntemine göre en iyi k'yı bulma
en_iyi_mse = cross_validation_df["mse"].min()
en_iyi_mse_k = cross_validation_df[cross_validation_df["mse"] == en_iyi_mse]["k"].values[0]
en_iyi_mse_n = cross_validation_df[cross_validation_df["mse"] == en_iyi_mse]["n"].values[0]
print("En iyi MSE:", en_iyi_mse, "En iyi k:", en_iyi_mse_k, "En iyi n:", en_iyi_mse_n)

# Farklı cross validation katları ve MAE yöntemine göre en iyi k'yı bulma
en_iyi_mae = cross_validation_df["mae"].min()
en_iyi_mae_k = cross_validation_df[cross_validation_df["mae"] == en_iyi_mae]["k"].values[0]
en_iyi_mae_n = cross_validation_df[cross_validation_df["mae"] == en_iyi_mae]["n"].values[0]
print("En iyi MAE:", en_iyi_mae, "En iyi k:", en_iyi_mae_k, "En iyi n:", en_iyi_mae_n)

# R kare skorları, n ve k değerlerini görselleştirme
plt.figure(figsize=(8,4))
sns.lineplot(data=cross_validation_df, x="k", y="r_kare", hue="n")
plt.title("R Kare Skorları")
plt.grid(True)
plt.savefig("r_kare.png", dpi=1000)
plt.show()