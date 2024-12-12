import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
import random

# Veri setini yükle
df = sns.load_dataset("mpg")
df = df.dropna(subset=["horsepower"])  # eksik değerleri at

# Özellik ve hedef seçimi
ozellik = df[["horsepower", "acceleration", "weight"]]
hedef = df["mpg"]

# Eğitim-test ayrımı
random_state = 300 #random.randint(0, 999)
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
cross_validation_df = pd.DataFrame(columns=["k", "r_kare", "mse", "mae"])

for k in k_degerleri:
    # Mesafeye göre ağırlıklı KNN regresyon modeli
    knn_modeli = KNeighborsRegressor(n_neighbors=k, weights='distance')

    # Cross-validation ile her k için ortalama R kare skoru
    r_kare_skorları = cross_val_score(knn_modeli, ozellik_egitim_normalize, hedef_egitim, cv=5, scoring=make_scorer(r2_score, greater_is_better=True))
    r_kare = r_kare_skorları.mean()

    # Cross-validation ile her k için ortalama MSE skoru
    mse_skorları = cross_val_score(knn_modeli, ozellik_egitim_normalize, hedef_egitim, cv=5, scoring=make_scorer(mean_squared_error, greater_is_better=False))
    mse = -1 * mse_skorları.mean()

    # Cross-validation ile her k için ortalama MAE skoru
    mae_skorları = cross_val_score(knn_modeli, ozellik_egitim_normalize, hedef_egitim, cv=5, scoring=make_scorer(mean_absolute_error, greater_is_better=False))
    mae = -1 * mae_skorları.mean()

    # Sonuçları dataframe'e ekle
    cross_validation_df = cross_validation_df._append({"k": k, "r_kare": r_kare, "mse": mse, "mae": mae}, ignore_index=True)
    print("k:", k, "R Kare:", r_kare, "MSE:", mse, "MAE:", mae)

# R kare yöntemine göre en iyi k'yı bulma
en_iyi_r_kare = cross_validation_df["r_kare"].max()
en_iyi_r_kare_k = cross_validation_df[cross_validation_df["r_kare"] == en_iyi_r_kare]["k"].values[0]
print("En iyi R Kare:", en_iyi_r_kare, "En iyi k:", en_iyi_r_kare_k)

# MSE yöntemine göre en iyi k'yı bulma
en_iyi_mse = cross_validation_df["mse"].min()
en_iyi_mse_k = cross_validation_df[cross_validation_df["mse"] == en_iyi_mse]["k"].values[0]
print("En iyi MSE:", en_iyi_mse, "En iyi k:", en_iyi_mse_k)

# MAE yöntemine göre en iyi k'yı bulma
en_iyi_mae = cross_validation_df["mae"].min()
en_iyi_mae_k = cross_validation_df[cross_validation_df["mae"] == en_iyi_mae]["k"].values[0]
print("En iyi MAE:", en_iyi_mae, "En iyi k:", en_iyi_mae_k)

# Grafik boyutlarını ayarla
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# R Kare grafiği
axes[0].plot(cross_validation_df["k"], cross_validation_df["r_kare"], label="R Kare")
axes[0].set_xlabel("k")
axes[0].set_ylabel("R Kare")
axes[0].set_title("R Kare'ye Göre KNN Regresyon Modeli Performansı")
axes[0].legend()
axes[0].grid(True)

# MSE grafiği
axes[1].plot(cross_validation_df["k"], cross_validation_df["mse"], label="MSE")
axes[1].set_xlabel("k")
axes[1].set_ylabel("MSE")
axes[1].set_title("MSE'ye Göre KNN Regresyon Modeli Performansı")
axes[1].legend()
axes[1].grid(True)

# MAE grafiği
axes[2].plot(cross_validation_df["k"], cross_validation_df["mae"], label="MAE")
axes[2].set_xlabel("k")
axes[2].set_ylabel("MAE")
axes[2].set_title("MAE'ye Göre KNN Regresyon Modeli Performansı")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("knn.png")
plt.show()