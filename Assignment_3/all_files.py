import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Seaborn'un mpg veri setini yükle
df = sns.load_dataset("mpg")

# Veri setinde eksik değer tespiti
print("\nEksik değerlerin sayısı:")
print(df.isnull().sum())

# 398 veriden 6 tanesi eksik olduğu için eksik verilerin kaybı kabul edilebilir.
df = df.dropna(subset=["horsepower"])

# Temel istatistiksel özet,
pd.set_option('display.max_rows', 12) # Max 12 satır göster
pd.set_option('display.max_columns', 12) # Max 12 sütun göster
print("\nVeri seti istatistiksel özeti:")
print(df.describe())

# Horsepower Dağılımı
plt.figure(figsize=(10,10))
sns.histplot(df["horsepower"], kde=True)
plt.title("Horsepower Dağılımı")
plt.savefig("horsepower.png")
plt.show()

# Acceleration Dağılımı
plt.figure(figsize=(10,10))
sns.histplot(df["acceleration"], kde=True, color="orange")
plt.title("Acceleration Dağılımı")
plt.savefig("acceleration.png")
plt.show()

# Weight Dağılımı
plt.figure(figsize=(10,10))
sns.histplot(df["weight"], kde=True, color="green")
plt.title("Weight Dağılımı")
plt.savefig("weight.png")
plt.show()

# MPG Dağılımı
plt.figure()
sns.histplot(df["mpg"], kde=True, color="red")
plt.title("MPG Dağılımı")
plt.savefig("mpg.png")
plt.show()

# Burada ilgili değişkenleri bir arada görerek ilişkilerini anlayabiliriz.
sns.pairplot(df[["mpg", "horsepower", "acceleration", "weight"]], diag_kind="kde")
plt.savefig("pairplot.png")
plt.show()

# Korelasyon matrisi
corr = df[["mpg", "horsepower", "acceleration", "weight"]].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Korelasyon Isı Haritası")
plt.savefig("correlation.png")
plt.show()

# Scatter plotlar için değerleri sırala
df["horsepower"] = df["horsepower"].sort_values().values
df["acceleration"] = df["acceleration"].sort_values().values
df["weight"] = df["weight"].sort_values().values
df["mpg"] = df["mpg"].sort_values().values

# Scatter plotlar
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Horsepower scatter plot
axs[0, 0].scatter(df.index, df["horsepower"], color='blue')
axs[0, 0].set_title("Horsepower Dağılımı")
axs[0, 0].set_xlabel("Index")
axs[0, 0].set_ylabel("Horsepower")

# Acceleration scatter plot
axs[0, 1].scatter(df.index, df["acceleration"], color='orange')
axs[0, 1].set_title("Acceleration Dağılımı")
axs[0, 1].set_xlabel("Index")
axs[0, 1].set_ylabel("Acceleration")

# Weight scatter plot
axs[1, 0].scatter(df.index, df["weight"], color='green')
axs[1, 0].set_title("Weight Dağılımı")
axs[1, 0].set_xlabel("Index")
axs[1, 0].set_ylabel("Weight")

# MPG scatter plot
axs[1, 1].scatter(df.index, df["mpg"], color='red')
axs[1, 1].set_title("MPG Dağılımı")
axs[1, 1].set_xlabel("Index")
axs[1, 1].set_ylabel("MPG")

# Layout düzenlemesi ve grafiğin kaydedilmesi
plt.tight_layout()
plt.savefig("scatter_plots.png")
plt.show()

# Boxplotlar ve outlier tespiti
özellikler = ["mpg", "horsepower", "acceleration", "weight"]

plt.figure(figsize=(10,8))
for i, feature in enumerate(özellikler, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[feature], color='skyblue', whis=1.5)
    plt.title(f"{feature.capitalize()} Boxplot")
plt.tight_layout()
plt.savefig("boxplots.png")
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
import random

# MSE = Mean Squared Error
# MAE = Mean Absolute Error

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

# K sayısının belirlenmesi
k_deger_adaylari = range(1, 100, 2) # Sadece tek sayılar
cross_validation_df = pd.DataFrame(columns=["k", "r_kare", "mse", "mae"])

for k in k_deger_adaylari:
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

# K değerlerinin skor grafikleri
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

# Tahmin Hedefleri
tahmin_hedef_horsepower = 130
tahmin_hedef_acceleration = 13
tahmin_hedef_weight = 3500

# En iyi R Kare sonucunu veren k değeri ile model oluşturma
k_final = int(en_iyi_r_kare_k)
print(f"\nEn iyi k (R Kare bazlı) ile model oluşturuluyor... k={k_final}")

final_knn_model = KNeighborsRegressor(n_neighbors=k_final, weights='distance')
final_knn_model.fit(ozellik_egitim_normalize, hedef_egitim)

# Yeni değerler için tahmin
araba_df = pd.DataFrame([[tahmin_hedef_horsepower, tahmin_hedef_acceleration, tahmin_hedef_weight]], columns=ozellik_egitim.columns)
araba_df_normalize = min_max_normalizator.transform(araba_df)

tahmin_mpg = final_knn_model.predict(araba_df_normalize)
print(f"Horsepower={tahmin_hedef_horsepower}, Acceleration={tahmin_hedef_acceleration}, Weight={tahmin_hedef_weight} için tahmin edilen MPG: {tahmin_mpg[0]:.2f}")

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

# Hiperparametre adayları
n_estimators_adayları = range(50, 501, 50)
max_depth_adayları = range(1, 11)

# Cross-validation sonuçlarını saklamak için DataFrame
cross_validation_sonuçlar = pd.DataFrame(columns=["n_estimators", "max_depth", "r_kare", "mse", "mae"])

# Skorlayıcılar
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
r_kare_scorer = make_scorer(r2_score, greater_is_better=True)

# Cross-validation ile sonuçların hesaplanması
for n_estimators in n_estimators_adayları:
    for max_depth in max_depth_adayları:
        random_forest_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

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
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "r_kare": r_kare,
            "mse": mse,
            "mae": mae
        }, ignore_index=True)

        print("n_estimators:", n_estimators, "max_depth:", max_depth, "R Kare:", r_kare, "MSE:", mse, "MAE:", mae)

# R kare yöntemine göre en iyi n_estimators ve max_depth'u bulma
en_iyi_r_kare = cross_validation_sonuçlar["r_kare"].max()
en_iyi_r_kare_satir = cross_validation_sonuçlar[cross_validation_sonuçlar["r_kare"] == en_iyi_r_kare].iloc[0]
en_iyi_r_kare_n_estimators = en_iyi_r_kare_satir["n_estimators"]
en_iyi_r_kare_max_depth = en_iyi_r_kare_satir["max_depth"]
print("En iyi R Kare:", en_iyi_r_kare, "En iyi n_estimators:", en_iyi_r_kare_n_estimators, "En iyi max_depth:", en_iyi_r_kare_max_depth)

# MSE yöntemine göre en iyi n_estimators ve max_depth'u bulma
en_iyi_mse = cross_validation_sonuçlar["mse"].min()
en_iyi_mse_satir = cross_validation_sonuçlar[cross_validation_sonuçlar["mse"] == en_iyi_mse].iloc[0]
en_iyi_mse_n_estimators = en_iyi_mse_satir["n_estimators"]
en_iyi_mse_max_depth = en_iyi_mse_satir["max_depth"]
print("En iyi MSE:", en_iyi_mse, "En iyi n_estimators:", en_iyi_mse_n_estimators, "En iyi max_depth:", en_iyi_mse_max_depth)

# MAE yöntemine göre en iyi n_estimators ve max_depth'u bulma
en_iyi_mae = cross_validation_sonuçlar["mae"].min()
en_iyi_mae_satir = cross_validation_sonuçlar[cross_validation_sonuçlar["mae"] == en_iyi_mae].iloc[0]
en_iyi_mae_n_estimators = en_iyi_mae_satir["n_estimators"]
en_iyi_mae_max_depth = en_iyi_mae_satir["max_depth"]
print("En iyi MAE:", en_iyi_mae, "En iyi n_estimators:", en_iyi_mae_n_estimators, "En iyi max_depth:", en_iyi_mae_max_depth)

# Grafikler: her bir max_depth için ayrı çizgi
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# R Kare grafiği (her max_depth farklı çizgi)
for max_depth in max_depth_adayları:
    subset = cross_validation_sonuçlar[cross_validation_sonuçlar["max_depth"] == max_depth]
    axes[0].plot(subset["n_estimators"], subset["r_kare"], marker='o', label=f"max_depth={max_depth}")
axes[0].set_xlabel("n_estimators")
axes[0].set_ylabel("R Kare")
axes[0].set_title("R Kare'ye Göre Random Forest Regresyon Modeli Performansı")
axes[0].legend()
axes[0].grid(True)

# MSE grafiği (her max_depth farklı çizgi)
for max_depth in max_depth_adayları:
    subset = cross_validation_sonuçlar[cross_validation_sonuçlar["max_depth"] == max_depth]
    axes[1].plot(subset["n_estimators"], subset["mse"], marker='o', label=f"max_depth={max_depth}")
axes[1].set_xlabel("n_estimators")
axes[1].set_ylabel("MSE")
axes[1].set_title("MSE'ye Göre Random Forest Regresyon Modeli Performansı")
axes[1].legend()
axes[1].grid(True)

# MAE grafiği (her max_depth farklı çizgi)
for max_depth in max_depth_adayları:
    subset = cross_validation_sonuçlar[cross_validation_sonuçlar["max_depth"] == max_depth]
    axes[2].plot(subset["n_estimators"], subset["mae"], marker='o', label=f"max_depth={max_depth}")
axes[2].set_xlabel("n_estimators")
axes[2].set_ylabel("MAE")
axes[2].set_title("MAE'ye Göre Random Forest Regresyon Modeli Performansı")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("random_forest.png")
plt.show()

# Tahmin Hedefleri
tahmin_hedef_horsepower = 130
tahmin_hedef_acceleration = 13
tahmin_hedef_weight = 3500

# En iyi R Kare sonucunu veren n_estimators ve max_depth değerleri ile model oluşturma
en_iyi_n_estimators_r_kare = int(en_iyi_r_kare_n_estimators)
en_iyi_max_depth_r_kare = int(en_iyi_r_kare_max_depth)
print(f"\nEn iyi R Kare sonucunu veren model oluşturuluyor... n_estimators={en_iyi_n_estimators_r_kare}, max_depth={en_iyi_max_depth_r_kare}")

final_random_forest_model = RandomForestRegressor(n_estimators=en_iyi_n_estimators_r_kare, max_depth=en_iyi_max_depth_r_kare, random_state=random_state)
final_random_forest_model.fit(ozellik_egitim, hedef_egitim)

# Yeni değerler için tahmin
araba_df = pd.DataFrame([[tahmin_hedef_horsepower, tahmin_hedef_acceleration, tahmin_hedef_weight]], columns=ozellik_egitim.columns)
tahmin_mpg = final_random_forest_model.predict(araba_df)
print(f"Horsepower={tahmin_hedef_horsepower}, Acceleration={tahmin_hedef_acceleration}, Weight={tahmin_hedef_weight} için tahmin edilen MPG: {tahmin_mpg[0]:.2f}")

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import random

# Veri setini yükle
df = sns.load_dataset("mpg").dropna(subset=["horsepower"])

# Özellik ve hedef seçimi
ozellik = df[["horsepower", "acceleration", "weight"]]
hedef = df["mpg"]

# Eğitim-test ayrımı
random_state = 300 #random.randint(0, 999)
ozellik_egitim, ozellik_test, hedef_egitim, hedef_test = train_test_split(ozellik, hedef, test_size=0.1, random_state=random_state)

# Normalizasyon
min_max_normalizator = MinMaxScaler()
ozellik_egitim_normalize = min_max_normalizator.fit_transform(ozellik_egitim)
ozellik_test_normalize = min_max_normalizator.transform(ozellik_test)

# Skorlayıcılar
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
r_kare_scorer = make_scorer(r2_score, greater_is_better=True)

# Hiperparametre adayları
hidden_layer_sizes_adayları = []
for katman_sayisi in range(1, 8):
    for düğüm_sayisi in range(2,13):
        hidden_layer = tuple([düğüm_sayisi] * katman_sayisi)
        hidden_layer_sizes_adayları.append(hidden_layer)
print("hidden_layer_sizes_adayları:", hidden_layer_sizes_adayları)

alpha_adayları = [0.0001, 0.001]

max_iter = 1000

# Eğitim süresini inanılmaz derecede uzattığı ve hali hazırda şu anki eğitime yetecek kadar iyi sonuçlar..
# ..aldığımız için (r_kare ~0.7) diğer hiperparametreleri değişkenlerini yorum satırına al. Eklemek istersen..
# ..yorum satırlarını kaldır ve içeriye yeni for döngüleri ekleyerek cross validation sonuçlarını..
# ..güncelle. Grafikleri güncellemek çok daha sıkıntı olacak, farklı yollar deneyebilirsin.

# activation_adayları = ["identity", "logistic", "tanh", "relu"]
# solver_adayları = ["lbfgs", "sgd", "adam"]
# learning_rate_init_adayları = [0.0001, 0.001, 0.01, 0.1, 1]
# max_iter_adayları = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Cross-validation sonuçlarını saklayacağımız DataFrame
cross_validation_sonuçlar = pd.DataFrame(columns=["hidden_layer_sizes", "alpha", "r_kare", "mse", "mae"])

# Cross-validation ile sonuçların hesaplanması
for hidden_layers in hidden_layer_sizes_adayları:
    for alpha_degeri in alpha_adayları:
        ann_model = MLPRegressor(hidden_layer_sizes=hidden_layers, alpha=alpha_degeri, random_state=random_state, max_iter=max_iter)

        # R Kare
        r_kare_skorları = cross_val_score(ann_model, ozellik_egitim_normalize, hedef_egitim, cv=5, scoring=r_kare_scorer)
        r_kare = r_kare_skorları.mean()

        # MSE
        mse_skorları = cross_val_score(ann_model, ozellik_egitim_normalize, hedef_egitim, cv=5, scoring=mse_scorer)
        mse = (-1) * mse_skorları.mean()

        # MAE
        mae_skorları = cross_val_score(ann_model, ozellik_egitim_normalize, hedef_egitim, cv=5, scoring=mae_scorer)
        mae = (-1) * mae_skorları.mean()

        # Sonuçları dataframe'e ekle
        cross_validation_sonuçlar = cross_validation_sonuçlar._append({
            "hidden_layer_sizes": hidden_layers,
            "alpha": alpha_degeri,
            "r_kare": r_kare,
            "mse": mse,
            "mae": mae
        }, ignore_index=True)

        print("hidden_layer_sizes:", hidden_layers, "alpha:", alpha_degeri, "R Kare:", r_kare, "MSE:", mse, "MAE:", mae)

# R kare yöntemine göre en iyi hiperparametreleri bulma
en_iyi_r_kare = cross_validation_sonuçlar["r_kare"].max()
en_iyi_r_kare_satir = cross_validation_sonuçlar[cross_validation_sonuçlar["r_kare"] == en_iyi_r_kare].iloc[0]
en_iyi_r_kare_hidden_layers = en_iyi_r_kare_satir["hidden_layer_sizes"]
en_iyi_r_kare_alpha = en_iyi_r_kare_satir["alpha"]
print("En iyi R Kare:", en_iyi_r_kare, "En iyi hidden_layer_sizes:", en_iyi_r_kare_hidden_layers, "En iyi alpha:", en_iyi_r_kare_alpha)

# MSE yöntemine göre en iyi hiperparametreleri bulma
en_iyi_mse = cross_validation_sonuçlar["mse"].min()
en_iyi_mse_satir = cross_validation_sonuçlar[cross_validation_sonuçlar["mse"] == en_iyi_mse].iloc[0]
en_iyi_mse_hidden_layers = en_iyi_mse_satir["hidden_layer_sizes"]
en_iyi_mse_alpha = en_iyi_mse_satir["alpha"]
print("En iyi MSE:", en_iyi_mse, "En iyi hidden_layer_sizes:", en_iyi_mse_hidden_layers, "En iyi alpha:", en_iyi_mse_alpha)

# MAE yöntemine göre en iyi hiperparametreleri bulma
en_iyi_mae = cross_validation_sonuçlar["mae"].min()
en_iyi_mae_satir = cross_validation_sonuçlar[cross_validation_sonuçlar["mae"] == en_iyi_mae].iloc[0]
en_iyi_mae_hidden_layers = en_iyi_mae_satir["hidden_layer_sizes"]
en_iyi_mae_alpha = en_iyi_mae_satir["alpha"]
print("En iyi MAE:", en_iyi_mae, "En iyi hidden_layer_sizes:", en_iyi_mae_hidden_layers, "En iyi alpha:", en_iyi_mae_alpha)

# Grafikler: her bir alpha için ayrı çizgi çekerek hidden_layer_sizes'a göre performansa bakalım.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# R Kare grafiği (her alpha için ayrı çizgi)
for alpha_degeri in alpha_adayları:
    subset = cross_validation_sonuçlar[cross_validation_sonuçlar["alpha"] == alpha_degeri]
    unique_layers = list(hidden_layer_sizes_adayları)
    x_indices = [unique_layers.index(h) for h in subset["hidden_layer_sizes"]]
    axes[0].plot(x_indices, subset["r_kare"], marker='o', label=f"alpha={alpha_degeri}")
axes[0].set_xlabel("hidden_layer_sizes index")
axes[0].set_ylabel("R Kare")
axes[0].set_title("R Kare'ye Göre ANN Modeli Performansı")
axes[0].legend()
axes[0].grid(True)

# MSE grafiği
for alpha_degeri in alpha_adayları:
    subset = cross_validation_sonuçlar[cross_validation_sonuçlar["alpha"] == alpha_degeri]
    unique_layers = list(hidden_layer_sizes_adayları)
    x_indices = [unique_layers.index(h) for h in subset["hidden_layer_sizes"]]
    axes[1].plot(x_indices, subset["mse"], marker='o', label=f"alpha={alpha_degeri}")
axes[1].set_xlabel("hidden_layer_sizes index")
axes[1].set_ylabel("MSE")
axes[1].set_title("MSE'ye Göre ANN Modeli Performansı")
axes[1].legend()
axes[1].grid(True)

# MAE grafiği
for alpha_degeri in alpha_adayları:
    subset = cross_validation_sonuçlar[cross_validation_sonuçlar["alpha"] == alpha_degeri]
    unique_layers = list(hidden_layer_sizes_adayları)
    x_indices = [unique_layers.index(h) for h in subset["hidden_layer_sizes"]]
    axes[2].plot(x_indices, subset["mae"], marker='o', label=f"alpha={alpha_degeri}")
axes[2].set_xlabel("hidden_layer_sizes index")
axes[2].set_ylabel("MAE")
axes[2].set_title("MAE'ye Göre ANN Modeli Performansı")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("ann.png")
plt.show()

# Tahmin Hedefleri
tahmin_hedef_horsepower = 130
tahmin_hedef_acceleration = 13
tahmin_hedef_weight = 3500

# En iyi R Kare sonucunu veren hiperparametreler ile model oluşturma
print(f"\nEn iyi hiperparametrelerle ANN modeli oluşturuluyor...")
print(f"Hidden Layer Sizes: {en_iyi_r_kare_hidden_layers}")
print(f"Alpha: {en_iyi_r_kare_alpha}")

# Final ANN modelini oluştur
final_ann_model = MLPRegressor(
    hidden_layer_sizes=en_iyi_r_kare_hidden_layers,
    alpha=en_iyi_r_kare_alpha,
    random_state=random_state,
    max_iter=max_iter
)

# Modeli normalize edilmiş eğitim verileriyle eğit
final_ann_model.fit(ozellik_egitim_normalize, hedef_egitim)

# Yeni değerler için tahmin
araba_df = pd.DataFrame([[tahmin_hedef_horsepower, tahmin_hedef_acceleration, tahmin_hedef_weight]],
                        columns=ozellik_egitim.columns)
araba_df_normalize = min_max_normalizator.transform(araba_df)

# Tahmin yap
tahmin_mpg = final_ann_model.predict(araba_df_normalize)
print(f"Horsepower={tahmin_hedef_horsepower}, Acceleration={tahmin_hedef_acceleration}, Weight={tahmin_hedef_weight} için tahmin edilen MPG: {tahmin_mpg[0]:.2f}")