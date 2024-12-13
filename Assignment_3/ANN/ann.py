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