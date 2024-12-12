import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
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
scorers = {
    'r2': make_scorer(r2_score),
    'neg_mse': make_scorer(mean_squared_error, greater_is_better=False),
    'neg_mae': make_scorer(mean_absolute_error, greater_is_better=False)
}

# Hiperparametre adayları
hidden_layer_sizes_adayları = []
for katman_sayisi in range(1, 6):  # 1'den 5'e kadar katman sayısı
    for düğüm_sayisi in range(10, 101, 10):  # Her katmanda 10'dan 100'e kadar
        hidden_layer = tuple([düğüm_sayisi] * katman_sayisi)
        hidden_layer_sizes_adayları.append(hidden_layer)

activation_adayları = ["identity", "logistic", "tanh", "relu"]
solver_adayları = ["lbfgs", "sgd", "adam"]
alpha_adayları = [0.0001, 0.001, 0.01, 0.1, 1]
learning_rate_init_adayları = [0.0001, 0.001, 0.01, 0.1, 1]
max_iter_adayları = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Cross-validation sonuçlarını saklayacağımız DataFrame
cross_validation_sonuçlar = pd.DataFrame(columns=["hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate_init", "max_iter", "r_kare", "mse", "mae"])

# Grid Search için parametreler
param_grid = {
    "hidden_layer_sizes": hidden_layer_sizes_adayları,
    "activation": activation_adayları,
    "solver": solver_adayları,
    "alpha": alpha_adayları,
    "learning_rate_init": learning_rate_init_adayları,
    "max_iter": max_iter_adayları
}

# Grid Search ile model optimizasyonu
grid_search = GridSearchCV(
    MLPRegressor(random_state=random_state),
    param_grid,
    cv=5,
    scoring=scorers,
    refit='r2'  # R-kare skoruna göre en iyi modeli seç
)

# Grid Search'ü çalıştır
grid_search.fit(ozellik_egitim_normalize, hedef_egitim)

# Detaylı sonuçları DataFrame'e aktar
for params, mean_score, scores in zip(
        grid_search.cv_results_['params'],
        grid_search.cv_results_['mean_test_r2'],
        grid_search.cv_results_['mean_test_neg_mse']
):
    cross_validation_sonuçlar = cross_validation_sonuçlar._append({
        "hidden_layer_sizes": params['hidden_layer_sizes'],
        "alpha": params['alpha'],
        "r_kare": mean_score,
        "mse": -scores,  # Negatif MSE'yi pozitife çevir
    }, ignore_index=True)

# En iyi modelin parametrelerini yazdır
print("En iyi parametreler:", grid_search.best_params_)
print("En iyi R-kare skoru:", grid_search.best_score_)

# R kare yöntemine göre en iyi hiperparametreleri bulma
en_iyi_r_kare = cross_validation_sonuçlar["r_kare"].max()
en_iyi_r_kare_satir = cross_validation_sonuçlar[cross_validation_sonuçlar["r_kare"] == en_iyi_r_kare].iloc[0]
en_iyi_r_kare_hidden_layers = en_iyi_r_kare_satir["hidden_layer_sizes"]
en_iyi_r_kare_activation = en_iyi_r_kare_satir["activation"]
en_iyi_r_kare_solver = en_iyi_r_kare_satir["solver"]
en_iyi_r_kare_alpha = en_iyi_r_kare_satir["alpha"]
en_iyi_r_kare_learning_rate_init = en_iyi_r_kare_satir["learning_rate_init"]
en_iyi_r_kare_max_iter = en_iyi_r_kare_satir["max_iter"]

print("En iyi R Kare:", en_iyi_r_kare,
        "En iyi hidden_layer_sizes:", en_iyi_r_kare_hidden_layers,
        "En iyi activation:", en_iyi_r_kare_activation,
        "En iyi solver:", en_iyi_r_kare_solver,
        "En iyi alpha:", en_iyi_r_kare_alpha,
        "En iyi learning_rate_init:", en_iyi_r_kare_learning_rate_init,
        "En iyi max_iter:", en_iyi_r_kare_max_iter)

# MSE yöntemine göre en iyi hiperparametreleri bulma
en_iyi_mse = cross_validation_sonuçlar["mse"].min()
en_iyi_mse_satir = cross_validation_sonuçlar[cross_validation_sonuçlar["mse"] == en_iyi_mse].iloc[0]
en_iyi_mse_hidden_layers = en_iyi_mse_satir["hidden_layer_sizes"]
en_iyi_mse_activation = en_iyi_mse_satir["activation"]
en_iyi_mse_solver = en_iyi_mse_satir["solver"]
en_iyi_mse_alpha = en_iyi_mse_satir["alpha"]
en_iyi_mse_learning_rate_init = en_iyi_mse_satir["learning_rate_init"]
en_iyi_mse_max_iter = en_iyi_mse_satir["max_iter"]

print("En iyi MSE:", en_iyi_mse,
        "En iyi hidden_layer_sizes:", en_iyi_mse_hidden_layers,
        "En iyi activation:", en_iyi_mse_activation,
        "En iyi solver:", en_iyi_mse_solver,
        "En iyi alpha:", en_iyi_mse_alpha,
        "En iyi learning_rate_init:", en_iyi_mse_learning_rate_init,
        "En iyi max_iter:", en_iyi_mse_max_iter)

# MAE yöntemine göre en iyi hiperparametreleri bulma
en_iyi_mae = cross_validation_sonuçlar["mae"].min()
en_iyi_mae_satir = cross_validation_sonuçlar[cross_validation_sonuçlar["mae"] == en_iyi_mae].iloc[0]
en_iyi_mae_hidden_layers = en_iyi_mae_satir["hidden_layer_sizes"]
en_iyi_mae_activation = en_iyi_mae_satir["activation"]
en_iyi_mae_solver = en_iyi_mae_satir["solver"]
en_iyi_mae_alpha = en_iyi_mae_satir["alpha"]
en_iyi_mae_learning_rate_init = en_iyi_mae_satir["learning_rate_init"]
en_iyi_mae_max_iter = en_iyi_mae_satir["max_iter"]

print("En iyi MAE:", en_iyi_mae,
        "En iyi hidden_layer_sizes:", en_iyi_mae_hidden_layers,
        "En iyi activation:", en_iyi_mae_activation,
        "En iyi solver:", en_iyi_mae_solver,
        "En iyi alpha:", en_iyi_mae_alpha,
        "En iyi learning_rate_init:", en_iyi_mae_learning_rate_init,
        "En iyi max_iter:", en_iyi_mae_max_iter)