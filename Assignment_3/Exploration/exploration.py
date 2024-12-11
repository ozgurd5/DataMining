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