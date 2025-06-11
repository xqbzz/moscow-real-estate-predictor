import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics import r2_score, mean_absolute_percentage_error

import joblib
import os

# Создание директорий, если не существуют
os.makedirs("output/plots", exist_ok=True)
os.makedirs("output/models", exist_ok=True)

# 📥 Загрузка данных
df = pd.read_csv("cian_moscow_detailed.csv")
df["Цена"] = df["Цена"].astype(str).str.replace(r"[^\d]", "", regex=True)
df = df[df["Цена"] != ""]
df["Цена"] = df["Цена"].astype(float)

# 📊 Разделение сегментов
df["is_luxury"] = df["Цена"] > 3e8
df["Цена_лог"] = np.log1p(df["Цена"])

# 🧹 Удаление выбросов
df_main = df[~df["is_luxury"]].copy()
q_high = df_main["Цена"].quantile(0.99)
df_main = df_main[df_main["Цена"] < q_high]

# 🔍 Разбор этажей
def parse_floor(raw):
    match = re.search(r"(\d+)\s*из\s*(\d+)", str(raw))
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)

df_main[["Этаж_текущий", "Этаж_всего"]] = df_main["Этаж"].apply(lambda x: pd.Series(parse_floor(x)))
df_lux = df[df["is_luxury"]].copy()
df_lux[["Этаж_текущий", "Этаж_всего"]] = df_lux["Этаж"].apply(lambda x: pd.Series(parse_floor(x)))

# 📍 Расчёт расстояния до метро
metro = pd.read_csv("metro_stations.csv")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def nearest_metro(row):
    return metro.apply(lambda m: haversine(row.lat, row.lon, m.lat, m.lon), axis=1).min()

# 🧠 Преобразование
def preprocess(df):
    df = df.copy()

    # Удаляем лишние поля
    df.drop(columns=["Этаж", "Комнаты", "Название", "Ссылка", "Чистый адрес"], inplace=True, errors="ignore")

    # Извлекаем район и метро
    def extract_district(info):
        match = re.search(r"р-н\s([\w\-]+)", str(info))
        return match.group(1) if match else "Неизвестно"

    def extract_metro(info):
        match = re.search(r"м\.\s([\w\- ]+)", str(info))
        return match.group(1) if match else "Неизвестно"

    df["Район"] = df["Инфо"].apply(extract_district)
    df["Метро"] = df["Инфо"].apply(extract_metro)
    df.drop(columns=["Инфо"], inplace=True)

    # ❗ Удаляем только строки с критичными NaN
    df.dropna(subset=["Площадь", "lat", "lon"], inplace=True)

    # 🧼 Заполняем остальные значения
    for col in ["Этаж_текущий", "Этаж_всего", "Тип дома", "Год постройки", "Потолки"]:
        if df[col].dtype == "O":
            df[col] = df[col].fillna("Неизвестно")
        else:
            df[col] = df[col].fillna(df[col].median())

    # ➕ Признаки
    df["Возраст_дома"] = 2025 - pd.to_numeric(df["Год постройки"], errors="coerce")
    df["Возраст_дома"] = df["Возраст_дома"].fillna(df["Возраст_дома"].median())

    df["Этаж_доля"] = pd.to_numeric(df["Этаж_текущий"], errors="coerce") / pd.to_numeric(df["Этаж_всего"], errors="coerce")
    df["Этаж_доля"] = df["Этаж_доля"].fillna(0.5)

    df["Расстояние_до_центра"] = np.sqrt((df["lat"] - 55.7558)**2 + (df["lon"] - 37.6176)**2)
    df["dist_to_metro_km"] = df.apply(nearest_metro, axis=1)

    df["Премиум_район"] = df["Район"].isin(["Хамовники", "Пресненский", "Якиманка", "Замоскворечье"]).astype(int)

    # Кластеризация
    kmeans = KMeans(n_clusters=15, random_state=42)
    df["Кластер"] = kmeans.fit_predict(df[["lat", "lon"]])
    df["Кластер"] = df["Кластер"].astype(str)

    # Категориальные и числовые
    categorical = ["Тип дома", "Район", "Метро", "Кластер"]
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoded = encoder.fit_transform(df[categorical]).toarray()
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical), index=df.index)

    numeric = ["Площадь", "Этаж_текущий", "Этаж_всего", "Потолки", "Возраст_дома", "Этаж_доля", "Расстояние_до_центра", "dist_to_metro_km"]
    numeric = [col for col in numeric if col in df.columns]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric])
    scaled_df = pd.DataFrame(scaled, columns=numeric, index=df.index)

    features = pd.concat([scaled_df, encoded_df, df[["Премиум_район"]]], axis=1)
    target = np.log1p(df["Цена"])

    return features, target

# 🚀 Модель стэкинг
X_main, y_main = preprocess(df_main)
X_train, X_test, y_train, y_test = train_test_split(X_main, y_main, test_size=0.2, random_state=42)

xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

model_main = StackingRegressor(
    estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
    final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
    passthrough=True,
    n_jobs=-1
)
model_main.fit(X_train, y_train)

# 💎 Luxury модель
X_lux, y_lux = preprocess(df_lux)
model_lux = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
model_lux.fit(X_lux, y_lux)

# 📈 Предсказания
y_pred_main = np.expm1(model_main.predict(X_test))
y_test_true = np.expm1(y_test)
mae_main = mean_absolute_error(y_test_true, y_pred_main)

y_pred_lux = np.expm1(model_lux.predict(X_lux))
y_lux_true = np.expm1(y_lux)
mae_lux = mean_absolute_error(y_lux_true, y_pred_lux)

print(f"\n📊 MAE на обычных квартирах: {mae_main:,.0f} ₽")
print(f"💎 MAE на luxury-сегменте: {mae_lux:,.0f} ₽")

# 📊 Визуализации
comparison_main = pd.DataFrame({"Реальная (₽)": y_test_true, "Предсказание (₽)": y_pred_main})
comparison_lux = pd.DataFrame({"Реальная (₽)": y_lux_true, "Предсказание (₽)": y_pred_lux})
print("\n🔍 Обычные квартиры:")
print(comparison_main.head(5))
print("\n🔍 Luxury квартиры:")
print(comparison_lux.head(5))

print(f"Объектов после препроцессинга: {len(X_main)} (всего)")
print(f"Объектов в тестовой выборке: {len(X_test)}")
print(f"R²: {r2_score(y_test_true, y_pred_main):.3f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test_true, y_pred_main):.2%}")
print(f"R²(LUX): {r2_score(y_lux_true, y_pred_lux):.3f}")
print(f"MAPE(LUX): {mean_absolute_percentage_error(y_lux_true, y_pred_lux):.2%}")


# Важность признаков
importances = model_main.named_estimators_["xgb"].feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({"Признак": feature_names, "Важность": importances}).sort_values("Важность", ascending=False).head(20)
plt.figure(figsize=(10, 6))
sns.barplot(x="Важность", y="Признак", data=importance_df)
plt.title("Топ-20 признаков (XGBoost)")
plt.tight_layout()
plt.savefig("output/plots/feature_importance.png")


# Ошибки
errors = np.abs(y_test_true - y_pred_main)
plt.figure(figsize=(8, 5))
sns.histplot(errors, bins=30, kde=True, color="crimson")
plt.title("Абсолютные ошибки")
plt.tight_layout()
plt.savefig("output/plots/missing.png")


# Плотности
plt.figure(figsize=(8, 5))
sns.kdeplot(y_test_true, label="Реальная цена")
sns.kdeplot(y_pred_main, label="Предсказанная")
plt.legend()
plt.title("Распределение (Обычные)")
plt.tight_layout()
plt.savefig("output/plots/plot(stangart).png")


plt.figure(figsize=(8, 5))
sns.kdeplot(y_lux_true, label="Реальная (Luxury)")
sns.kdeplot(y_pred_lux, label="Предсказанная (Luxury)")
plt.legend()
plt.title("Распределение (Luxury)")
plt.tight_layout()
plt.savefig("output/plots/plot_lux.png")


# Сохраняем модель и препроцессор
joblib.dump(model_main, "output/models/model_main.joblib")
joblib.dump(preprocess, "output/models/preprocess_func.joblib")
joblib.dump(X_test, "output/models/X_test.joblib")
joblib.dump(y_test_true, "output/models/y_test_true.joblib")
joblib.dump(y_pred_main, "output/models/y_pred_main.joblib")
joblib.dump(df_main, "output/models/df_main.joblib")
