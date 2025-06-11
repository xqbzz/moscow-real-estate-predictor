import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import joblib

# Загрузка модели и функции препроцессинга
model = joblib.load("model_main.joblib")
from predict import preprocess

# Загрузка расширенного набора данных
df = pd.read_csv("cian_moscow_detailed.csv")
df = df[df["lat"].notna() & df["lon"].notna()]

# 🧱 Разбор этажей (нужно для preprocess)
import re
def parse_floor(raw):
    match = re.search(r"(\d+)\s*из\s*(\d+)", str(raw))
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)

df[["Этаж_текущий", "Этаж_всего"]] = df["Этаж"].apply(lambda x: pd.Series(parse_floor(x)))

# Преобразование цены в числовой формат
df["Цена"] = df["Цена"].astype(str).str.replace(r"[^\d]", "", regex=True)
df = df[df["Цена"] != ""]
df["Цена"] = df["Цена"].astype(float)

# Предсказание
X, _ = preprocess(df)
df["Предсказанная цена"] = np.expm1(model.predict(X))

# Градиент цвета по предсказанной цене
price_min = df["Предсказанная цена"].min()
price_max = df["Предсказанная цена"].max()

def price_to_color(price):
    norm = (price - price_min) / (price_max - price_min)
    r = int(255 * norm)
    g = int(255 * (1 - norm))
    return f'#{r:02x}{g:02x}00'

# Карта
m = folium.Map(location=[55.75, 37.62], zoom_start=11)
cluster = MarkerCluster().add_to(m)

for _, row in df.iterrows():
    color = price_to_color(row["Предсказанная цена"])
    popup = folium.Popup(
        f"<b>{row.get('Название', 'Квартира')}</b><br>"
        f"Предсказанная цена: {row['Предсказанная цена']:,.0f} ₽<br>"
        f"Реальная цена: {row['Цена']:,.0f} ₽",
        max_width=300
    )
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=popup
    ).add_to(cluster)

m.save("map_predictions.html")
