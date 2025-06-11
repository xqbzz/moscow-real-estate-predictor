import joblib
import folium
from folium.plugins import MarkerCluster

X_test = joblib.load("output/models/X_test.joblib")
y_test_true = joblib.load("output/models/y_test_true.joblib")
y_pred_main = joblib.load("output/models/y_pred_main.joblib")
df_main = joblib.load("output/models/df_main.joblib")

# Создание карты
m = folium.Map(location=[55.7558, 37.6176], zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

for i in range(len(X_test)):
    row = df_main.loc[X_test.index[i]]
    lat = row["lat"]
    lon = row["lon"]
    real_price = y_test_true.iloc[i]
    pred_price = y_pred_main[i]
    error = abs(real_price - pred_price)

    popup_text = f"""
    <b>Реальная цена:</b> {real_price:,.0f} ₽<br>
    <b>Предсказание:</b> {pred_price:,.0f} ₽<br>
    <b>Ошибка:</b> {error:,.0f} ₽
    """

    folium.CircleMarker(
        location=(lat, lon),
        radius=6,
        popup=folium.Popup(popup_text, max_width=300),
        color="green" if error < 2_000_000 else "red",
        fill=True,
        fill_opacity=0.6,
    ).add_to(marker_cluster)
    
m.save("output/plots/predictions_map.html")
print("✅ Карта сохранена в output/plots/predictions_map.html")
