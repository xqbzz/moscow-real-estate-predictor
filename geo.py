import requests
import pandas as pd
import re
from tqdm import tqdm
tqdm.pandas()  # интеграция с pandas

YANDEX_API_KEY = "d5fd1298-cad2-48b6-b327-e2e13a324c25"

def clean_address(raw):
    raw = str(raw)
    # Удаляем метро, районы, округа, микрорайоны
    raw = re.sub(r"м\.\s?\S+", "", raw)
    raw = re.sub(r"р-н\s+\S+", "", raw)
    raw = re.sub(r"\b(ЦАО|САО|СЗАО|ЮЗАО|ЮАО|ЮВАО|ВАО|ЗАО|СВАО|СЦАО|НАО)\b", "", raw)
    raw = re.sub(r"мкр.*", "", raw)

    # Удаляем лишние запятые и пробелы
    raw = re.sub(r",\s*,", ",", raw)
    raw = re.sub(r"\s+", " ", raw).strip(", ")

    # Удаляем "Москва" в начале, если уже есть
    raw = re.sub(r"^Москва\s*,?\s*", "", raw)

    return "Москва, " + raw.strip(", ")

def geocode_yandex(address):
    url = "https://geocode-maps.yandex.ru/1.x/"
    params = {
        "apikey": YANDEX_API_KEY,
        "geocode": address,
        "format": "json",
        "lang": "ru_RU"
    }
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        resp_json = response.json()
        pos = resp_json['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point']['pos']
        lon, lat = map(float, pos.split())
        return lat, lon
    except Exception as e:
        print(f"❌ Ошибка при геокодировании '{address}': {e}")
        return None, None


# Загрузка и очистка
df = pd.read_csv("cian_moscow_flats.csv")
df["Чистый адрес"] = df["Инфо"].apply(clean_address)

# Геокодирование
coords = df["Чистый адрес"].progress_apply(geocode_yandex)
df["lat"] = coords.apply(lambda x: x[0])
df["lon"] = coords.apply(lambda x: x[1])

# Проверка
print(df[["Чистый адрес", "lat", "lon"]].head(10))

# Сохраняем обновлённый DataFrame
df.to_csv("cian_moscow_flats_with_coords.csv", index=False, encoding="utf-8-sig")
print("✅ Сохранено в 'cian_moscow_flats_with_coords.csv'")
