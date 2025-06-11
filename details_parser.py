import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from tqdm import tqdm

# Настройка драйвера для Yandex Browser
options = webdriver.ChromeOptions()
options.binary_location = "C:/Users/Сергей/AppData/Local/Yandex/YandexBrowser/Application/browser.exe"
# options.add_argument('--headless')  # если хочешь — включай headless
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36")

driver = webdriver.Chrome(
    service=Service("C:/Users/Сергей/Downloads/134.0.6998.35 chromedriver-win64/chromedriver-win64/chromedriver.exe"),
    options=options
)

# Загрузка данных
df = pd.read_csv("cian_moscow_flats_with_coords.csv")

# Новые поля
df["Комнаты"] = None
df["Площадь"] = None
df["Этаж"] = None
df["Тип дома"] = None
df["Год постройки"] = None
df["Потолки"] = None

# Основной цикл с прогрессбаром
for i, row in tqdm(df.iterrows(), total=len(df)):
    url = row["Ссылка"]
    print(f"\n[{i+1}/{len(df)}] {url}")

    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, '//div[contains(@class, "a10a3f92e9--item--")]'))
        )

        params = driver.find_elements(By.XPATH, '//div[contains(@class, "a10a3f92e9--item--")]')
        print(f"  ▶ Найдено параметров: {len(params)}")

        for p in params:
            try:
                lines = p.text.strip().split("\n")
                if len(lines) < 2:
                    continue

                label = lines[0].strip()
                value = lines[1].strip()
                print(f"    {label} → {value}")

                if "Общая площадь" in label:
                    df.at[i, "Площадь"] = value.replace(" м²", "").replace(",", ".")
                elif "Этаж" in label:
                    df.at[i, "Этаж"] = value
                elif "Тип дома" in label:
                    df.at[i, "Тип дома"] = value
                elif "Год сдачи" in label or "Год постройки" in label:
                    df.at[i, "Год постройки"] = value
                elif "Высота потолков" in label:
                    df.at[i, "Потолки"] = value.replace(" м", "").replace(",", ".")
                elif "комнат" in label.lower() or "комн" in value.lower():
                    df.at[i, "Комнаты"] = value

            except Exception as e:
                print(f"    ⚠ Не удалось обработать элемент: {e}")

    except Exception as e:
        print(f"  ⚠ Ошибка при парсинге {url}: {e}")
        time.sleep(5)

    # 💾 Автосейв каждые 50 квартир
    if i % 50 == 0 and i > 0:
        df.iloc[:i].to_csv("tmp_details_progress.csv", index=False, encoding="utf-8-sig")
        print(f"💾 Промежуточное сохранение на {i} объектах")

driver.quit()
df.to_csv("cian_moscow_detailed.csv", index=False, encoding="utf-8-sig")
print("✅ Сохранено в 'cian_moscow_detailed.csv'")
