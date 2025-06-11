import os
import sys
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# 🧾 Аргументы командной строки
if len(sys.argv) != 3:
    print("❌ Использование: python multi_parser.py <start_page> <end_page>")
    sys.exit(1)

start_page = int(sys.argv[1])
end_page = int(sys.argv[2])

output_path = f"data/cian_part_{start_page}_{end_page}.csv"
os.makedirs("data", exist_ok=True)

# 🔧 Настройка драйвера
options = webdriver.ChromeOptions()
options.binary_location = "C:/Users/Сергей/AppData/Local/Yandex/YandexBrowser/Application/browser.exe"
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(
    service=Service("C:/Users/Сергей/Downloads/134.0.6998.35 chromedriver-win64/chromedriver-win64/chromedriver.exe"),
    options=options
)

data = []

for page in range(start_page, end_page + 1):
    try:
        url = f"https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&p={page}&region=1"
        driver.get(url)
        time.sleep(random.uniform(4.5, 8.0))

        cards = driver.find_elements(By.XPATH, '//article[@data-name="CardComponent"]')
        print(f"[Страница {page}] Найдено карточек: {len(cards)}")

        for card in cards:
            try:
                title = card.find_element(By.XPATH, './/span[contains(@data-mark,"Title")]').text
            except:
                title = "Нет названия"

            try:
                price = card.find_element(By.XPATH, './/span[contains(@data-mark,"Price")]').text
            except:
                price = "Нет цены"

            try:
                info = card.find_element(By.XPATH, './/div[contains(@class,"_93444fe79c--labels--")]').text
            except:
                info = "Нет описания"

            try:
                link = card.find_element(By.XPATH, './/a[contains(@href, "/sale/")]').get_attribute("href")
            except:
                link = "Нет ссылки"

            data.append({
                "Название": title,
                "Цена": price,
                "Инфо": info,
                "Ссылка": link
            })

        # 💾 Автосейв
        pd.DataFrame(data).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"💾 Сохранено {len(data)} объявлений (до страницы {page})")

    except Exception as e:
        print(f"❌ Ошибка на странице {page}: {e}")
        time.sleep(10)

driver.quit()
print(f"✅ Готово. Сохранено: {len(data)} в {output_path}")
