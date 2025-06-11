import pandas as pd
import glob

# 🔍 Сбор всех частей
files = glob.glob("data/cian_part_*.csv")
dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)
print(f"📦 Загружено строк (до очистки): {len(combined)}")

# 🚫 Удаление дублей по ссылке
cleaned = combined.drop_duplicates(subset="Ссылка")
print(f"✅ Уникальных ссылок: {len(cleaned)}")

# 💾 Сохранение
cleaned.to_csv("cian_moscow_flats.csv", index=False, encoding='utf-8-sig')
print("📁 Сохранено в 'cian_moscow_flats.csv'")
