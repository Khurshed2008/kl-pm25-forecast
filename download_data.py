"""Шаг 1. Скачивание данных (Open-Meteo) и проверка часового пояса.

Запуск:  python download_data.py
Результат: data/air_quality_raw.csv
"""
import pandas as pd
import requests

CITY = "Kuala Lumpur"
START, END = "2023-01-01", "2024-12-31"
TIMEZONE = "Asia/Kuala_Lumpur"  # без этого время приходит в UTC (сдвиг на 8 часов)
OUT_PATH = "data/air_quality_raw.csv"


def get_json(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    # 1. Координаты города
    geo = get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": CITY, "count": 1},
    )["results"][0]
    lat, lon = geo["latitude"], geo["longitude"]
    print(f"{CITY}: lat={lat}, lon={lon}")

    common = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START,
        "end_date": END,
        "timezone": TIMEZONE,
    }

    # 2. Качество воздуха
    air = get_json(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        {**common, "hourly": "pm2_5"},
    )
    df_air = pd.DataFrame(air["hourly"])

    # 3. Погода
    weather = get_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            **common,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        },
    )
    df_weather = pd.DataFrame(weather["hourly"])

    # 4. Объединяем и сохраняем
    df = df_air.merge(df_weather, on="time", how="inner")
    df["time"] = pd.to_datetime(df["time"])
    df.to_csv(OUT_PATH, index=False)
    print(f"Сохранено: {OUT_PATH}, форма {df.shape}")

    # 5. Проверка часового пояса: самое жаркое время должно быть днём (~12-16 ч)
    peak_hour = df.groupby(df["time"].dt.hour)["temperature_2m"].mean().idxmax()
    print(f"Час максимальной температуры: {peak_hour}:00")
    if not 12 <= peak_hour <= 16:
        print("ВНИМАНИЕ: пик температуры не днём. Проверь параметр timezone!")


if __name__ == "__main__":
    main()
