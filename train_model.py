"""Шаги 2-5. EDA, признаки, разделение по времени, модели, графики.

Запуск:  python train_model.py
Результаты: figures/*.png, results/metrics.csv
"""
import matplotlib

matplotlib.use("Agg")  # сохраняем графики в файлы, окна не открываем
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/air_quality_raw.csv"
TEST_START = "2024-07-01"  # последние 6 месяцев оставляем на тест
RANDOM_STATE = 42
WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]


# ---------- 1. Загрузка и проверки ----------
def load_hourly() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    print("Форма:", df.shape)
    print("Дней в данных:", len(df) / 24)
    print("\nПропуски:\n", df.isna().sum())
    print("\nСтатистика:\n", df.describe().round(2))
    return df


# ---------- 2. EDA ----------
def make_eda_plots(df: pd.DataFrame, daily: pd.DataFrame) -> None:
    # PM2.5 по дням
    ax = daily["pm2_5"].plot(figsize=(12, 4), title="PM2.5 по дням")
    ax.axvline(pd.Timestamp(TEST_START), color="red", linestyle="--", label="начало теста")
    ax.set_ylabel("µg/m³")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/01_pm25_daily.png", dpi=150)
    plt.close()

    # PM2.5 по месяцам
    monthly = daily.groupby(daily.index.month)["pm2_5"].mean()
    monthly.plot(kind="bar", figsize=(8, 4), title="Средний PM2.5 по месяцам")
    plt.xlabel("месяц")
    plt.ylabel("µg/m³")
    plt.tight_layout()
    plt.savefig("figures/02_pm25_monthly.png", dpi=150)
    plt.close()

    # По часам суток: PM2.5 и температура (заодно проверка часового пояса)
    by_hour = df.groupby(df["time"].dt.hour)[["pm2_5", "temperature_2m"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    by_hour["pm2_5"].plot(kind="bar", ax=axes[0], title="PM2.5 по часам суток")
    by_hour["temperature_2m"].plot(kind="bar", ax=axes[1], title="Температура по часам суток")
    axes[0].set_xlabel("час")
    axes[1].set_xlabel("час")
    plt.tight_layout()
    plt.savefig("figures/03_by_hour.png", dpi=150)
    plt.close()

    # Корреляции
    corr = daily[["pm2_5"] + WEATHER_COLS].corr()
    print("\nКорреляция с PM2.5:\n", corr["pm2_5"].round(3))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Корреляции (суточные значения)")
    plt.tight_layout()
    plt.savefig("figures/04_correlations.png", dpi=150)
    plt.close()


# ---------- 3. Суточная таблица и признаки ----------
def make_daily(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("time").resample("D").agg(
        {
            "pm2_5": "mean",
            "temperature_2m": "mean",
            "relative_humidity_2m": "mean",
            "wind_speed_10m": "mean",
            "precipitation": "sum",  # осадки суммируются, остальное усредняется
        }
    )


def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    f = daily.copy()
    # Всё, что используется как признак, известно к концу сегодняшнего дня
    f["pm25_lag1"] = f["pm2_5"].shift(1)
    f["pm25_lag2"] = f["pm2_5"].shift(2)
    f["pm25_lag7"] = f["pm2_5"].shift(7)
    f["pm25_roll3"] = f["pm2_5"].rolling(3).mean()
    f["pm25_roll7"] = f["pm2_5"].rolling(7).mean()
    f["pm25_change"] = f["pm2_5"] - f["pm25_lag1"]
    f["precip_roll3"] = f["precipitation"].rolling(3).sum()
    f["month"] = f.index.month
    f["month_sin"] = np.sin(2 * np.pi * f["month"] / 12)
    f["month_cos"] = np.cos(2 * np.pi * f["month"] / 12)
    f["dayofweek"] = f.index.dayofweek
    # Цель: PM2.5 завтра
    f["target"] = f["pm2_5"].shift(-1)
    return f


# ---------- 4. Разделение по времени ----------
def split_by_time(data: pd.DataFrame):
    data = data.dropna()
    test_start = pd.Timestamp(TEST_START)
    # Последний день трейна исключаем: его target лежит уже в тестовом периоде
    train = data[data.index < test_start - pd.Timedelta(days=1)]
    test = data[data.index >= test_start]
    features = [c for c in data.columns if c != "target"]
    return train[features], train["target"], test[features], test["target"], features


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


# ---------- 5. Модели ----------
def get_models() -> dict:
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, random_state=RANDOM_STATE
        ),
    }


def main() -> None:
    df = load_hourly()
    daily = make_daily(df)
    make_eda_plots(df, daily)

    data = add_features(daily)
    X_train, y_train, X_test, y_test, features = split_by_time(data)
    print(f"\nTrain: {len(X_train)} дней ({X_train.index.min().date()} - {X_train.index.max().date()})")
    print(f"Test:  {len(X_test)} дней ({X_test.index.min().date()} - {X_test.index.max().date()})")

    results = {}
    predictions = {}

    # Базовая модель: «завтра будет как сегодня»
    base_pred = X_test["pm2_5"]
    results["Baseline (как сегодня)"] = metrics(y_test, base_pred)
    predictions["Baseline (как сегодня)"] = base_pred

    # Модели: кросс-валидация по времени на трейне + оценка на тесте
    tscv = TimeSeriesSplit(n_splits=5)
    for name, model in get_models().items():
        cv_mae = -cross_val_score(
            model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error"
        ).mean()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pd.Series(pred, index=X_test.index)
        results[name] = {**metrics(y_test, pred), "CV_MAE_train": cv_mae}

    res = pd.DataFrame(results).T.round(3)
    print("\nРезультаты на тесте:\n", res)
    res.to_csv("results/metrics.csv")

    # Лучшая модель (без baseline)
    model_names = list(get_models().keys())
    best_name = res.loc[model_names, "MAE"].astype(float).idxmin()
    best_model = get_models()[best_name].fit(X_train, y_train)
    print(f"\nЛучшая модель: {best_name}")

    # График: прогноз против факта
    plt.figure(figsize=(12, 4))
    plt.plot(y_test.index, y_test, label="Факт", color="black")
    plt.plot(predictions[best_name], label=f"Прогноз ({best_name})", color="tab:orange")
    plt.plot(predictions["Baseline (как сегодня)"], label="Baseline", alpha=0.5, color="tab:blue")
    plt.title("PM2.5 на следующий день: тестовый период")
    plt.ylabel("µg/m³")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/05_predictions.png", dpi=150)
    plt.close()

    # График: важность признаков (permutation importance на тесте)
    imp = permutation_importance(
        best_model, X_test, y_test, n_repeats=20,
        scoring="neg_mean_absolute_error", random_state=RANDOM_STATE,
    )
    imp_s = pd.Series(imp.importances_mean, index=features).sort_values()
    imp_s.plot(kind="barh", figsize=(7, 5), title=f"Важность признаков ({best_name})")
    plt.xlabel("рост MAE при перемешивании признака")
    plt.tight_layout()
    plt.savefig("figures/06_feature_importance.png", dpi=150)
    plt.close()
    print("\nТоп-5 признаков:\n", imp_s.sort_values(ascending=False).head(5).round(3))

    # График: остатки лучшей модели
    resid = y_test - predictions[best_name]
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=25, color="tab:orange", edgecolor="white")
    plt.title("Распределение ошибок (факт - прогноз)")
    plt.xlabel("µg/m³")
    plt.tight_layout()
    plt.savefig("figures/07_residuals.png", dpi=150)
    plt.close()

    print("\nГотово. Графики в figures/, метрики в results/metrics.csv")


if __name__ == "__main__":
    main()
