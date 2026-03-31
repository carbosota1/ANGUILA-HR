import os
import pandas as pd

DATA_PATH = "data/anguilla_hourly_history.csv"
OUT_PATH = "data/anguilla_dataset.csv"


def load_data():
    df = pd.read_csv(DATA_PATH, dtype=str).fillna("")
    return df


def clean_data(df):
    # Solo resultados válidos
    df = df[df["status"] == "OK"].copy()

    # Convertir números
    for col in ["primero", "segundo", "tercero"]:
        df[col] = df[col].astype(str).str.zfill(2)

    return df


def sort_data(df):
    # Orden correcto por tiempo
    df["datetime"] = pd.to_datetime(df["fecha"] + " " + df["hora"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def add_lags(df, max_lag=3):
    for lag in range(1, max_lag + 1):
        df[f"lag{lag}_p1"] = df["primero"].shift(lag)
        df[f"lag{lag}_p2"] = df["segundo"].shift(lag)
        df[f"lag{lag}_p3"] = df["tercero"].shift(lag)

    return df


def explode_numbers(df):
    """
    Convierte cada fila en múltiples filas:
    cada número como evento individual
    """
    rows = []

    for _, r in df.iterrows():
        nums = [r["primero"], r["segundo"], r["tercero"]]

        for num in nums:
            new_row = r.to_dict()
            new_row["num"] = num
            rows.append(new_row)

    return pd.DataFrame(rows)


def build_dataset():
    df = load_data()
    df = clean_data(df)
    df = sort_data(df)
    df = add_lags(df, max_lag=3)
    df = explode_numbers(df)

    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Dataset creado: {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    build_dataset()