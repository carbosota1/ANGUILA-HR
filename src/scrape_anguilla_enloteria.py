import os
import re
import sys
import time
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

TZ_RD = ZoneInfo("America/Santo_Domingo")

BASE_URL = "https://enloteria.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}

ANGUILLA_HOURLY_SLOTS = [
    "8AM", "9AM", "10AM", "11AM", "12PM",
    "1PM", "2PM", "3PM", "4PM", "5PM",
    "6PM", "7PM", "8PM", "9PM", "10PM",
]

VALID_SORTEOS = {f"Anguilla {slot}" for slot in ANGUILLA_HOURLY_SLOTS}

# Rutas absolutas, calculadas igual que en runner.py (este archivo vive en src/,
# por lo que BASE_DIR es la raiz del repo). Asi no importa desde donde se
# ejecute el proceso ni cual sea el cwd: siempre se lee/escribe el mismo CSV
# que usa runner.py.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CSV_PATH = os.path.join(DATA_DIR, "anguilla_hourly_history.csv")
XLSX_PATH = os.path.join(DATA_DIR, "Anguilla history.xlsx")
XLSX_SHEET = "history"

COLUMNS = [
    "fecha", "sorteo", "hora",
    "primero", "segundo", "tercero",
    "fuente", "source_url", "capturado_rd",
    "status", "raw_date_hint", "notes",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rd_now() -> datetime:
    return datetime.now(TZ_RD)


def normalize_2d(value: str) -> str:
    s = str(value).strip()
    if s.isdigit():
        return s.zfill(2)
    return s


def build_daily_url(target_date: date) -> str:
    return f"{BASE_URL}/resultados-loterias-{target_date.isoformat()}"


def fetch_html(url: str, timeout: int = 25) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extraer_sorteos_de_pagina(html: str, debug: bool = False) -> dict:
    """
    Extrae resultados usando data-lottery-name y result-number.
    Esta es la lógica que SÍ funciona (tomada de scraper.py).
    Devuelve un dict {nombre_sorteo: [p1, p2, p3]}.
    """
    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.find_all(attrs={"data-lottery-name": True})

    if debug:
        nombres = [b.get("data-lottery-name") for b in blocks]
        print(f"  [DEBUG] Bloques encontrados: {nombres}")

    resultados = {}
    for block in blocks:
        nombre = block.get("data-lottery-name", "").strip()
        if nombre not in VALID_SORTEOS:
            continue

        num_divs = block.find_all("div", class_="result-number")
        numeros = []
        for d in num_divs:
            txt = d.get_text(strip=True)
            if re.match(r"^\d{1,2}$", txt):
                numeros.append(str(int(txt)).zfill(2))

        if len(numeros) >= 3:
            resultados[nombre] = numeros[:3]
            if debug:
                print(f"  [DEBUG] {nombre}: {numeros[:3]}")
        else:
            if debug:
                print(f"  [DEBUG] {nombre}: numeros insuficientes {numeros}")

    return resultados


def scrape_day(target_date: date, sleep_sec: float = 0.2, debug: bool = False) -> pd.DataFrame:
    """
    IMPORTANTE:
    Solo devuelve filas OK encontradas.
    Si un slot aún no ha salido, NO se guarda.
    """
    url = build_daily_url(target_date)

    try:
        html = fetch_html(url)
        resultados = extraer_sorteos_de_pagina(html, debug=debug)

        if not resultados:
            return pd.DataFrame(columns=COLUMNS)

        target_iso = target_date.isoformat()
        rows = []
        for sorteo, nums in resultados.items():
            slot = sorteo.replace("Anguilla ", "").strip()
            rows.append({
                "fecha": target_iso,
                "sorteo": sorteo,
                "hora": slot,
                "primero": nums[0],
                "segundo": nums[1],
                "tercero": nums[2],
                "fuente": "enloteria_daily",
                "source_url": url,
                "capturado_rd": rd_now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "OK",
                "raw_date_hint": target_iso,
                "notes": "",
            })

        time.sleep(sleep_sec)
        return pd.DataFrame(rows)

    except Exception as e:
        print(f"SCRAPE_DAY ERROR {target_date}: {e}")
        return pd.DataFrame(columns=COLUMNS)


def load_existing_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_csv(path, dtype=str).fillna("")
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[COLUMNS]


def dedupe_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    tmp = df.copy()
    tmp["capturado_rd_sort"] = pd.to_datetime(tmp["capturado_rd"], errors="coerce")

    tmp = tmp.sort_values(
        by=["fecha", "sorteo", "capturado_rd_sort"],
        ascending=[True, True, False]
    )

    tmp = tmp.drop_duplicates(subset=["fecha", "sorteo"], keep="first")
    tmp = tmp.drop(columns=["capturado_rd_sort"], errors="ignore")
    tmp = tmp.sort_values(by=["fecha", "hora"], ascending=[True, True]).reset_index(drop=True)

    return tmp


def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_xlsx(df: pd.DataFrame, path: str, sheet_name: str = "history") -> None:
    ensure_dir(os.path.dirname(path))

    out = df.copy()
    for col in ["primero", "segundo", "tercero"]:
        out[col] = out[col].astype(str).apply(
            lambda x: normalize_2d(x) if x.strip().isdigit() else x
        )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]

        for row in ws.iter_rows():
            for cell in row:
                cell.number_format = "@"

        widths = {
            "A": 12, "B": 18, "C": 8, "D": 10, "E": 10, "F": 10,
            "G": 16, "H": 60, "I": 20, "J": 12, "K": 14, "L": 40
        }
        for col, width in widths.items():
            ws.column_dimensions[col].width = width


def update_history_with_day(target_date: date, debug: bool = False) -> pd.DataFrame:
    existing = load_existing_csv(CSV_PATH)
    fresh = scrape_day(target_date, debug=debug)

    if fresh.empty:
        # No guardar nada si no apareció ningún resultado nuevo
        return fresh

    combined = pd.concat([existing, fresh], ignore_index=True).fillna("")
    combined = combined[combined["status"] == "OK"].copy()
    combined = dedupe_history(combined)

    save_csv(combined, CSV_PATH)
    save_xlsx(combined, XLSX_PATH, XLSX_SHEET)

    return fresh


def backfill_days(days_back: int, pause_sec: float = 0.25) -> pd.DataFrame:
    all_new = []
    today_rd = rd_now().date()

    for i in range(days_back):
        d = today_rd - timedelta(days=i)
        daily = update_history_with_day(d)
        if not daily.empty:
            all_new.append(daily)
        print(f"[{i + 1}/{days_back}] {d} procesado")
        time.sleep(pause_sec)

    if all_new:
        return pd.concat(all_new, ignore_index=True)

    return pd.DataFrame(columns=COLUMNS)


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Sin resultados nuevos para guardar.")
        return

    cols = ["fecha", "sorteo", "primero", "segundo", "tercero", "status", "notes"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    ensure_dir(DATA_DIR)
    ensure_dir(OUT_DIR)

    if len(sys.argv) == 1:
        fresh = backfill_days(7)
        print_summary(fresh.tail(50))
        print("\n✅ Backfill completado: 7 días")
        print(f"✅ Actualizado: {CSV_PATH}")
        print(f"✅ Actualizado: {XLSX_PATH}")

    elif len(sys.argv) == 3 and sys.argv[1].lower() == "day":
        target = datetime.strptime(sys.argv[2], "%Y-%m-%d").date()
        fresh = update_history_with_day(target, debug=True)
        print_summary(fresh)
        print(f"\n✅ Actualizado: {CSV_PATH}")
        print(f"✅ Actualizado: {XLSX_PATH}")

    elif len(sys.argv) == 3 and sys.argv[1].lower() == "backfill":
        days_back = int(sys.argv[2])
        fresh = backfill_days(days_back)
        print_summary(fresh.tail(50))
        print(f"\n✅ Backfill completado: {days_back} días")
        print(f"✅ Actualizado: {CSV_PATH}")
        print(f"✅ Actualizado: {XLSX_PATH}")

    else:
        print("Uso:")
        print("  python scrape_anguilla_enloteria.py")
        print("  python scrape_anguilla_enloteria.py day 2026-03-28")
        print("  python scrape_anguilla_enloteria.py backfill 365")