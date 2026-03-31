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

SLOT_TO_CLOCK = {
    "8AM": "8:00AM",
    "9AM": "9:00AM",
    "10AM": "10:00AM",
    "11AM": "11:00AM",
    "12PM": "12:00PM",
    "1PM": "1:00PM",
    "2PM": "2:00PM",
    "3PM": "3:00PM",
    "4PM": "4:00PM",
    "5PM": "5:00PM",
    "6PM": "6:00PM",
    "7PM": "7:00PM",
    "8PM": "8:00PM",
    "9PM": "9:00PM",
    "10PM": "10:00PM",
}

MESES_ES = {
    "ene": 1, "enero": 1,
    "feb": 2, "febrero": 2,
    "mar": 3, "marzo": 3,
    "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "junio": 6,
    "jul": 7, "julio": 7,
    "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "set": 9, "septiembre": 9, "setiembre": 9,
    "oct": 10, "octubre": 10,
    "nov": 11, "noviembre": 11,
    "dic": 12, "diciembre": 12,
}

DATA_DIR = "data"
OUT_DIR = "outputs"
CSV_PATH = os.path.join(DATA_DIR, "anguilla_hourly_history.csv")
XLSX_PATH = os.path.join(DATA_DIR, "Anguilla history.xlsx")
XLSX_SHEET = "history"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rd_now() -> datetime:
    return datetime.now(TZ_RD)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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


def parse_es_date(text: str) -> str | None:
    """
    Ej:
      'Sáb 28 de marzo, 2026'
      'Sab 28 de marzo, 2026'
      'Lunes 30 de marzo, 2026'
    -> '2026-03-28'
    """
    t = clean_text(text).lower()
    t = (
        t.replace("á", "a")
         .replace("é", "e")
         .replace("í", "i")
         .replace("ó", "o")
         .replace("ú", "u")
    )

    m = re.search(r"(\d{1,2})\s+de\s+([a-z]+),\s*(20\d{2})", t)
    if not m:
        return None

    day = int(m.group(1))
    month_name = m.group(2)
    year = int(m.group(3))

    month = MESES_ES.get(month_name)
    if not month:
        return None

    return f"{year:04d}-{month:02d}-{day:02d}"


def html_to_lines(html: str) -> list[str]:
    """
    Convierte el HTML a líneas de texto simples.
    Este formato se parece mucho a lo que vemos en la página abierta con web.
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [clean_text(x) for x in text.splitlines()]
    lines = [x for x in lines if x]
    return lines


def is_anguilla_title(line: str) -> bool:
    return clean_text(line) in VALID_SORTEOS


def is_clock(line: str) -> bool:
    return re.fullmatch(r"[0-9]{1,2}:\d{2}[AP]M", clean_text(line).upper()) is not None


def is_ball_number(line: str) -> bool:
    return re.fullmatch(r"[0-9]{1,2}", clean_text(line)) is not None


def extract_anguilla_blocks_from_lines(lines: list[str], target_date: date) -> list[dict]:
    """
    Parseo por máquina de estados:
    Anguilla 8AM
    ...
    Sáb 28 de marzo, 2026
    8:00AM
    ...
    90
    ...
    97
    ...
    70
    """
    target_iso = target_date.isoformat()
    rows = []
    seen = set()

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        if not is_anguilla_title(line):
            i += 1
            continue

        sorteo = clean_text(line)
        slot = sorteo.replace("Anguilla ", "").strip()
        expected_clock = SLOT_TO_CLOCK.get(slot)

        # Buscar fecha/hora/números dentro de una ventana local
        date_found = None
        hour_found = None
        nums = []

        j = i + 1
        max_j = min(i + 35, n)

        while j < max_j:
            cur = lines[j]

            # Si ya arrancó otro bloque de lotería, cortamos
            if j > i + 1 and cur.startswith("Anguilla ") and cur != sorteo:
                break

            maybe_date = parse_es_date(cur)
            if maybe_date:
                date_found = maybe_date

            if is_clock(cur):
                hour_found = cur.upper()

            if is_ball_number(cur):
                nums.append(normalize_2d(cur))
                if len(nums) == 3:
                    break

            j += 1

        if date_found == target_iso and hour_found == expected_clock and len(nums) >= 3:
            key = (target_iso, sorteo)
            if key not in seen:
                seen.add(key)
                rows.append({
                    "fecha": target_iso,
                    "sorteo": sorteo,
                    "hora": slot,
                    "primero": nums[0],
                    "segundo": nums[1],
                    "tercero": nums[2],
                    "fuente": "enloteria_daily",
                    "source_url": "",
                    "capturado_rd": rd_now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "OK",
                    "raw_date_hint": target_iso,
                    "notes": "",
                })

        i += 1

    return rows


def scrape_day(target_date: date, sleep_sec: float = 0.2) -> pd.DataFrame:
    url = build_daily_url(target_date)

    base_rows = []
    for slot in ANGUILLA_HOURLY_SLOTS:
        base_rows.append({
            "fecha": target_date.isoformat(),
            "sorteo": f"Anguilla {slot}",
            "hora": slot,
            "primero": "",
            "segundo": "",
            "tercero": "",
            "fuente": "enloteria_daily",
            "source_url": url,
            "capturado_rd": rd_now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "NOT_FOUND",
            "raw_date_hint": "",
            "notes": "No apareció en página diaria",
        })

    try:
        html = fetch_html(url)
        lines = html_to_lines(html)
        found_blocks = extract_anguilla_blocks_from_lines(lines, target_date)
        found_map = {(r["fecha"], r["sorteo"]): r for r in found_blocks}

        rows = []
        for row in base_rows:
            key = (row["fecha"], row["sorteo"])
            if key in found_map:
                found = found_map[key].copy()
                found["source_url"] = url
                rows.append(found)
            else:
                rows.append(row)

        time.sleep(sleep_sec)
        return pd.DataFrame(rows)

    except requests.HTTPError as e:
        for row in base_rows:
            row["status"] = "ERROR"
            row["notes"] = f"HTTPError: {e}"
        return pd.DataFrame(base_rows)

    except Exception as e:
        for row in base_rows:
            row["status"] = "ERROR"
            row["notes"] = f"Error: {e}"
        return pd.DataFrame(base_rows)


def load_existing_csv(path: str) -> pd.DataFrame:
    cols = [
        "fecha", "sorteo", "hora",
        "primero", "segundo", "tercero",
        "fuente", "source_url", "capturado_rd",
        "status", "raw_date_hint", "notes"
    ]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path, dtype=str).fillna("")
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]


def dedupe_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    priority = {
        "OK": 4,
        "NOT_FOUND": 2,
        "ERROR": 1,
    }

    tmp = df.copy()
    tmp["priority"] = tmp["status"].map(priority).fillna(0).astype(int)
    tmp["capturado_rd_sort"] = pd.to_datetime(tmp["capturado_rd"], errors="coerce")

    tmp = tmp.sort_values(
        by=["fecha", "sorteo", "priority", "capturado_rd_sort"],
        ascending=[True, True, False, False]
    )

    tmp = tmp.drop_duplicates(subset=["fecha", "sorteo"], keep="first")
    tmp = tmp.drop(columns=["priority", "capturado_rd_sort"], errors="ignore")
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

        # Solo conservar históricos buenos
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


def update_history_with_day(target_date: date) -> pd.DataFrame:
    existing = load_existing_csv(CSV_PATH)
    fresh = scrape_day(target_date)

    combined = pd.concat([existing, fresh], ignore_index=True).fillna("")
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
        all_new.append(daily)
        print(f"[{i + 1}/{days_back}] {d} procesado")
        time.sleep(pause_sec)

    if all_new:
        return pd.concat(all_new, ignore_index=True)
    return pd.DataFrame()


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Sin resultados.")
        return

    cols = ["fecha", "sorteo", "primero", "segundo", "tercero", "status", "notes"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    ensure_dir(DATA_DIR)
    ensure_dir(OUT_DIR)

    # Uso:
    # python scrape_anguilla_enloteria.py
    # python scrape_anguilla_enloteria.py day 2026-03-28
    # python scrape_anguilla_enloteria.py backfill 30

    if len(sys.argv) == 1:
        # default rápido para probar
        fresh = backfill_days(7)
        print_summary(fresh.tail(50))
        print("\n✅ Backfill completado: 7 días")
        print(f"✅ Actualizado: {CSV_PATH}")
        print(f"✅ Actualizado: {XLSX_PATH}")

    elif len(sys.argv) == 3 and sys.argv[1].lower() == "day":
        target = datetime.strptime(sys.argv[2], "%Y-%m-%d").date()
        fresh = update_history_with_day(target)
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