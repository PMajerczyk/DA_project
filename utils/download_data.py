"""Download earthquake data from the USGS Earthquake Catalog API for Japan.

The USGS FDSN event API returns at most 20 000 records per query, so we download
year-by-year (2000-2023) and concatenate. A short pause between requests keeps us
polite towards the public API.

Region: Japan bounding box (lat 24-50N, lon 122-154E), magnitude >= 4.0.
"""
import time
import pandas as pd

BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"
PARAMS = (
    "?format=csv&minmagnitude=4.0&orderby=time"
    "&minlatitude=24&maxlatitude=50"
    "&minlongitude=122&maxlongitude=154"
)
OUT_PATH = "data/raw/earthquakes_japan.csv"


def download(start_year: int = 2000, end_year: int = 2023) -> pd.DataFrame:
    frames = []
    for year in range(start_year, end_year + 1):
        url = f"{BASE}{PARAMS}&starttime={year}-01-01&endtime={year}-12-31"
        try:
            df = pd.read_csv(url)
            frames.append(df)
            print(f"{year}: {len(df)} events")
            time.sleep(1)  # polite API usage
        except Exception as e:  # noqa: BLE001
            print(f"{year}: ERROR {e}")
    all_events = pd.concat(frames, ignore_index=True)
    all_events.to_csv(OUT_PATH, index=False)
    print(f"\nTotal: {len(all_events)} events saved to {OUT_PATH}")
    return all_events


if __name__ == "__main__":
    df = download()
    print("\nShape:", df.shape)
    print(df.head())
