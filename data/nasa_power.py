from datetime import datetime
from urllib.parse import urlencode

import numpy as np
import pandas as pd


POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def build_power_url(lat, lon, start_date, end_date, parameters=None):
    if parameters is None:
        parameters = ["T2M", "PRECTOT"]

    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%Y%m%d")
    if isinstance(end_date, datetime):
        end_date = end_date.strftime("%Y%m%d")

    query = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start": start_date,
        "end": end_date,
        "community": "AG",
        "parameters": ",".join(parameters),
        "format": "JSON",
    }
    return f"{POWER_BASE_URL}?{urlencode(query)}"


def fetch_power_data(lat, lon, start_date, end_date, allow_network=False):
    if not allow_network:
        raise RuntimeError("Network access disabled. Pass allow_network=True to fetch.")

    import json
    from urllib.request import urlopen

    url = build_power_url(lat, lon, start_date, end_date)
    with urlopen(url) as response:
        payload = json.loads(response.read().decode("utf-8"))

    records = payload["properties"]["parameter"]
    dates = list(records[next(iter(records))].keys())
    data = {"date": dates}
    for key, values in records.items():
        data[key.lower()] = [values[date] for date in dates]

    frame = pd.DataFrame(data)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    frame = frame.rename(columns={"t2m": "air_temp", "prectot": "precip"})
    return frame


def load_or_fetch_power_data(
    lat,
    lon,
    start_date,
    end_date,
    cache_path,
    allow_network=False,
):
    if cache_path and pd.io.common.file_exists(cache_path):
        frame = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        frame = fetch_power_data(
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            allow_network=allow_network,
        )
        if cache_path:
            frame.to_csv(cache_path, index=False)

    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def _build_grid(bbox, grid_size):
    lat_vals = np.linspace(bbox["lat_min"], bbox["lat_max"], grid_size)
    lon_vals = np.linspace(bbox["lon_min"], bbox["lon_max"], grid_size)
    return [(float(lat), float(lon)) for lat in lat_vals for lon in lon_vals]


def fetch_power_grid(
    bbox,
    start_date,
    end_date,
    grid_size=4,
    allow_network=False,
    verbose=True,
):
    rows = []
    grid_points = _build_grid(bbox, grid_size)
    iterator = grid_points
    if verbose:
        try:
            from tqdm import tqdm

            iterator = tqdm(grid_points, desc="NASA POWER fetch")
        except ImportError:
            iterator = grid_points

    total = len(grid_points)
    for idx, (lat, lon) in enumerate(iterator, start=1):
        if verbose and iterator is grid_points:
            print(f"NASA POWER fetch {idx}/{total} at {lat:.2f}, {lon:.2f}")
        frame = fetch_power_data(
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            allow_network=allow_network,
        )
        frame["lat"] = lat
        frame["lon"] = lon
        rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=["date", "air_temp", "precip", "lat", "lon"])
    combined = pd.concat(rows, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date
    return combined


def load_or_fetch_power_grid(
    bbox,
    start_date,
    end_date,
    grid_size,
    cache_path,
    allow_network=False,
    verbose=True,
):
    if cache_path and pd.io.common.file_exists(cache_path):
        frame = pd.read_csv(cache_path, parse_dates=["date"])
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        return frame

    frame = fetch_power_grid(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        grid_size=grid_size,
        allow_network=allow_network,
        verbose=verbose,
    )
    if cache_path:
        frame.to_csv(cache_path, index=False)
    return frame
