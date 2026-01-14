import hashlib
import math
from datetime import datetime


DEFAULT_MOCK_BBOX = {
    "lat_min": -10.0,
    "lat_max": 10.0,
    "lon_min": 20.0,
    "lon_max": 50.0,
}


def _seed(lat, lon, date_str):
    token = f"{lat:.3f}:{lon:.3f}:{date_str}"
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)


def mock_chlorophyll(lat, lon, date, bbox=None):
    if bbox is None:
        bbox = DEFAULT_MOCK_BBOX
    date_obj = datetime.strptime(date, "%Y-%m-%d") if isinstance(date, str) else date
    seed = _seed(lat, lon, date_obj.strftime("%Y-%m-%d"))
    seasonal = math.sin(2 * math.pi * (date_obj.timetuple().tm_yday / 365.0))
    lat_norm = (lat - bbox["lat_min"]) / (bbox["lat_max"] - bbox["lat_min"])
    lon_norm = (lon - bbox["lon_min"]) / (bbox["lon_max"] - bbox["lon_min"])
    base = 0.4 + 0.6 * lat_norm + 0.2 * lon_norm
    noise = ((seed % 1000) / 1000.0 - 0.5) * 0.15
    value = base + 0.3 * seasonal + noise
    return max(0.05, min(2.5, value))


def mock_flood_extent(lat, lon, date, bbox=None):
    if bbox is None:
        bbox = DEFAULT_MOCK_BBOX
    date_obj = datetime.strptime(date, "%Y-%m-%d") if isinstance(date, str) else date
    seasonal = math.cos(2 * math.pi * (date_obj.timetuple().tm_yday / 365.0))
    lat_norm = (lat - bbox["lat_min"]) / (bbox["lat_max"] - bbox["lat_min"])
    lon_norm = (lon - bbox["lon_min"]) / (bbox["lon_max"] - bbox["lon_min"])
    base = 0.2 + 0.5 * (1 - lat_norm) + 0.2 * lon_norm
    value = base + 0.3 * seasonal
    return max(0.0, min(1.0, value))
