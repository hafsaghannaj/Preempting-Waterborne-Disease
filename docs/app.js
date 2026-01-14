const themeToggle = document.getElementById('themeToggle');
const rootEl = document.documentElement;
const storedTheme = localStorage.getItem('theme');
if (storedTheme) {
  rootEl.dataset.theme = storedTheme;
}
if (themeToggle) {
  themeToggle.addEventListener('click', () => {
    const next = rootEl.dataset.theme === 'dark' ? 'light' : 'dark';
    rootEl.dataset.theme = next;
    localStorage.setItem('theme', next);
  });
}

const map = L.map('map', { scrollWheelZoom: false });
const baseLayers = {
  "Light": L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', { maxZoom: 19 }),
  "Streets": L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19 }),
  "Satellite": L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 19 })
};
baseLayers.Light.addTo(map);

const markerLayer = L.layerGroup().addTo(map);
const heatLayer = L.heatLayer([], { radius: 18, blur: 20, maxZoom: 6 });
const overlays = { "Hotspots": markerLayer, "Heat": heatLayer };
L.control.layers(baseLayers, overlays).addTo(map);

const timeSlider = document.getElementById('timeSlider');
const timeLabel = document.getElementById('timeLabel');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdLabel = document.getElementById('thresholdLabel');
const alertCount = document.getElementById('alertCount');

const latInput = document.getElementById('latInput');
const lonInput = document.getElementById('lonInput');
const dateInput = document.getElementById('dateInput');
const thresholdInput = document.getElementById('thresholdInput');
const scoreButton = document.getElementById('scoreButton');
const scoreValue = document.getElementById('scoreValue');
const intervalValue = document.getElementById('intervalValue');

let points = [];
let dates = [];

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(',');
  return lines.map(line => {
    const values = line.split(',');
    const row = {};
    headers.forEach((header, idx) => {
      row[header] = values[idx];
    });
    return row;
  });
}

function setMapView() {
  if (points.length === 0) {
    map.setView([0.5, 32.5], 4);
    return;
  }
  const lat = points.reduce((sum, p) => sum + p.lat, 0) / points.length;
  const lon = points.reduce((sum, p) => sum + p.lon, 0) / points.length;
  map.setView([lat, lon], 5);
}

function render() {
  const date = dates[parseInt(timeSlider.value, 10)] || null;
  const threshold = parseFloat(thresholdSlider.value);
  thresholdLabel.textContent = threshold.toFixed(0);
  thresholdInput.value = threshold.toFixed(0);
  timeLabel.textContent = date || '--';
  if (!dateInput.value && date) {
    dateInput.value = date;
  }

  markerLayer.clearLayers();
  heatLayer.setLatLngs([]);

  let alertTotal = 0;
  const heatPoints = [];

  points.filter(p => !date || p.date === date).forEach(p => {
    const isAlert = p.risk_score >= threshold;
    if (isAlert) {
      alertTotal += 1;
    }
    const color = isAlert ? '#d1495b' : p.risk_score >= 40 ? '#f4b860' : '#2d7d7d';
    const marker = L.circleMarker([p.lat, p.lon], {
      radius: isAlert ? 7 : 5,
      color,
      fillColor: color,
      fillOpacity: 0.8,
    }).bindPopup(`Risk: ${p.risk_score.toFixed(1)}<br/>${p.date}`);
    markerLayer.addLayer(marker);
    heatPoints.push([p.lat, p.lon, p.risk_score / 100]);
  });

  heatLayer.setLatLngs(heatPoints);
  alertCount.textContent = `${alertTotal} hotspots`;
}

function syncThreshold() {
  thresholdSlider.value = thresholdInput.value || 70;
  render();
}

function estimateRisk() {
  const lat = parseFloat(latInput.value);
  const lon = parseFloat(lonInput.value);
  if (!points.length || Number.isNaN(lat) || Number.isNaN(lon)) {
    return;
  }
  const date = dateInput.value || dates[dates.length - 1];
  const sameDate = points.filter(p => p.date === date);
  if (!sameDate.length) {
    scoreValue.textContent = '—';
    intervalValue.textContent = 'Prediction interval unavailable';
    return;
  }
  let best = sameDate[0];
  let bestDist = Number.MAX_VALUE;
  sameDate.forEach(p => {
    const dist = Math.hypot(p.lat - lat, p.lon - lon);
    if (dist < bestDist) {
      best = p;
      bestDist = dist;
    }
  });
  scoreValue.textContent = best.risk_score.toFixed(1);
  if (best.interval_lower && best.interval_upper) {
    intervalValue.textContent = `${best.interval_lower.toFixed(1)} to ${best.interval_upper.toFixed(1)} expected range`;
  } else {
    intervalValue.textContent = 'Prediction interval unavailable';
  }
}

function initControls() {
  timeSlider.addEventListener('input', render);
  thresholdSlider.addEventListener('input', render);
  thresholdInput.addEventListener('change', syncThreshold);
  scoreButton.addEventListener('click', estimateRisk);
}

fetch('data/risk_scored_points.csv')
  .then(resp => resp.text())
  .then(text => {
    const rows = parseCSV(text);
    points = rows.map(row => ({
      lat: parseFloat(row.lat),
      lon: parseFloat(row.lon),
      date: row.date,
      risk_score: parseFloat(row.risk_score),
      interval_lower: row.interval_lower ? parseFloat(row.interval_lower) : null,
      interval_upper: row.interval_upper ? parseFloat(row.interval_upper) : null,
    }));
    dates = Array.from(new Set(points.map(p => p.date))).sort();
    timeSlider.max = Math.max(dates.length - 1, 0);
    timeSlider.value = dates.length ? dates.length - 1 : 0;
    setMapView();
    initControls();
    render();
  })
  .catch(() => {
    setMapView();
  });
