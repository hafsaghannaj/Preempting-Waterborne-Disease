import folium
import geopandas as gpd


def _risk_color(score):
    if score >= 70:
        return "red"
    if score >= 40:
        return "orange"
    return "green"


def generate_risk_map(df, output_html):
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    center_lat = gdf["lat"].mean()
    center_lon = gdf["lon"].mean()

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB positron")

    for _, row in gdf.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color=_risk_color(row["risk_score"]),
            fill=True,
            fill_opacity=0.7,
            popup=f"Risk: {row['risk_score']:.1f}",
        ).add_to(fmap)

    fmap.save(output_html)
