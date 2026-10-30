"""Exportação das trajetórias para KML 3D (Google Earth), com tempo para reproduzir a missão."""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from multi_robot_system.core.data.trajectory import MissionDataset
from multi_robot_system.core.math.geodesy import ned_to_ecef, ecef_to_geodetic


def hex_to_kml_color(hex_color: str, alpha: str = "ff") -> str:
    """Converte uma cor #RRGGBB para o formato de cor do KML (aabbggrr)."""
    clean = hex_color.lstrip("#")
    if len(clean) != 6:
        return f"{alpha}ffffff"
    r = clean[0:2]
    g = clean[2:4]
    b = clean[4:6]
    return f"{alpha}{b}{g}{r}"


def export_mission_to_kml(
    dataset: MissionDataset,
    output_path: str,
    subsample_step: int = 8
) -> str:
    """Esta função gera o ficheiro KML com um percurso temporal (gx:Track) por robô."""
    lat0, lon0, h0 = dataset.origin

    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"
     xmlns:gx="http://www.google.com/kml/ext/2.2"
     xmlns:kml="http://www.opengis.net/kml/2.2"
     xmlns:atom="http://www.w3.org/2005/Atom">
<Document>
    <name>TOMSA 2026 - Missão Multi-Robô Subaquático (Aguçadoura)</name>
    <description>Trajetórias 3D suavizadas com Filtro de Kalman e Suavizador RTS (INESC TEC).</description>
"""

    styles = []
    placemarks = []

    # Para cada robô: um estilo de cor e o percurso com tempo e posição (NED convertido para lon/lat/alt)
    for r_id, traj in dataset.trajectories.items():
        clean_id = r_id.replace(" ", "_").lower()
        kml_col = hex_to_kml_color(traj.color, alpha="e6")

        styles.append(f"""
    <Style id="style_{clean_id}">
        <LineStyle>
            <color>{kml_col}</color>
            <width>4</width>
        </LineStyle>
        <IconStyle>
            <scale>1.2</scale>
        </IconStyle>
    </Style>""")

        N_pts = len(traj.timestamps)
        indices = np.arange(0, N_pts, subsample_step)
        if indices[-1] != N_pts - 1:
            indices = np.append(indices, N_pts - 1)

        when_tags = []
        coord_tags = []

        for idx in indices:
            t = traj.timestamps[idx]
            dt_iso = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            when_tags.append(f"            <when>{dt_iso}</when>")

            N = traj.pos_ned[0, idx]
            E = traj.pos_ned[1, idx]
            D = traj.pos_ned[2, idx]

            X, Y, Z = ned_to_ecef(N, E, D, lat0, lon0, h0)
            lat, lon, h = ecef_to_geodetic(X, Y, Z)

            coord_tags.append(f"            <gx:coord>{lon:.7f} {lat:.7f} {h:.2f}</gx:coord>")

        when_str = "\n".join(when_tags)
        coord_str = "\n".join(coord_tags)

        placemarks.append(f"""
    <Placemark>
        <name>{traj.label}</name>
        <styleUrl>#style_{clean_id}</styleUrl>
        <gx:Track>
            <altitudeMode>absolute</altitudeMode>
{when_str}
{coord_str}
        </gx:Track>
    </Placemark>""")

    kml_footer = """
</Document>
</kml>
"""

    full_kml = kml_header + "".join(styles) + "".join(placemarks) + kml_footer
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_kml)

    return output_path
