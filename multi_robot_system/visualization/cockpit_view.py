"""Módulo de Renderização do Cockpit / Sala de Controlo do Submarino (3ª Vista).

Gera o painel instrumental avançado de navegação e controlo tático para o operador,
incluindo:
1. Horizonte Artificial (Pitch & Roll).
2. Bússola Giroscópica (Heading).
3. Profundímetro e Distância ao Fundo.
4. Velocímetro Longitudinal e Taxa de Viragem (Turn Rate).
5. Radar Tático AER com posições relativas dos outros submarinos.
6. Painel Anunciador de Risco de Colisão Crítico (< 10m) e Alertas CPA.
"""

from typing import Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from multi_robot_system.core.math.proximity import CPAResult


def create_cockpit_figure(
    state: Dict,
    cpa_results: List[CPAResult],
    mission_time_s: float,
    max_depth_m: float = 35.0
) -> go.Figure:
    """Cria o painel de instrumentos virtuais (Cockpit / Sala de Controlo) para o robô selecionado."""
    r_id = state["robot_id"]
    color = state["color"]
    roll_deg, pitch_deg, yaw_deg = state["euler_deg"]
    depth_m = state["depth_m"]
    speed_mps = state["speed_longitudinal"]
    turn_rate = state["turn_rate_yaw"]

    # Criação do layout em grelha com tema tático militar/industrial
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "polar"}, {"type": "indicator"}, {"type": "indicator"}]
        ],
        subplot_titles=(
            f"<b>ATITUDE & ORIENTAÇÃO ({r_id})</b>",
            "<b>PROFUNDÍMETRO SUBMARINO</b>",
            "<b>VELOCÍMETRO LONGITUDINAL</b>",
            "<b>RADAR TÁTICO AER (VISADA)</b>",
            "<b>TAXA DE VIRAGEM (YAW RATE)</b>",
            "<b>SEGURANÇA CPA & ALERTA COLISÃO</b>"
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )

    # 1. Horizonte Artificial / Heading (Compass)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=yaw_deg % 360.0,
            number={"suffix": "° PROA", "font": {"color": "#48cae4", "size": 22}},
            gauge={
                "axis": {"range": [0, 360], "tickmode": "array", "tickvals": [0, 90, 180, 270],
                         "ticktext": ["N", "E", "S", "W"], "tickfont": {"color": "#ffffff"}},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "#1b263b",
                "borderwidth": 2,
                "bordercolor": "#415a77",
                "steps": [
                    {"range": [0, 360], "color": "#0d1b2a"}
                ]
            },
            title={"text": f"Roll: {roll_deg:+.1f}° | Pitch: {pitch_deg:+.1f}°", "font": {"color": "#e0e1dd", "size": 14}}
        ),
        row=1, col=1
    )

    # 2. Profundímetro Digital
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=depth_m,
            number={"suffix": " m", "font": {"color": "#00b4d8", "size": 26}},
            gauge={
                "axis": {"range": [0, max_depth_m], "tickfont": {"color": "#ffffff"}},
                "bar": {"color": "#0077b6", "thickness": 0.3},
                "bgcolor": "#1b263b",
                "threshold": {
                    "line": {"color": "#e63946", "width": 4},
                    "thickness": 0.75,
                    "value": max_depth_m - 3.0
                }
            },
            title={"text": "Profundidade Real (Z)", "font": {"color": "#e0e1dd", "size": 14}}
        ),
        row=1, col=2
    )

    # 3. Velocímetro Longitudinal
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=speed_mps,
            number={"suffix": " m/s", "font": {"color": "#52b788", "size": 26}},
            gauge={
                "axis": {"range": [-0.5, 3.0], "tickfont": {"color": "#ffffff"}},
                "bar": {"color": "#2d6a4f", "thickness": 0.3},
                "bgcolor": "#1b263b",
                "steps": [
                    {"range": [-0.5, 0.0], "color": "#780000"},
                    {"range": [0.0, 3.0], "color": "#081c15"}
                ]
            },
            title={"text": "Avanço Longitudinal", "font": {"color": "#e0e1dd", "size": 14}}
        ),
        row=1, col=3
    )

    # 4. Radar Tático Polar AER (Robôs visíveis no horizonte do corpo)
    target_thetas = []
    target_rs = []
    target_texts = []
    target_colors = []

    for cpa in cpa_results:
        other_id = cpa.robot_j if cpa.robot_i == r_id else (cpa.robot_i if cpa.robot_j == r_id else None)
        if other_id is not None:
            az_deg, el_deg, rng_m = cpa.relative_aer
            target_thetas.append(az_deg % 360.0)
            target_rs.append(rng_m)
            target_texts.append(f"{other_id}<br>Dist: {rng_m:.1f}m<br>CPA: {cpa.cpa_distance_m:.1f}m")
            target_colors.append("#e63946" if cpa.is_collision_risk else "#52b788")

    fig.add_trace(
        go.Scatterpolar(
            r=target_rs,
            theta=target_thetas,
            mode="markers+text",
            text=[t.split("<br>")[0] for t in target_texts],
            textposition="top center",
            hovertext=target_texts,
            hoverinfo="text",
            marker=dict(size=14, color=target_colors, symbol="triangle-up", line=dict(color="#ffffff", width=2)),
            showlegend=False
        ),
        row=2, col=1
    )

    # 5. Taxa de Viragem (Turn Rate)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=turn_rate,
            number={"suffix": " °/s", "font": {"color": "#f4a261", "size": 24}},
            gauge={
                "axis": {"range": [-45.0, 45.0], "tickfont": {"color": "#ffffff"}},
                "bar": {"color": "#e76f51", "thickness": 0.3},
                "bgcolor": "#1b263b"
            },
            title={"text": "Velocidade Angular Yaw", "font": {"color": "#e0e1dd", "size": 14}}
        ),
        row=2, col=2
    )

    # 6. Painel de Alarme CPA / Segurança (< 10m)
    min_dist = min([c.current_distance_m for c in cpa_results]) if cpa_results else 999.0
    min_cpa = min([c.cpa_distance_m for c in cpa_results]) if cpa_results else 999.0
    collision_imminent = any(c.is_collision_risk for c in cpa_results)

    status_color = "#e63946" if collision_imminent else "#2a9d8f"
    status_msg = "PERIGO DE COLISÃO (< 10m)" if collision_imminent else "ROTA SEGURA"

    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=min_dist,
            number={"suffix": " m", "font": {"color": status_color, "size": 30}},
            delta={"reference": 10.0, "position": "bottom", "increasing": {"color": "#2a9d8f"}, "decreasing": {"color": "#e63946"}},
            title={"text": f"<b>{status_msg}</b><br><span style='font-size:0.8em;color:#e0e1dd'>Menor Distância CPA: {min_cpa:.1f}m</span>",
                   "font": {"size": 16, "color": status_color}}
        ),
        row=2, col=3
    )

    # Estilização do Dark Mode Tático
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        font=dict(family="Segoe UI, Roboto, Helvetica, Arial, sans-serif", color="#e0e1dd"),
        margin=dict(l=40, r=40, t=60, b=40),
        polar=dict(
            bgcolor="#111827",
            radialaxis=dict(range=[0, 150], tickfont=dict(color="#9ca3af"), linecolor="#374151"),
            angularaxis=dict(direction="clockwise", rotation=90, tickfont=dict(color="#9ca3af"), linecolor="#374151")
        )
    )

    return fig
