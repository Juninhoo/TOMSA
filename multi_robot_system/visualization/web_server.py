"""Servidor Web Multithreaded e Dashboard Interativo em Plotly.

Implementa a interface gráfica central do sistema com:
1. Comutação instantânea entre as Vistas: Global 3D, 3ª Pessoa (com radar e estatísticas), Cockpit e Distâncias CPA.
2. Controlo temporal com slider, Play/Pausa e vários níveis de velocidade (até Ultra Rápido em ~2s).
3. Teclas de atalho integradas:
   - '1', '2', '3': Seleciona o robô ativo.
   - 'v': Alterna ciclicamente entre as vistas.
   - '+' / '-' (ou roda do rato): aproxima / afasta a câmara da Vista 3ª Pessoa.
   - 'r': Reinicia a reprodução das trajetórias (reset ao início).
   - 'f': Liga ou desliga a câmara que segue o robô ativo (Vista Global); 'g': volta à Vista Geral.
   - Espaço: Play / Pausa.
4. Geração de bundle HTML 100% autónomo (offline).
"""

import json
import math
import os
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import plotly.io as pio

from multi_robot_system.core.data.trajectory import MissionDataset
from multi_robot_system.core.data.mission_report import compute_mission_report
from multi_robot_system.visualization.plotly_views import (
    create_global_3d_figure,
    create_pov_camera_figure,
    create_proximity_timeline_figure,
    build_submarine_mesh,
    build_submarine_lettering,
    get_scene_geometry,
    BODY_AXIS_LENGTH_M
)
from multi_robot_system.visualization.cockpit_view import create_cockpit_figure


class MissionWebDashboard:
    """Dashboard de controlo e visualização em tempo real."""

    def __init__(self, dataset: MissionDataset, config: dict):
        self.dataset = dataset
        self.config = config
        self.host = config.get("output", {}).get("web_host", "127.0.0.1")
        self.port = config.get("output", {}).get("web_port", 8050)

        self.robots = list(dataset.trajectories.keys())
        self.active_robot = config.get("selected_robot", self.robots[0])
        if self.active_robot not in self.robots:      # o robô escolhido na configuração pode não existir nos dados novos
            self.active_robot = self.robots[0]
        self.active_view = "global"  # 'global', 'pov', 'cockpit'

        # Linha temporal completa da missão: cada robô opera numa janela própria (só se sobrepõem em parte)
        self.t_start = dataset.t0_mission
        self.t_end = dataset.t_end_mission
        self.current_t = self.t_start

        self.is_playing = False
        self.speed_multiplier = 1.0
        self.lock = threading.Lock()

    def build_dashboard_html(self, standalone_t: Optional[float] = None) -> str:
        """Gera a página HTML completa com componentes visuais do Plotly e animação 60 FPS no browser."""
        t_eval = standalone_t if standalone_t is not None else self.current_t
        # Gera as 4 figuras Plotly com IDs estáticos garantidos
        fig_global = create_global_3d_figure(self.dataset, t_eval, self.active_robot)
        fig_pov = create_pov_camera_figure(self.dataset, t_eval, self.active_robot)
        _sp = self.config.get("spatial", {})
        _danger = float(_sp.get("collision_threshold_m", 10.0))
        _warn = max(_danger, float(_sp.get("proximity_warning_m", 30.0)))
        fig_prox = create_proximity_timeline_figure(self.dataset, t_eval, collision_threshold_m=_danger, warning_threshold_m=_warn)

        div_global = pio.to_html(fig_global, div_id="plot-global", full_html=False, include_plotlyjs="cdn")
        div_pov = pio.to_html(fig_pov, div_id="plot-pov", full_html=False, include_plotlyjs=False)
        fig_fp = create_pov_camera_figure(self.dataset, t_eval, self.active_robot)   # mesma cena 3D, câmara em 1ª pessoa feita em JS
        div_fp = pio.to_html(fig_fp, div_id="plot-fp", full_html=False, include_plotlyjs=False)
        div_cockpit = '<div id="cockpit-root"></div>'   # o Cockpit é desenhado no browser (SVG), sem figura Plotly
        div_prox = pio.to_html(fig_prox, div_id="plot-prox", full_html=False, include_plotlyjs=False)
        div_report = '<div id="report-root"></div>'   # o relatório é desenhado no browser a partir dos números de Python

        min_t = float(self.t_start)
        max_t = float(self.t_end)

        # Amostragem da linha temporal COMPLETA da missão para animação no cliente (sem latência de rede)
        dt_sample = 2.0  # passo temporal [s]; o JS interpola linearmente entre amostras
        t_arr = np.arange(min_t, max_t + 1e-6, dt_sample)
        sorted_robots = sorted(self.dataset.trajectories.keys())

        def _unwrap_deg(a):
            # Método: np.unwrap tira os saltos de 360 graus. Obtemos: ângulos contínuos (359 para 1 deixa de ser um salto),
            # necessários para interpolar e alisar o rumo no navegador.
            return np.degrees(np.unwrap(np.radians(a)))

        # Limiares do aviso de proximidade: perigo de colisão (vermelho) e aviso de aproximação (amarelo) [m]
        spatial_cfg = self.config.get("spatial", {})
        danger_m = float(spatial_cfg.get("collision_threshold_m", 10.0))
        warning_m = max(danger_m, float(spatial_cfg.get("proximity_warning_m", 30.0)))

        # O timeline é o único objeto que o JavaScript recebe: uma grelha de tempo fixa (dt_sample) e, por robô, uma
        # série por grandeza (posição NED, ângulos, velocidades, incerteza), mais o relatório, a origem, os limiares
        # e a malha 3D do submarino. O navegador só interpola entre amostras e desenha.
        timeline_dict = {
            "dt_step": dt_sample,
            "t0_base": min_t,
            "t_max": max_t,
            "num_steps": len(t_arr),
            "scene": get_scene_geometry(self.dataset),
            "axis_length_m": BODY_AXIS_LENGTH_M,   # comprimento das setas x, y, z do corpo [m]
            "collision_threshold_m": danger_m,     # abaixo disto: PERIGO DE COLISÃO (vermelho)
            "proximity_warning_m": warning_m,      # abaixo disto: ATENÇÃO, robots próximos (amarelo)
            "contact_distance_m": float(spatial_cfg.get("contact_distance_m", 2.0)),   # abaixo disto: COLISÃO
            # alcance inicial do radar da Vista 3ª Pessoa [m] (botões - / +: 50/100/250/500/1000 m)
            "radar_range_m": float(spatial_cfg.get("radar_range_m", 100.0)),
            "robots": {}
        }

        for r_id in sorted_robots:
            traj = self.dataset.trajectories[r_id]
            ts = traj.timestamps
            f = lambda y: np.interp(t_arr, ts, y)  # fora do intervalo do robô mantém a 1ª/última posição
            # Distância horizontal acumulada desde o início do robô [m] (painel de estatísticas)
            step_len = np.hypot(np.diff(traj.pos_ned[0]), np.diff(traj.pos_ned[1]))
            dist_cum = np.concatenate([[0.0], np.cumsum(step_len)])
            vb = traj.vel_body_meas if traj.vel_body_meas is not None else np.zeros((3, len(ts)))
            sig = traj.sigma_h_decl if traj.sigma_h_decl is not None else np.zeros(len(ts))
            timeline_dict["robots"][r_id] = {
                "label": traj.label,
                "color": traj.color,
                # índices fracionários (em passos) em que o robô começa/termina a operar
                "k0": float((ts[0] - min_t) / dt_sample),
                "k1": float((ts[-1] - min_t) / dt_sample),
                "N": f(traj.pos_ned[0]).round(2).tolist(),
                "E": f(traj.pos_ned[1]).round(2).tolist(),
                "D": f(traj.pos_ned[2]).round(2).tolist(),
                "roll": f(_unwrap_deg(traj.euler_deg[:, 0])).round(2).tolist(),
                "pitch": f(traj.euler_deg[:, 1]).round(2).tolist(),
                "yaw": f(_unwrap_deg(traj.euler_deg[:, 2])).round(2).tolist(),
                "spd": f(traj.speed_longitudinal).round(2).tolist(),
                "turn": f(traj.turn_rate_yaw).round(2).tolist(),
                "dist": f(dist_cum).round(1).tolist(),
                # Cockpit: velocidade estimada em NED, velocidade medida no corpo (twist) e incerteza declarada
                "vN": f(traj.vel_ned[0]).round(3).tolist(),
                "vE": f(traj.vel_ned[1]).round(3).tolist(),
                "vD": f(traj.vel_ned[2]).round(3).tolist(),
                "u": f(np.nan_to_num(vb[0])).round(3).tolist(),
                "v": f(np.nan_to_num(vb[1])).round(3).tolist(),
                "w": f(np.nan_to_num(vb[2])).round(3).tolist(),
                "sig": f(sig).round(4).tolist(),
                # saltos de posição do registo bruto: instante desde o início da missão [s], tamanho [m], se estava imerso
                "jumps": [{"t": round(float(e["t"] - min_t), 1), "m": round(float(e["m"]), 1), "sub": bool(e["depth_m"] > 3.0)}
                          for e in (traj.jump_events or [])],
            }
        # Origem em uso e relatório da missão (encontros, resumo por par, saltos)
        og = self.config.get("origin_resolved") or {}
        if og:
            mode_txt = "média dos dados" if og.get("mode") == "data_mean" else "Surface Reference oficial"
            origin_txt = f"{mode_txt} (lat {og['latitude']:.6f}°, lon {og['longitude']:.6f}°, altura {og['altitude']:.2f} m)"
        else:
            origin_txt = "não indicada"
        timeline_dict["origin"] = {"text": origin_txt}
        timeline_dict["sub_mesh"] = build_submarine_mesh()
        timeline_dict["sub_text"] = build_submarine_lettering()
        timeline_dict["report"] = compute_mission_report(self.dataset, danger_m, warning_m)
        timeline_json = json.dumps(timeline_dict)

        template = """<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="utf-8">
    <title>TOMSA 2026 - Centro de Comando e Visualização Multi-Robô Subaquático</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: #0b0f19;
            color: #e0e1dd;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            overflow-x: hidden;
            /* coluna flexível: a área das vistas ocupa o espaço que sobra, mesmo quando o cabeçalho passa a 2 linhas */
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        header, .controls-bar, .help-banner { flex: 0 0 auto; }
        header {
            background: linear-gradient(180deg, #111827 0%, #0b0f19 100%);
            border-bottom: 1px solid #1f2937;
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .brand { display: flex; align-items: center; gap: 12px; }
        .badge { background: #0077b6; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }
        .nav-buttons, .speed-buttons { display: flex; gap: 8px; align-items: center; }
        button {
            background: #1f2937;
            color: #f3f4f6;
            border: 1px solid #374151;
            padding: 8px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.15s ease;
        }
        button:hover { background: #374151; border-color: #4b5563; }
        button.active { background: #0077b6; border-color: #0096c7; color: white; }
        button.danger { background: #9b2226; border-color: #ae2012; }
        button.danger:hover { background: #ae2012; }
        
        .controls-bar {
            background: #111827;
            border-bottom: 1px solid #1f2937;
            padding: 10px 24px;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .slider-box { flex: 1; display: flex; align-items: center; gap: 12px; }
        .slider-track-wrap { position: relative; flex: 1; display: flex; align-items: center; }
        .slider-track-wrap input[type=range] { width: 100%; }
        /* Marcas de proximidade por baixo da barra do tempo (--c = posição do centro, --w = largura, de 0 a 1) */
        #prox-marks { position: absolute; left: 0; right: 0; top: 0; bottom: 0; pointer-events: none; }
        .prox-mark {
            position: absolute;
            top: calc(50% + 11px);
            left: calc(10px + (100% - 20px) * var(--c));
            width: calc((100% - 20px) * var(--w));
            min-width: 8px;
            height: 9px;
            transform: translateX(-50%);
            border-radius: 2px;
            pointer-events: auto;
            cursor: pointer;
        }
        .prox-mark.warning { background: #f4a261; }
        .prox-mark.danger { background: #e63946; height: 12px; min-width: 10px; }
        .prox-mark:hover { filter: brightness(1.3); }
        input[type=range] {
            flex: 1;
            accent-color: #00b4d8;
            height: 6px;
            cursor: pointer;
        }
        .time-label { font-family: monospace; font-size: 14px; color: #48cae4; min-width: 90px; font-weight: bold; }
        
        .main-container {
            padding: 16px 24px;
            flex: 1 1 auto;
            min-height: 0;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .view-panel {
            width: 100%;
            height: 100%;
            display: none;
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 8px;
            overflow: hidden;
        }
        .view-panel.active { display: flex; flex-direction: column; }
        .view-panel > div, .js-plotly-plot, .plotly-graph-div, .plot-container, .svg-container { width: 100% !important; height: 100% !important; flex: 1; }
        .main-svg { width: 100% !important; }
        
        .help-banner {
            background: rgba(17, 24, 39, 0.9);
            border-top: 1px solid #1f2937;
            padding: 6px 24px;
            font-size: 12px;
            color: #9ca3af;
            display: flex;
            justify-content: space-between;
        }
        .main-container { position: relative; }
        #proximity-banner {
            position: absolute;
            top: 28px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 50;
            display: none;
            padding: 10px 24px;
            border-radius: 8px;
            text-align: center;
            font-size: 14px;
            pointer-events: none;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.55);
        }
        #proximity-banner .title { font-size: 20px; font-weight: 800; letter-spacing: 1px; margin-bottom: 2px; }
        #proximity-banner.danger {
            background: rgba(230, 57, 70, 0.94);
            color: #ffffff;
            border: 2px solid #ffb3b8;
            animation: proximity-pulse 0.8s ease-in-out infinite alternate;
        }
        #proximity-banner.warning {
            background: rgba(244, 162, 97, 0.94);
            color: #1b1b1b;
            border: 2px solid #ffd9b3;
        }
        @keyframes proximity-pulse { from { opacity: 1; } to { opacity: 0.55; } }

        /* ---------- Vista 3ª Pessoa: radar (canto superior esquerdo) e mini-cockpit (canto inferior esquerdo) ----------
           Os painéis são <aside> (e não <div>) para não herdarem a regra '.view-panel > div' que estica os gráficos.
           Em ecrãs baixos o JS reduz os dois painéis à mesma escala (fitThirdPersonOverlays) para não se sobreporem. */
        #panel-pov { position: relative; }
        .tp-overlay {
            position: absolute;
            left: 16px;
            z-index: 40;
            background: rgba(4, 10, 18, 0.92);
            border: 1px solid rgba(72, 202, 228, 0.45);
            border-radius: 10px;
            box-shadow: 0 6px 22px rgba(0, 0, 0, 0.6);
        }
        #tp-radar { top: 16px; padding: 10px; user-select: none; transform-origin: top left; }
        #tp-radar canvas { display: block; width: 300px; height: 300px; }
        #tp-radar .radar-bar {
            display: flex; align-items: center; justify-content: space-between;
            margin-top: 8px; font-size: 13px; color: #cbd5e1;
        }
        #tp-radar .radar-bar button { padding: 2px 11px; font-size: 15px; line-height: 18px; }
        #tp-radar .radar-bar b { color: #5eead4; font-family: Consolas, Menlo, monospace; font-size: 14px; }
        #tp-cockpit { bottom: 16px; width: 452px; padding: 12px; transform-origin: bottom left; }
        #tp-cockpit .tpk-head { display: flex; align-items: center; gap: 8px; font-size: 16px; font-weight: 700; margin-bottom: 10px; }
        #tp-cockpit .tpk-dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
        #tp-cockpit .tpk-time { margin-left: auto; font-family: Consolas, Menlo, monospace; font-size: 12px; color: #48cae4; font-weight: 400; }
        #tp-cockpit .status { font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; letter-spacing: 0.5px; }
        #tp-cockpit .status.on { background: rgba(42, 157, 143, 0.25); color: #52d1b5; }
        #tp-cockpit .status.off { background: rgba(156, 163, 175, 0.2); color: #9ca3af; }
        #tp-cockpit .tpk-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        #tp-cockpit .tpk-cell {
            background: rgba(27, 38, 59, 0.65); border: 1px solid #243247; border-radius: 8px;
            padding: 6px 4px 4px; text-align: center;
        }
        #tp-cockpit .tpk-cell canvas { display: block; width: 200px; height: 112px; margin: 2px auto 0; }
        #tp-cockpit .tpk-title { font-size: 11.5px; letter-spacing: 0.6px; color: #b6c2d4; text-transform: uppercase; font-weight: 700; }
        #tp-hint {
            position: absolute; right: 16px; bottom: 16px; z-index: 40;
            font-size: 11px; color: #9ca3af; background: rgba(8, 14, 24, 0.7);
            padding: 6px 10px; border-radius: 6px; pointer-events: none;
        }
        kbd {
            background: #374151;
            padding: 2px 6px;
            border-radius: 4px;
            color: #f9fafb;
            font-size: 11px;
            font-family: monospace;
        }
        /* ---- Cockpit novo (SVG) ---- */
        #cockpit-root { overflow-y: auto; padding: 12px; display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); gap: 12px; align-content: start; background: #0a1220; }
        .ck-c3 { grid-column: span 3; } .ck-c12 { grid-column: span 12; }
        @media (max-width: 1500px) { .ck-c3 { grid-column: span 6; } }
        @media (max-width: 760px) { .ck-c3 { grid-column: span 12; } }
        .ck-head { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; align-items: baseline; color: #9db0cc; font-size: 13px; }
        .ck-head b { color: #e6edf7; font-size: 15px; letter-spacing: 1px; text-transform: uppercase; }
        .ck-head span b { font-size: 13px; letter-spacing: 0; text-transform: none; font-family: monospace; }
        .ck-pn { background: #0f1a2e; border: 1px solid #1d2f4d; border-radius: 8px; padding: 10px 12px; min-width: 0; }
        .ck-pn h4 { margin: 0 0 6px; font-size: 12px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: #9db0cc; }
        .ck-pn svg { display: block; width: 100%; height: auto; max-height: 270px; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
        .ck-ro { display: flex; flex-wrap: wrap; gap: 4px 16px; font-family: monospace; font-size: 12px; color: #9db0cc; margin-top: 6px; align-items: center; }
        .ck-ro b { color: #e6edf7; font-weight: 600; }
        .ck-att3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 8px; }
        .ck-at { background: #0a1220; border: 1px solid #1d2f4d; border-radius: 6px; padding: 7px 8px; min-width: 0; }
        .ck-at .k { font-size: 10.5px; letter-spacing: 0.8px; text-transform: uppercase; color: #9db0cc; }
        .ck-at .v { font-family: monospace; font-size: 20px; font-weight: 700; line-height: 1.2; }
        .ck-at .d { font-size: 11px; color: #8da2c0; line-height: 1.35; margin-top: 2px; }
        .ck-btn { padding: 2px 8px; font-size: 11px; }
        .ck-alert { border-radius: 6px; padding: 7px 10px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; font-size: 14px; margin-bottom: 8px; display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
        .ck-alert.safe { background: #123a2b; color: #3fbf86; } .ck-alert.warn { background: #463408; color: #e8a838; }
        .ck-alert.crit { background: #4a1219; color: #ff5a67; } .ck-alert.none { background: #16233b; color: #9db0cc; }
        .ck-nb svg { max-height: 340px; }
        .ck-nb { display: grid; grid-template-columns: minmax(0, 340px) minmax(0, 1fr); gap: 12px; align-items: center; }
        @media (max-width: 760px) { .ck-nb { grid-template-columns: 1fr; } }
        .ck-tw { overflow-x: auto; }
        .ck-tw table { border-collapse: collapse; width: 100%; font-size: 12.5px; color: #e6edf7; }
        .ck-tw th { color: #6d83a6; text-align: left; font-size: 10.5px; letter-spacing: 1px; text-transform: uppercase; padding: 5px 8px; border-bottom: 1px solid #1d2f4d; white-space: nowrap; }
        .ck-tw td { padding: 6px 8px; border-bottom: 1px solid #1d2f4d; white-space: nowrap; }
        .ck-tw .n { text-align: right; font-family: monospace; }
        /* ---- Separador 6: Vista 1ª Pessoa, painel de bordo de submarino ---- */
        #panel-fp { position: relative; }
        #fp-hud { position: absolute; top: 0; left: 0; right: 0; bottom: 0; pointer-events: none; z-index: 20; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
        #fp-hud::before { content: ""; position: absolute; top: 0; left: 0; right: 0; bottom: 0; border-radius: 8px;
            box-shadow: inset 0 0 0 3px #0e2a47, inset 0 0 120px rgba(8,60,90,0.55), inset 0 0 0 9px rgba(39,195,221,0.18);
            background: radial-gradient(ellipse at center, rgba(0,0,0,0) 62%, rgba(2,12,24,0.55) 100%); }
        #fp-hud.off::before { display: none; }
        #hud-dyn { position: absolute; top: 0; left: 0; right: 0; bottom: 0; }
        .hud-tape { position: absolute; top: 20px; left: 50%; transform: translateX(-50%); max-width: 96%; }
        .hud-tape svg { display: block; max-width: 100%; height: auto; }
        .hud-box { position: absolute; top: 50%; transform: translateY(-50%); background: rgba(6,18,34,0.62); border: 1px solid #27c3dd66; border-radius: 6px; padding: 8px 12px; min-width: 150px; }
        .hud-l { left: 28px; } .hud-r { right: 28px; }
        .hud-t { color: #27c3dd; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; font-size: 12px; margin-bottom: 4px; }
        .hud-k { color: #9db0cc; font-size: 10.5px; letter-spacing: 0.8px; text-transform: uppercase; margin-top: 5px; }
        .hud-v { color: #e6edf7; font-family: monospace; font-size: 19px; font-weight: 700; }
        .hud-nb { position: absolute; left: 50%; bottom: 62px; transform: translateX(-50%); max-width: 92%; text-align: center; padding: 8px 16px; border-radius: 6px; font-size: 13px; color: #e6edf7; background: rgba(6,18,34,0.7); border: 1px solid #27c3dd66; }
        .hud-nb.warn { background: rgba(70,52,8,0.85); border-color: #e8a838; color: #ffd78a; }
        .hud-nb.crit { background: rgba(120,18,28,0.9); border-color: #ff5a67; color: #fff; animation: proximity-pulse 0.8s ease-in-out infinite alternate; }
        .hud-nb.none { color: #9db0cc; }
        .hud-msg { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; font-size: 20px; font-weight: 700; color: #e6edf7; background: rgba(6,18,34,0.78); border: 1px solid #27c3dd66; padding: 14px 24px; border-radius: 8px; }
        .hud-msg span { font-size: 13px; font-weight: 400; color: #9db0cc; }
        .hud-btns { position: absolute; right: 28px; bottom: 18px; display: flex; gap: 8px; pointer-events: auto; }
        .hud-btns button { background: rgba(6,18,34,0.85); border-color: #27c3dd88; padding: 6px 12px; font-size: 12px; }
        .hud-btns button:hover { background: #12304f; }
        /* ---- Separador 5: encontros e relatório ---- */
        #report-root { overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 12px; background: #0a1220; }
        #report-root > * { flex-shrink: 0; }
        .rp-lv { display: inline-block; padding: 1px 9px; border-radius: 10px; font-size: 11px; font-weight: 700; }
        .rp-lv.crit { background: #4a1219; color: #ff5a67; } .rp-lv.warn { background: #463408; color: #e8a838; } .rp-lv.apr { background: #16233b; color: #9db0cc; }
        tr.now td { background: #16304f; }
        .rp-note { color: #9db0cc; font-size: 12.5px; line-height: 1.55; max-width: 1100px; }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <span class="badge">TOMSA 2026</span>
            <h1 style="font-size: 18px; font-weight: 700;">Multi-Robot Trajectory Viewer & Mission Analyzer</h1>
            <span style="font-size: 12px; color: #9ca3af;">(Telemetria Real INESC TEC Aguçadoura)</span>
        </div>
        <div class="nav-buttons">
            <span style="font-size: 12px; color: #9ca3af; margin-right: 4px;">VISTAS:</span>
            <button id="btn-global" class="active" onclick="setView('global')">1. Vista Global 3D</button>
            <button id="btn-pov" onclick="setView('pov')">2. Vista 3ª Pessoa</button>
            <button id="btn-cockpit" onclick="setView('cockpit')">3. Cockpit / Controlo</button>
            <button id="btn-prox" onclick="setView('prox')">4. Distâncias CPA 2D</button>
            <button id="btn-report" onclick="setView('report')">5. Encontros e relatório</button>
            <button id="btn-fp" onclick="setView('fp')">6. Vista 1ª Pessoa</button>
            
            <span style="font-size: 12px; color: #9ca3af; margin-left: 12px; margin-right: 4px;">ROBÔ:</span>
            <span id="robot-buttons"></span>
        </div>
    </header>

    <div class="controls-bar">
        <button id="btn-play" onclick="togglePlay()">&#9654; Play</button>
        <button id="btn-reset" class="danger" onclick="resetTrajectory()">&#8634; Reiniciar (R)</button>
        <button id="btn-follow" onclick="toggleFollow()" title="A câmara da Vista Global segue o robô selecionado">&#9737; Seguir (F)</button>
        <button id="btn-overview" onclick="overviewCamera()" title="Volta à vista de cima com os 3 robôs">&#9635; Vista geral (G)</button>
        <button id="btn-freeze" class="danger" style="display:none" onclick="toggleFreeze()" title="Depois da colisão os dois robots ficam parados. Carregar outra vez mostra os dados reais.">Robots parados (C)</button>
        <button id="btn-predict" class="active" onclick="togglePredict()" title="Mostra onde cada robô estará daqui a 60 s se mantiver a velocidade">&#10140; Previsão 60 s (P)</button>
        
        <div class="slider-box">
            <span style="font-size: 12px; color: #9ca3af;">Tempo:</span>
            <div class="slider-track-wrap">
                <input type="range" id="time-slider" min="__MIN_T__" max="__MAX_T__" step="0.5" value="__T_EVAL__" oninput="onSliderChange(this.value)">
                <div id="prox-marks"></div>
            </div>
            <span id="time-display" class="time-label">00:00 s</span>
        </div>

        <div class="speed-buttons">
            <span style="font-size: 12px; color: #9ca3af;">VELOCIDADE:</span>
            <button id="spd-slow" onclick="setSpeed(0.25)">0.25x Lento</button>
            <button id="spd-norm" class="active" onclick="setSpeed(1.0)">1x Normal</button>
            <button id="spd-fast" onclick="setSpeed(5.0)">5x</button>
            <button id="spd-10" onclick="setSpeed(10.0)">10x</button>
            <button id="spd-20" onclick="setSpeed(20.0)">20x</button>
            <button id="spd-50" onclick="setSpeed(50.0)">50x</button>
            <button id="spd-ultra" style="background:#5a189a;border-color:#7b2cbf;" onclick="setSpeed(1000.0)">&#9889; Ultra (~2s)</button>
        </div>
    </div>

    <div class="main-container">
        <div id="proximity-banner"></div>
        <div id="panel-global" class="view-panel active">
            __DIV_GLOBAL__
        </div>
        <div id="panel-pov" class="view-panel">
            __DIV_POV__
            <aside id="tp-radar" class="tp-overlay">
                <canvas id="tp-radar-canvas" width="600" height="600"></canvas>
                <section class="radar-bar">
                    <button id="tp-range-down" title="Diminuir o alcance">&minus;</button>
                    <span>Alcance <b id="tp-radar-range">100 m</b></span>
                    <button id="tp-range-up" title="Aumentar o alcance">+</button>
                    <span>Rumo <b id="tp-radar-hdg">-</b> &#9650;</span>
                </section>
            </aside>
            <aside id="tp-cockpit" class="tp-overlay">
                <section class="tpk-head">
                    <span id="tpk-dot" class="tpk-dot"></span>
                    <span id="tpk-name">Robot</span>
                    <span id="tpk-status" class="status on">ATIVO</span>
                    <span id="tpk-time" class="tpk-time">-</span>
                </section>
                <section class="tpk-grid">
                    <section class="tpk-cell"><section class="tpk-title">Velocidade</section><canvas id="tpk-spd" width="600" height="336"></canvas></section>
                    <section class="tpk-cell"><section class="tpk-title">Rotação</section><canvas id="tpk-turn" width="600" height="336"></canvas></section>
                    <section class="tpk-cell"><section class="tpk-title">Profundidade</section><canvas id="tpk-depth" width="600" height="336"></canvas></section>
                    <section class="tpk-cell"><section class="tpk-title">Inclinação</section><canvas id="tpk-horizon" width="600" height="336"></canvas></section>
                </section>
            </aside>
            <aside id="tp-hint">Arrastar: rodar à volta do robô &middot; Roda do rato / <kbd>+</kbd> <kbd>-</kbd>: distância &middot; Duplo clique: repor</aside>
        </div>
        <div id="panel-cockpit" class="view-panel">
            __DIV_COCKPIT__
        </div>
        <div id="panel-prox" class="view-panel">
            __DIV_PROX__
        </div>
        <div id="panel-report" class="view-panel">
            __DIV_REPORT__
        </div>
        <div id="panel-fp" class="view-panel">
            __DIV_FP__
            <div id="fp-hud">
                <div id="hud-dyn"></div>
                <div class="hud-btns">
                    <button onclick="firstPersonNearest()" title="Passa a ver o mar pelos olhos do robô mais próximo">Ver pelo vizinho mais próximo</button>
                    <button id="btn-hud" onclick="toggleHud()">Painel de bordo: sim (H)</button>
                </div>
            </div>
        </div>
    </div>

    <div class="help-banner">
        <div>
            <b>Atalhos de Teclado:</b>
            <kbd>1</kbd>, <kbd>2</kbd>, <kbd>3</kbd>: Selecionar Submarino |
            <kbd>V</kbd>: Alternar Vista |
            <kbd>R</kbd>: Reiniciar Trajetória |
            <kbd>F</kbd>: Seguir Robô |
            <kbd>G</kbd>: Vista Geral |
            <kbd>+</kbd>/<kbd>-</kbd>: Zoom 3ª Pessoa |
            <kbd>Espaço</kbd>: Play / Pausa
        </div>
        <div>
            Eixos do corpo: <b style="color:#ff3b3b">x proa</b> | <b style="color:#2ee66b">y estibordo</b> | <b style="color:#3b8bff">z quilha</b>
            &nbsp;|&nbsp; Limiar de Alarme de Colisão: <b>&lt; __DANGER_TXT__ metros</b> | Filtro: <b>RTS Smoother Ativo</b> | Origem: <b>__ORIGIN_TXT__</b>
        </div>
    </div>

    <script>
        // =====================================================================================================
        // MAPA DESTE FICHEIRO (JavaScript que corre no navegador)
        // De onde vêm os dados: o Python faz as contas (ler os CSV, passar WGS-84 para NED, filtro de Kalman com
        // suavizador RTS, ângulos, velocidades, relatório) e entrega ao navegador um único objeto, o timeline, numa
        // grelha de tempo fixa de dt_step segundos. O JavaScript não recalcula a trajetória: interpola entre
        // amostras e desenha.
        // Convenções: N = Norte, E = Este, D = para baixo (profundidade positiva); ângulos em graus; tempos em
        // segundos; f = índice fracionário na grelha (f = 12.5 fica a meio caminho entre as amostras 12 e 13).
        // Blocos, por ordem: interpolação e geometria, cena 3D (Global e 3ª Pessoa), câmaras, radar e mini-cockpit,
        // Cockpit (separador 3), previsão, relatório (separador 5), distâncias (separador 4), aviso e marcas de
        // proximidade, controlos (botões, teclado, reprodução).
        // =====================================================================================================
        const timeline = __TIMELINE_JSON__;
        const dt_step = timeline.dt_step;
        const t0_base = timeline.t0_base;
        const t_max = timeline.t_max;
        const numSteps = timeline.num_steps;
        const robotKeys = Object.keys(timeline.robots).sort();
        // Fator para passar graus a radianos.
        const DEG = Math.PI / 180.0;

        let currentView = 'global';
        let currentRobot = '__ACTIVE_ROBOT__';
        if (!timeline.robots[currentRobot]) currentRobot = robotKeys[0];

        // Obtemos: os botões dos robôs saem dos dados carregados, por isso servem quaisquer nomes e qualquer número de robots.
        (function buildRobotButtons() {
            const holder = document.getElementById('robot-buttons');
            robotKeys.forEach((id, i) => {
                const b = document.createElement('button');
                b.id = 'btn-robot-' + i;
                b.textContent = id;
                b.title = 'Tecla ' + (i + 1);
                if (id === currentRobot) b.classList.add('active');
                b.onclick = () => setRobot(id);
                holder.appendChild(b);
            });
        })();
        let isPlaying = false;
        let speed = 1.0;
        let currentTime = t0_base;
        let animFrameId = null;
        let lastFrameTimestamp = null;
        let lastRender = 0;

        // ---------- utilitários de interpolação ----------
        // Método: converter o tempo atual (segundos) num índice fracionário da grelha, dividindo por dt_step.
        // Obtemos: o f que todas as outras funções usam para ler os dados do instante atual.
        function curF() {
            return Math.min(numSteps - 1, Math.max(0, (currentTime - t0_base) / dt_step));
        }
        // Método: interpolação linear entre as duas amostras vizinhas (peso w = parte fracionária de f).
        // Obtemos: o valor de qualquer série (E, N, D, yaw, velocidade...) num instante entre amostras.
        function valueAt(arr, f) {
            const i = Math.floor(f);
            if (i >= arr.length - 1) return arr[arr.length - 1];
            const w = f - i;
            return arr[i] * (1 - w) + arr[i + 1] * w;
        }
        // Obtemos: se o robô tem dados neste instante (os três robôs não começam nem acabam todos ao mesmo tempo).
        function isActive(r, f) { return f >= r.k0 && f <= r.k1; }
        // Método: interpolar com valueAt() cada série do robô.
        // Obtemos: posição NED, atitude (yaw, pitch, roll), velocidade estimada e taxa de viragem no instante f.
        function stateAt(r, f) {
            return {
                E: valueAt(r.E, f), N: valueAt(r.N, f), D: valueAt(r.D, f),
                yaw: valueAt(r.yaw, f), pitch: valueAt(r.pitch, f), roll: valueAt(r.roll, f),
                spd: valueAt(r.spd, f), turn: valueAt(r.turn, f)
            };
        }
        // Vetor no referencial do corpo (x frente, y direita, z baixo): R^T (ZYX) * (dN, dE, dD)
        // Método: rodar um vetor do referencial NED para o referencial do corpo, com a transposta da matriz de
        // rotação ZYX (yaw, pitch, roll). Entra o vetor (dN, dE, dD) de um robô ao outro.
        // Obtemos: onde esse robô fica visto do nariz do primeiro (x frente, y direita, z baixo).
        function toBody(s, dN, dE, dD) {
            const psi = s.yaw * DEG, the = s.pitch * DEG, phi = s.roll * DEG;
            const cPsi = Math.cos(psi), sPsi = Math.sin(psi);
            const cThe = Math.cos(the), sThe = Math.sin(the);
            const cPhi = Math.cos(phi), sPhi = Math.sin(phi);
            return {
                x: cThe * cPsi * dN + cThe * sPsi * dE - sThe * dD,
                y: (sPhi * sThe * cPsi - cPhi * sPsi) * dN + (sPhi * sThe * sPsi + cPhi * cPsi) * dE + sPhi * cThe * dD,
                z: (cPhi * sThe * cPsi + sPhi * sPsi) * dN + (cPhi * sThe * sPsi - sPhi * cPsi) * dE + cPhi * cThe * dD
            };
        }
        // Obtemos: segundos formatados como minutos e segundos (por exemplo 62m 9.0s).
        function fmtElapsed(sec) {
            const m = Math.floor(sec / 60);
            return m + 'm ' + (sec - 60 * m).toFixed(1) + 's';
        }

        // Eixos do corpo (x proa, y estibordo, z quilha) escritos no mundo NED = colunas de R = Rz(yaw) Ry(pitch) Rx(roll).
        // Devolve 3 vetores [N, E, D]; é a mesma matriz que toBody() usa transposta.
        // Método: construir a matriz de rotação R = Rz(yaw) Ry(pitch) Rx(roll) e ler as suas colunas.
        // Obtemos: os eixos x (proa), y (estibordo) e z (quilha) do robô escritos no mundo NED, que servem para
        // desenhar as setas coloridas e para orientar a malha 3D do submarino.
        function bodyAxesNED(s) {
            const psi = s.yaw * DEG, the = s.pitch * DEG, phi = s.roll * DEG;
            const cPsi = Math.cos(psi), sPsi = Math.sin(psi);
            const cThe = Math.cos(the), sThe = Math.sin(the);
            const cPhi = Math.cos(phi), sPhi = Math.sin(phi);
            return [
                [cThe * cPsi, cThe * sPsi, -sThe],
                [sPhi * sThe * cPsi - cPhi * sPsi, sPhi * sThe * sPsi + cPhi * cPsi, sPhi * cThe],
                [cPhi * sThe * cPsi + sPhi * sPsi, cPhi * sThe * sPsi - sPhi * cPsi, cPhi * cThe]
            ];
        }

        // ---------- Modelo 3D do submarino ----------
        // A malha (vértices no referencial do corpo) vem de Python. Aqui roda-se com roll, pitch e yaw e põe-se na posição do robot.
        // Método: cada vértice da malha (guardada no referencial do corpo) é rodado pelas colunas de R e somado à
        // posição do robô; o fator scale só aumenta ou diminui o casco.
        // Obtemos: os vértices x, y, z do submarino no mundo, com a orientação real do robô naquele instante.
        function subMeshAt(s, scale) {
            const M = timeline.sub_mesh, ax = bodyAxesNED(s), n = M.x.length;
            const X = new Array(n), Y = new Array(n), Z = new Array(n);
            for (let q = 0; q < n; q++) {
                const px = M.x[q] * scale, py = M.y[q] * scale, pz = M.z[q] * scale;
                X[q] = s.E + px * ax[0][1] + py * ax[1][1] + pz * ax[2][1];
                Y[q] = s.N + px * ax[0][0] + py * ax[1][0] + pz * ax[2][0];
                Z[q] = s.D + px * ax[0][2] + py * ax[1][2] + pz * ax[2][2];
            }
            return { x: X, y: Y, z: Z };
        }
        // Método: as letras (guardadas no referencial do corpo, com null a separar os traços) rodam e deslocam-se
        // exatamente como os vértices do casco; os null passam sem conta.
        // Obtemos: o texto INESC TEC colado ao casco, a acompanhar a atitude do robô.
        function subLettersAt(s, scale) {
            const M = timeline.sub_text, ax = bodyAxesNED(s), n = M.x.length;
            const X = new Array(n), Y = new Array(n), Z = new Array(n);
            for (let q = 0; q < n; q++) {
                if (M.x[q] === null) { X[q] = null; Y[q] = null; Z[q] = null; continue; }
                const px = M.x[q] * scale, py = M.y[q] * scale, pz = M.z[q] * scale;
                X[q] = s.E + px * ax[0][1] + py * ax[1][1] + pz * ax[2][1];
                Y[q] = s.N + px * ax[0][0] + py * ax[1][0] + pz * ax[2][0];
                Z[q] = s.D + px * ax[0][2] + py * ax[1][2] + pz * ax[2][2];
            }
            return { x: X, y: Y, z: Z };
        }
        // Cor do casco: a do robot, mais clara no robot selecionado
        // Obtemos: a cor do casco (a do robô; o robô selecionado fica 12 por cento mais claro).
        function hullColor(id) {
            const c = timeline.robots[id].color.replace('#', '');
            const f = (id === currentRobot) ? 0.12 : 0.0;
            const v = [0, 2, 4].map(function (k) { return Math.round(parseInt(c.substring(k, k + 2), 16) * (1 - f) + 255 * f); });
            return 'rgb(' + v[0] + ',' + v[1] + ',' + v[2] + ')';
        }

        // ---------- Cena 3D (Vista Global e Vista 3ª Pessoa partilham o mesmo código) ----------
        // Opacidade de um robô sem dados neste instante (ainda não começou ou já acabou). 1.0 = igual aos outros.
        const INACTIVE_OPACITY = 0.45;
        const lastKeys = {};
        let frameCount = 0;
        // Método: reescrever com Plotly.restyle só as coordenadas dos traços que mudam (mais leve do que refazer o
        // gráfico). Ordem dos traços: 0 superfície, 1 fundo, 2+3i rasto, 3+3i marcador, 4+3i casco,
        // 2+3N+i rota completa, 2+4N+3i+a setas dos eixos, 2+7N+i previsão, 2+8N+i letras (i = robô, N = nº de robôs).
        // Obtemos: a cena 3D no instante f, igual para a Vista Global e para a 3ª Pessoa (thirdPerson = true
        // põe os cascos 1.5 vezes maiores e liga a câmara de 3ª pessoa).
        function updateScene(gdId, f, thirdPerson, firstPerson) {
            const gd = document.getElementById(gdId);
            if (!gd || !gd.data) return;
            frameCount++;
            const hull = firstPerson ? 3.0 : (thirdPerson ? 1.5 : 1.0);           // na 3ª pessoa os cascos são desenhados um pouco maiores
            const traceIdx = [], X = [], Y = [], Z = [];
            const i0 = Math.floor(f);
            const activeFlags = [];
            const markerTraceIdx = [], markerOpacity = [], markerText = [], fadeOpacity = [], fadeTraceIdx = [];
            const axisBase = 2 + 4 * robotKeys.length;      // as setas dos eixos vêm depois das rotas completas
            const lettersBase = 2 + 8 * robotKeys.length;   // as letras INESC TEC vêm depois da previsão
            const predBase = 2 + 7 * robotKeys.length;      // as linhas de previsão vêm depois das setas

            robotKeys.forEach((id, i) => {
                const r = timeline.robots[id];
                const s = stateAt(r, f);
                const act = isActive(r, f);
                const own = thirdPerson && id === currentRobot;
                const hideSelf = firstPerson && id === currentRobot;   // por dentro, o próprio casco não se vê    // o robô seguido pela câmara da 3ª pessoa
                activeFlags.push(act ? 1 : 0);

                // Rasto progressivo: só a parte já percorrida pelo robô
                const sIdx = Math.max(0, Math.floor(r.k0));
                const eIdx = Math.max(sIdx, Math.min(i0, Math.ceil(r.k1)));
                const tx = r.E.slice(sIdx, eIdx + 1), ty = r.N.slice(sIdx, eIdx + 1), tz = r.D.slice(sIdx, eIdx + 1);
                if (act) { tx.push(s.E); ty.push(s.N); tz.push(s.D); }
                traceIdx.push(2 + 3 * i); X.push(tx); Y.push(ty); Z.push(tz);

                traceIdx.push(3 + 3 * i); X.push([s.E]); Y.push([s.N]); Z.push([s.D - 2.2 * hull]);
                const mesh = subMeshAt(s, hull);
                traceIdx.push(4 + 3 * i); X.push(mesh.x); Y.push(mesh.y); Z.push(mesh.z);

                // Eixos x, y, z colados ao robô: rodam com roll, pitch e yaw
                const axes = bodyAxesNED(s), L = timeline.axis_length_m;
                for (let a = 0; a < 3; a++) {
                    traceIdx.push(axisBase + 3 * i + a);
                    X.push([s.E, s.E + L * axes[a][1]]);
                    Y.push([s.N, s.N + L * axes[a][0]]);
                    Z.push([s.D, s.D + L * axes[a][2]]);
                    fadeTraceIdx.push(axisBase + 3 * i + a); fadeOpacity.push((hideSelf ? 0.0 : (act ? 1.0 : INACTIVE_OPACITY)));
                }

                // Letras INESC TEC coladas ao casco
                const lt = subLettersAt(s, hull);
                traceIdx.push(lettersBase + i); X.push(lt.x); Y.push(lt.y); Z.push(lt.z);
                fadeTraceIdx.push(lettersBase + i); fadeOpacity.push(hideSelf ? 0.0 : (act ? 1.0 : INACTIVE_OPACITY));

                // Previsão em linha reta com a velocidade estimada, durante PRED_H segundos (pára à superfície)
                traceIdx.push(predBase + i);
                if (act && predictOn) {
                    const pvN = valueAt(r.vN, f), pvE = valueAt(r.vE, f), pvD = valueAt(r.vD, f);
                    let horizonS = PRED_H;
                    if (pvD < 0 && s.D + horizonS * pvD < 0) horizonS = Math.max(0, -s.D / pvD);
                    X.push([s.E, s.E + horizonS * pvE]); Y.push([s.N, s.N + horizonS * pvN]); Z.push([s.D, s.D + horizonS * pvD]);
                } else { X.push([]); Y.push([]); Z.push([]); }

                markerTraceIdx.push(3 + 3 * i); markerOpacity.push((hideSelf ? 0.0 : (act ? 1.0 : INACTIVE_OPACITY)));
                // o robô seguido já tem o painel de estatísticas: não repete a etiqueta na cena
                markerText.push([own ? '' : (act ? ('<b>' + id + '</b><br>' + s.spd.toFixed(1) + ' m/s') : ('<b>' + id + '</b><br>inativo'))]);
                fadeTraceIdx.push(4 + 3 * i); fadeOpacity.push((hideSelf ? 0.0 : (act ? 1.0 : INACTIVE_OPACITY)));
            });
            Plotly.restyle(gd, { x: X, y: Y, z: Z }, traceIdx);

            const key = activeFlags.join('') + ((thirdPerson || firstPerson) ? currentRobot : '');
            if (key !== lastKeys[gdId] || (frameCount % 8) === 0) {
                lastKeys[gdId] = key;
                Plotly.restyle(gd, { 'marker.opacity': markerOpacity, text: markerText }, markerTraceIdx);
                Plotly.restyle(gd, { opacity: fadeOpacity }, fadeTraceIdx);
            }

            if (thirdPerson) setThirdPersonCamera(gd, f);
            else if (firstPerson) setFirstPersonCamera(gd, f);
            else if (followOn) followCamera(gd, f);
        }

        // Câmara que segue o robô ativo na Vista Global.
        // O centro da câmara fica no robô (a cada instante) e o olho mantém o afastamento (eye - center) que o
        // utilizador escolheu com o rato (rodar e zoom). Coordenadas da câmara = (E - cE, N - cN, -(D - cD)) / span.
        const FOLLOW_DIST_M = 45.0;                  // afastamento inicial ao robô [m]
        const FOLLOW_DIRECTION = [-0.62, -0.62, 0.48];     // direção do olho: sudoeste e acima (E, N, para cima)
        let followOn = false;
        let followCamSet = false;                    // já aplicámos a câmara de seguimento nesta sessão de seguimento?
        let camOffset = null;                        // eye - center, em unidades da câmara do Plotly

        // Obtemos: o afastamento inicial entre olho e robô na câmara de seguimento, em unidades do Plotly.
        function defaultFollowOffset() {
            const k = FOLLOW_DIST_M / timeline.scene.span;
            return { x: FOLLOW_DIRECTION[0] * k, y: FOLLOW_DIRECTION[1] * k, z: FOLLOW_DIRECTION[2] * k };
        }

        // Método: pôr o centro da câmara no robô a cada instante e manter o afastamento (olho menos centro) que o
        // utilizador escolheu com o rato; o afastamento é relido da câmara real antes de cada atualização.
        // Obtemos: uma câmara que segue o robô sem perder o zoom e a rotação que o utilizador deu.
        function followCamera(gd, f) {
            const r = timeline.robots[currentRobot];
            if (!r) return;
            const s = stateAt(r, f), G = timeline.scene, k = 1.0 / G.span;
            if (camOffset === null) camOffset = defaultFollowOffset();
            if (followCamSet) {
                // Se o utilizador rodou ou fez zoom desde a última imagem, guarda o novo afastamento
                try {
                    const live = gd._fullLayout.scene._scene.getCamera();
                    camOffset = { x: live.eye.x - live.center.x, y: live.eye.y - live.center.y, z: live.eye.z - live.center.z };
                } catch (err) { /* sem acesso à câmara atual: mantém o último afastamento */ }
            }
            const center = { x: (s.E - G.cE) * k, y: (s.N - G.cN) * k, z: -(s.D - G.cD) * k };
            const eye = { x: center.x + camOffset.x, y: center.y + camOffset.y, z: center.z + camOffset.z };
            Plotly.relayout(gd, { 'scene.camera': { eye: eye, center: center, up: { x: 0, y: 0, z: 1 }, projection: { type: 'perspective' } } });
            followCamSet = true;
        }

        // Obtemos: liga ou desliga o modo seguir e acerta o aspeto do botão.
        function setFollow(on) {
            followOn = on;
            followCamSet = false;
            document.getElementById('btn-follow').classList.toggle('active', on);
        }

        // Obtemos: o atalho F (alterna seguir e vista geral).
        function toggleFollow() {
            setFollow(!followOn);
            updateFrame(curF());
        }

        // Volta à vista de cima com os 3 robôs (câmara por omissão do Plotly)
        // Obtemos: o atalho G (volta à vista de cima com os três robôs).
        function overviewCamera() {
            setFollow(false);
            const gd = document.getElementById('plot-global');
            if (gd && gd.data) {
                Plotly.relayout(gd, { 'scene.camera': { eye: { x: 1.25, y: 1.25, z: 1.25 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 }, projection: { type: 'perspective' } } });
            }
        }

        // ---------- Câmara em 1ª pessoa (dentro do submarino) ----------
        // Método: a proa (eixo x do corpo) e o eixo z do corpo escrevem-se no mundo NED com yaw, pitch e roll.
        // O olho fica 4 m à frente do centro (no nariz) e 1 m acima do eixo; olha ao longo da proa, e o "para cima"
        // da câmara é o oposto do eixo z do corpo, por isso a imagem inclina com o rolamento e a arfagem do robô.
        // Obtemos: o que o robô veria da janela da frente, com o horizonte a rodar como a atitude medida.
        function setFirstPersonCamera(gd, f) {
            const act = timeline.robots[currentRobot];
            if (!act) return;
            const s = stateAt(act, f);
            const G = timeline.scene, k = 1.0 / G.span;
            const psi = s.yaw * DEG, the = s.pitch * DEG, phi = s.roll * DEG;
            const cPsi = Math.cos(psi), sPsi = Math.sin(psi), cThe = Math.cos(the), sThe = Math.sin(the);
            const cPhi = Math.cos(phi), sPhi = Math.sin(phi);
            // proa (eixo x do corpo) e eixo z do corpo (para baixo) no mundo NED
            const fN = cThe * cPsi, fE = cThe * sPsi, fD = -sThe;
            const zN = cPhi * sThe * cPsi + sPhi * sPsi, zE = cPhi * sThe * sPsi - sPhi * cPsi, zD = cPhi * cThe;
            const eyeE = s.E + fE * 4.0, eyeN = s.N + fN * 4.0, eyeD = s.D + fD * 4.0 - 1.0;   // nariz, 1 m acima do eixo
            const eye = { x: (eyeE - G.cE) * k, y: (eyeN - G.cN) * k, z: -(eyeD - G.cD) * k };
            const L = 0.1;
            const center = { x: eye.x + fE * L, y: eye.y + fN * L, z: eye.z - fD * L };
            const up = { x: -zE, y: -zN, z: zD };
            Plotly.relayout(gd, { 'scene.camera': { eye: eye, center: center, up: up, projection: { type: 'perspective' } } });
        }

        // ---------- Câmara em 3ª pessoa ----------
        // A câmara fica atrás e acima do robô ativo e olha para um ponto um pouco à frente da proa.
        // Só usa o rumo (yaw), alisado nos últimos ~6 s, para a imagem não tremer com o balanço (roll/pitch)
        // nem rodar bruscamente nas viragens. O utilizador pode rodar à volta do robô (arrastar),
        // mudar a distância (roda do rato, + / -) e repor (duplo clique).
        // Coordenadas da câmara do Plotly = (E - cE, N - cN, -(D - cD)) / span (eixo Z invertido).
        const TP_DEFAULT = { dist: 40.0, elev: 22.0, orbit: 0.0 };   // distância [m], elevação [°], órbita [°]
        const TP_LOOK_AHEAD_M = 6.0;                                   // ponto visado à frente da proa [m]
        const TP_SMOOTH_STEPS = 3;                                     // nº de amostras (de dt_step s) para alisar o rumo
        const tpCam = { dist: TP_DEFAULT.dist, elev: TP_DEFAULT.elev, orbit: TP_DEFAULT.orbit };

        // Método: média do rumo nas últimas amostras (os ângulos já vêm desenrolados, por isso a média é válida).
        // Obtemos: um rumo sem tremor, para a câmara de 3ª pessoa não abanar com o balanço do robô.
        function smoothedYaw(r, f) {
            // média do rumo nas últimas amostras (os ângulos vêm 'unwrapped', por isso a média é válida)
            let sum = 0.0, n = 0;
            for (let j = 0; j <= TP_SMOOTH_STEPS; j++) { sum += valueAt(r.yaw, Math.max(0, f - j)); n++; }
            return sum / n;
        }

        // Método: trigonometria simples. Olho = posição do robô recuada pela proa (distância vezes cosseno
        // da elevação) e subida (distância vezes seno da elevação); alvo = ponto um pouco à frente da proa.
        // Obtemos: a câmara atrás e acima do robô selecionado, já no sistema de coordenadas do Plotly.
        function setThirdPersonCamera(gd, f) {
            const act = timeline.robots[currentRobot];
            if (!act) return;
            const s = stateAt(act, f);
            const G = timeline.scene, k = 1.0 / G.span;
            const psi = (smoothedYaw(act, f) + tpCam.orbit) * DEG;
            const el = tpCam.elev * DEG;
            const back = tpCam.dist * Math.cos(el), up = tpCam.dist * Math.sin(el);
            // olho: atrás do robô (no sentido oposto à proa) e acima (D diminui para cima)
            const eyeN = s.N - back * Math.cos(psi), eyeE = s.E - back * Math.sin(psi), eyeD = s.D - up;
            // alvo: ligeiramente à frente da proa, à profundidade do robô
            const tgN = s.N + TP_LOOK_AHEAD_M * Math.cos(psi), tgE = s.E + TP_LOOK_AHEAD_M * Math.sin(psi), tgD = s.D;
            const toCam = (e, n, d) => ({ x: (e - G.cE) * k, y: (n - G.cN) * k, z: -(d - G.cD) * k });
            Plotly.relayout(gd, { 'scene.camera': {
                eye: toCam(eyeE, eyeN, eyeD), center: toCam(tgE, tgN, tgD),
                up: { x: 0, y: 0, z: 1 }, projection: { type: 'perspective' } } });
        }

        function tpZoom(factor) {
            tpCam.dist = Math.min(250.0, Math.max(10.0, tpCam.dist * factor));
            if (currentView === 'pov') updateFrame(curF());
        }
        function tpReset() {
            Object.assign(tpCam, TP_DEFAULT);
            if (currentView === 'pov') updateFrame(curF());
        }

        // O rato controla a câmara da 3ª pessoa (os eventos são intercetados antes de chegarem ao Plotly,
        // que de outra forma rodaria a cena por conta própria e lutaria com a câmara que segue o robô).
        function installThirdPersonMouse() {
            const panel = document.getElementById('panel-pov');
            if (!panel) return;
            const onPlot = (e) => !e.target.closest('.tp-overlay');
            let drag = null;
            panel.addEventListener('wheel', (e) => {
                if (!onPlot(e)) return;
                e.preventDefault(); e.stopPropagation();
                tpZoom(e.deltaY > 0 ? 1.12 : 1.0 / 1.12);
            }, { capture: true, passive: false });
            panel.addEventListener('mousedown', (e) => {
                if (!onPlot(e) || e.button !== 0) return;
                e.preventDefault(); e.stopPropagation();
                drag = { x: e.clientX, y: e.clientY, orbit: tpCam.orbit, elev: tpCam.elev };
            }, { capture: true });
            window.addEventListener('mousemove', (e) => {
                if (!drag) return;
                e.preventDefault(); e.stopPropagation();
                tpCam.orbit = drag.orbit - (e.clientX - drag.x) * 0.4;
                tpCam.elev = Math.min(85.0, Math.max(3.0, drag.elev + (e.clientY - drag.y) * 0.3));
                updateFrame(curF());
            }, { capture: true });
            window.addEventListener('mouseup', () => { drag = null; }, { capture: true });
            panel.addEventListener('dblclick', (e) => {
                if (!onPlot(e)) return;
                e.preventDefault(); e.stopPropagation(); tpReset();
            }, { capture: true });
        }

        // ---------- Radar (mini-mapa) e mini-cockpit da Vista 3ª Pessoa ----------
        // Desenhados em <canvas> 2D num ciclo próprio (requestAnimationFrame), independente do Plotly:
        // é leve e permite o varrimento animado do radar mesmo com a reprodução em pausa.
        const RADAR_RANGES = [50, 100, 250, 500, 1000];
        let radarRange = RADAR_RANGES.reduce((a, b) => Math.abs(b - timeline.radar_range_m) < Math.abs(a - timeline.radar_range_m) ? b : a);
        const RADAR_TRAIL_STEPS = 60;        // rasto no radar: últimas 60 amostras (2 min com dt = 2 s)
        let radarLoopId = null, lastStatsUpdate = 0;

        // Obtemos: passa ao alcance seguinte ou anterior do radar (50, 100, 250, 500 ou 1000 m).
        function changeRadarRange(step) {
            const i = Math.min(RADAR_RANGES.length - 1, Math.max(0, RADAR_RANGES.indexOf(radarRange) + step));
            radarRange = RADAR_RANGES[i];
            document.getElementById('tp-radar-range').innerText = radarRange + ' m';
        }

        // Posição relativa no plano horizontal com a proa para cima: frente = (dN, dE) projetado na proa
        // Método: rodar o vetor (dN, dE) pelo rumo do robô, com um produto escalar para cada eixo.
        // Obtemos: quanto o outro robô está à frente (fwd) e à direita (right), para o radar ter a proa para cima.
        function relHeadingUp(psiRad, dN, dE) {
            return { fwd: Math.cos(psiRad) * dN + Math.sin(psiRad) * dE, right: -Math.sin(psiRad) * dN + Math.cos(psiRad) * dE };
        }

        function roundRect(ctx, x, y, w, h, r) {
            ctx.beginPath();
            ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
            ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
            ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
            ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
        }

        // Método: converter cada robô para (frente, direita) com relHeadingUp, multiplicar pelos píxeis por metro
        // (raio útil dividido pelo alcance) e desenhar no canvas 2D.
        // Obtemos: um radar centrado no robô selecionado, com a proa para cima, rasto, Δ de profundidade e distância.
        function drawRadar(f, nowMs) {
            const cv = document.getElementById('tp-radar-canvas');
            if (!cv) return;
            const ctx = cv.getContext('2d');
            const W = cv.width, u = W / 600, c = W / 2;
            const R = W / 2 - 52 * u;                                // raio útil; a coroa exterior leva a rosa dos rumos
            const me = timeline.robots[currentRobot];
            const s = stateAt(me, f);
            const psi = s.yaw * DEG;
            const px = R / radarRange;                               // píxeis por metro
            const toXY = (dN, dE) => { const p = relHeadingUp(psi, dN, dE); return [c + p.right * px, c - p.fwd * px]; };
            const hdg = ((s.yaw % 360.0) + 360.0) % 360.0;
            setText('tp-radar-hdg', hdg.toFixed(0).padStart(3, '0') + '°');

            ctx.clearRect(0, 0, W, W);

            // coroa exterior: rosa dos rumos (roda com o rumo, porque o radar tem a proa para cima)
            ctx.fillStyle = '#071722';
            ctx.beginPath(); ctx.arc(c, c, W / 2 - 2 * u, 0, 2 * Math.PI); ctx.fill();
            for (let a = 0; a < 360; a += 10) {
                const ang = (a - hdg) * DEG;                         // ângulo no ecrã a partir do topo, sentido horário
                const major = a % 30 === 0;
                const r0 = R + 4 * u, r1 = R + (major ? 16 : 9) * u;
                ctx.strokeStyle = major ? 'rgba(226, 232, 240, 0.85)' : 'rgba(148, 163, 184, 0.55)';
                ctx.lineWidth = (major ? 3 : 2) * u;
                ctx.beginPath();
                ctx.moveTo(c + r0 * Math.sin(ang), c - r0 * Math.cos(ang));
                ctx.lineTo(c + r1 * Math.sin(ang), c - r1 * Math.cos(ang));
                ctx.stroke();
            }
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            [['N', 0], ['E', 90], ['S', 180], ['W', 270]].forEach(([t, a]) => {
                const ang = (a - hdg) * DEG, rr = R + 33 * u;
                ctx.font = 'bold ' + Math.round(30 * u) + 'px Segoe UI, Arial';
                ctx.fillStyle = t === 'N' ? '#ff5d5d' : '#e2e8f0';
                ctx.fillText(t, c + rr * Math.sin(ang), c - rr * Math.cos(ang));
            });
            // linha de fé (marca fixa da proa, no topo)
            ctx.fillStyle = '#facc15';
            ctx.beginPath(); ctx.moveTo(c, c - R - 2 * u); ctx.lineTo(c - 11 * u, c - R - 20 * u); ctx.lineTo(c + 11 * u, c - R - 20 * u); ctx.closePath(); ctx.fill();

            // ecrã do radar
            ctx.fillStyle = '#02121a';
            ctx.beginPath(); ctx.arc(c, c, R, 0, 2 * Math.PI); ctx.fill();
            ctx.save();
            ctx.beginPath(); ctx.arc(c, c, R, 0, 2 * Math.PI); ctx.clip();

            // varrimento animado (discreto, para não esconder os contactos)
            const sweep = ((nowMs / 1000.0) * 1.6) % (2 * Math.PI);
            if (ctx.createConicGradient) {
                const g = ctx.createConicGradient(sweep - Math.PI / 2 - 0.8, c, c);
                const w = 0.8 / (2 * Math.PI);
                g.addColorStop(0.0, 'rgba(52, 211, 153, 0.0)');
                g.addColorStop(w, 'rgba(52, 211, 153, 0.16)');
                g.addColorStop(w + 0.001, 'rgba(52, 211, 153, 0.0)');
                g.addColorStop(1.0, 'rgba(52, 211, 153, 0.0)');
                ctx.fillStyle = g; ctx.fillRect(0, 0, W, W);
            }

            // anéis de distância e eixos
            ctx.strokeStyle = 'rgba(110, 231, 183, 0.5)'; ctx.lineWidth = 2 * u;
            for (let q = 1; q <= 4; q++) { ctx.beginPath(); ctx.arc(c, c, R * q / 4, 0, 2 * Math.PI); ctx.stroke(); }
            ctx.strokeStyle = 'rgba(110, 231, 183, 0.25)'; ctx.lineWidth = 1.5 * u;
            ctx.beginPath(); ctx.moveTo(c - R, c); ctx.lineTo(c + R, c); ctx.moveTo(c, c - R); ctx.lineTo(c, c + R); ctx.stroke();

            // limiares de proximidade (laranja = aviso, vermelho = colisão)
            [[timeline.proximity_warning_m, 'rgba(251, 146, 60, 0.95)'], [timeline.collision_threshold_m, 'rgba(248, 113, 113, 1.0)']].forEach(([m, col]) => {
                if (m * px < 4 * u || m > radarRange) return;
                ctx.setLineDash([9 * u, 7 * u]); ctx.strokeStyle = col; ctx.lineWidth = 2.5 * u;
                ctx.beginPath(); ctx.arc(c, c, m * px, 0, 2 * Math.PI); ctx.stroke();
            });
            ctx.setLineDash([]);

            // rastos recentes (o próprio e os dos outros robôs ativos)
            const i0 = Math.floor(f);
            robotKeys.forEach(id => {
                const r = timeline.robots[id];
                if (!isActive(r, f)) return;
                const a = Math.max(Math.ceil(r.k0), i0 - RADAR_TRAIL_STEPS);
                ctx.strokeStyle = r.color; ctx.globalAlpha = id === currentRobot ? 0.7 : 0.55; ctx.lineWidth = 3 * u;
                ctx.beginPath();
                let first = true;
                for (let i = a; i <= i0; i++) {
                    const [x, y] = toXY(r.N[i] - s.N, r.E[i] - s.E);
                    if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
                }
                const t = stateAt(r, f); const [xe, ye] = toXY(t.N - s.N, t.E - s.E); ctx.lineTo(xe, ye);
                ctx.stroke(); ctx.globalAlpha = 1.0;
            });
            ctx.restore();

            // contorno do ecrã e escala (valores dos anéis ao longo do eixo horizontal, à direita)
            ctx.strokeStyle = 'rgba(94, 234, 212, 0.9)'; ctx.lineWidth = 3 * u;
            ctx.beginPath(); ctx.arc(c, c, R, 0, 2 * Math.PI); ctx.stroke();
            ctx.font = 'bold ' + Math.round(19 * u) + 'px Consolas, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            for (let q = 1; q <= 4; q++) {                       // na diagonal inferior direita, um valor por anel
                const txt = Math.round(radarRange * q / 4) + (q === 4 ? ' m' : '');
                const rq = R * q / 4 - (q === 4 ? 22 : 0) * u;
                const x = c + rq * Math.SQRT1_2, y = c + rq * Math.SQRT1_2;
                const tw = ctx.measureText(txt).width + 8 * u;
                ctx.fillStyle = 'rgba(2, 18, 26, 0.85)'; ctx.fillRect(x - tw / 2, y - 11 * u, tw, 22 * u);
                ctx.fillStyle = '#a7f3d0'; ctx.fillText(txt, x, y);
            }

            // outros robôs: ponto com halo, nome, distância e diferença de profundidade (seta na borda se fora de alcance).
            // As etiquetas ficam do lado de fora do ponto (afastadas do centro) e não se sobrepõem umas às outras.
            const placed = [];
            const LW = 176 * u, LH = 58 * u;                         // tamanho de uma etiqueta [px do canvas]
            robotKeys.forEach(id => {
                if (id === currentRobot) return;
                const r = timeline.robots[id];
                if (!isActive(r, f) || !isActive(me, f)) return;
                const t = stateAt(r, f);
                const dN = t.N - s.N, dE = t.E - s.E, dD = t.D - s.D;
                const dist = Math.sqrt(dN * dN + dE * dE + dD * dD);
                const hor = Math.hypot(dN, dE);
                let [x, y] = toXY(dN, dE);
                const out = hor > radarRange;
                if (out) { const sc = R / (hor * px); x = c + (x - c) * sc; y = c + (y - c) * sc; }
                const alert = dist < timeline.collision_threshold_m ? '#f87171' : (dist < timeline.proximity_warning_m ? '#fb923c' : null);
                const col = alert || r.color;
                if (out) {
                    const ang = Math.atan2(y - c, x - c);
                    ctx.save(); ctx.translate(x, y); ctx.rotate(ang);
                    ctx.fillStyle = col; ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 3 * u;
                    ctx.beginPath(); ctx.moveTo(6 * u, 0); ctx.lineTo(-20 * u, -14 * u); ctx.lineTo(-20 * u, 14 * u); ctx.closePath(); ctx.fill(); ctx.stroke();
                    ctx.restore();
                } else {
                    ctx.fillStyle = col; ctx.globalAlpha = 0.3;
                    ctx.beginPath(); ctx.arc(x, y, 22 * u, 0, 2 * Math.PI); ctx.fill(); ctx.globalAlpha = 1.0;
                    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 3 * u;
                    ctx.beginPath(); ctx.arc(x, y, 12 * u, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
                }
                // posição da etiqueta: para fora (ou para dentro, se o ponto estiver perto da borda)
                const ux = x - c, uy = y - c, un = Math.hypot(ux, uy) || 1;
                const dir = (out || un > R * 0.55) ? -1 : 1;
                let lx = x + dir * (ux / un) * 62 * u, ly = y + dir * (uy / un) * 48 * u;
                lx = Math.min(W - LW / 2 - 4 * u, Math.max(LW / 2 + 4 * u, lx));
                ly = Math.min(W - LH / 2 - 4 * u, Math.max(LH / 2 + 4 * u, ly));
                // se colidir com uma etiqueta já desenhada, desloca-se para cima/baixo dela (sempre no mesmo sentido)
                let shiftDir = 0;
                for (let tries = 0; tries < 6; tries++) {
                    const q = placed.find(p => Math.abs(p[0] - lx) < LW && Math.abs(p[1] - ly) < LH);
                    if (!q) break;
                    if (shiftDir === 0) {
                        shiftDir = ly >= q[1] ? 1 : -1;
                        const ny = q[1] + shiftDir * LH;
                        if (ny > W - LH / 2 || ny < LH / 2) shiftDir = -shiftDir;
                    }
                    ly = q[1] + shiftDir * LH;
                }
                placed.push([lx, ly]);
                ctx.fillStyle = 'rgba(0, 0, 0, 0.82)';
                roundRect(ctx, lx - LW / 2, ly - LH / 2 + 2 * u, LW, LH - 4 * u, 7 * u); ctx.fill();
                ctx.fillStyle = col; ctx.fillRect(lx - LW / 2, ly - LH / 2 + 8 * u, 5 * u, LH - 16 * u);
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.font = 'bold ' + Math.round(24 * u) + 'px Segoe UI, Arial'; ctx.fillStyle = alert || '#ffffff';
                ctx.fillText(id.replace('Robot ', 'R') + '  ' + dist.toFixed(0) + ' m', lx + 3 * u, ly - 11 * u);
                ctx.font = Math.round(19 * u) + 'px Consolas, monospace'; ctx.fillStyle = '#cbd5e1';
                ctx.fillText('Δprof ' + (dD >= 0 ? '+' : '') + dD.toFixed(1) + ' m', lx + 3 * u, ly + 13 * u);
            });

            // o próprio robô no centro (triângulo a apontar para a proa)
            ctx.fillStyle = me.color; ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 3.5 * u;
            ctx.beginPath(); ctx.moveTo(c, c - 26 * u); ctx.lineTo(c + 17 * u, c + 18 * u); ctx.lineTo(c, c + 9 * u); ctx.lineTo(c - 17 * u, c + 18 * u); ctx.closePath();
            ctx.fill(); ctx.stroke();
            ctx.textBaseline = 'alphabetic';
        }

        function setText(id, txt) { const el = document.getElementById(id); if (el && el.textContent !== txt) el.textContent = txt; }

        // ---------- Mini-cockpit: as mesmas grandezas e escalas da página Cockpit ----------
        // Conta-quilómetros de ponteiro (velocidades): arco de 240°, marcas, zonas coloridas e uma seta que aponta o valor
        // Método: mapear o valor numa fração do arco de 240 graus (frac) e daí num ângulo (angOf).
        // Obtemos: um mostrador de ponteiro (velocidade e rotação) com as mesmas escalas do separador Cockpit.
        function drawNeedleGauge(id, o) {
            const cv = document.getElementById(id);
            if (!cv) return;
            const ctx = cv.getContext('2d');
            ctx.setTransform(cv.width / 300, 0, 0, cv.width / 300, 0, 0);   // desenho em coordenadas lógicas 300x168
            const W = 300, H = 168, cx = W / 2, cy = 94, r = 64;
            const A0 = 150 * DEG, SWEEP = 240 * DEG;                   // do canto inferior esquerdo ao inferior direito
            const frac = (v) => (Math.min(o.max, Math.max(o.min, v)) - o.min) / (o.max - o.min);
            const angOf = (v) => A0 + frac(v) * SWEEP;
            const polar = (a, rr) => [cx + rr * Math.cos(a), cy + rr * Math.sin(a)];
            ctx.clearRect(0, 0, W, H);
            ctx.lineCap = 'butt';
            // mostrador
            ctx.fillStyle = '#0d1b2a';
            ctx.beginPath(); ctx.arc(cx, cy, r + 8, 0, 2 * Math.PI); ctx.fill();
            ctx.strokeStyle = '#1b263b'; ctx.lineWidth = 9;
            ctx.beginPath(); ctx.arc(cx, cy, r, A0, A0 + SWEEP); ctx.stroke();
            (o.zones || []).forEach(z => {
                ctx.strokeStyle = z.color; ctx.lineWidth = 9;
                ctx.beginPath(); ctx.arc(cx, cy, r, angOf(z.from), angOf(z.to)); ctx.stroke();
            });
            // marcas pequenas e grandes
            for (let v = o.min; v <= o.max + 1e-9; v += o.minor) {
                const a = angOf(v), major = o.ticks.some(t => Math.abs(t - v) < 1e-6);
                const [x0, y0] = polar(a, r - (major ? 14 : 8)), [x1, y1] = polar(a, r - 3);
                ctx.strokeStyle = major ? '#e2e8f0' : '#64748b'; ctx.lineWidth = major ? 3 : 2;
                ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
            }
            ctx.fillStyle = '#cbd5e1'; ctx.font = '16px Segoe UI, Arial'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            o.ticks.forEach(t => { const [x, y] = polar(angOf(t), r + 20); ctx.fillText(String(t), x, y); });
            // ponteiro em forma de seta
            if (o.value !== null) {
                const a = angOf(o.value), [tx, ty] = polar(a, r - 6);
                const n = [Math.cos(a + Math.PI / 2), Math.sin(a + Math.PI / 2)];
                const [bx, by] = polar(a + Math.PI, 12);                      // cauda do ponteiro, do outro lado do eixo
                const [hx, hy] = polar(a, r - 22);                            // base da ponta da seta
                ctx.fillStyle = o.color; ctx.strokeStyle = '#0b0f19'; ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(bx + n[0] * 3, by + n[1] * 3);
                ctx.lineTo(hx + n[0] * 3, hy + n[1] * 3);
                ctx.lineTo(hx + n[0] * 9, hy + n[1] * 9);
                ctx.lineTo(tx, ty);
                ctx.lineTo(hx - n[0] * 9, hy - n[1] * 9);
                ctx.lineTo(hx - n[0] * 3, hy - n[1] * 3);
                ctx.lineTo(bx - n[0] * 3, by - n[1] * 3);
                ctx.closePath(); ctx.fill(); ctx.stroke();
            }
            ctx.fillStyle = '#e2e8f0'; ctx.beginPath(); ctx.arc(cx, cy, 7, 0, 2 * Math.PI); ctx.fill();
            ctx.fillStyle = '#0b0f19'; ctx.beginPath(); ctx.arc(cx, cy, 3, 0, 2 * Math.PI); ctx.fill();
            // valor numérico por baixo do eixo
            ctx.textBaseline = 'alphabetic';
            ctx.font = 'bold 26px Consolas, Menlo, monospace'; ctx.fillStyle = o.textColor || '#f3f4f6';
            ctx.fillText(o.value === null ? '–' : o.text, cx, cy + 48);
            ctx.font = '16px Segoe UI, Arial'; ctx.fillStyle = '#9ca3af';
            ctx.fillText(o.unit, cx, cy + 68);
        }

        // Profundímetro de coluna: tubo vertical (superfície em cima, fundo em baixo) com a água a encher até à
        // profundidade atual; a barra desce quando o robô mergulha e sobe quando ele vem à superfície.
        // Método: mapear a profundidade na altura de um tubo (yOf) e mostrar a seta da velocidade vertical.
        // Obtemos: a barra de profundidade com o limite de aviso e a tendência (a subir ou a descer).
        function drawDepthBar(id, depth, maxDepth, warnDepth, vz, active) {
            const cv = document.getElementById(id);
            if (!cv) return;
            const ctx = cv.getContext('2d');
            ctx.setTransform(cv.width / 300, 0, 0, cv.width / 300, 0, 0);   // desenho em coordenadas lógicas 300x168
            const W = 300, H = 168;
            const x0 = 96, w = 46, y0 = 12, h = H - 24;                 // tubo
            const yOf = (d) => y0 + Math.min(1, Math.max(0, d / maxDepth)) * h;
            ctx.clearRect(0, 0, W, H);
            // tubo vazio
            ctx.fillStyle = '#0d1b2a'; ctx.fillRect(x0, y0, w, h);
            // coluna de água até à profundidade atual
            if (active) {
                const yd = yOf(depth);
                const g = ctx.createLinearGradient(0, y0, 0, y0 + h);
                g.addColorStop(0, '#48cae4'); g.addColorStop(1, '#023e8a');
                ctx.fillStyle = g; ctx.fillRect(x0, y0, w, yd - y0);
            }
            // zona perigosa perto do fundo
            ctx.fillStyle = 'rgba(230, 57, 70, 0.35)'; ctx.fillRect(x0, yOf(warnDepth), w, y0 + h - yOf(warnDepth));
            ctx.strokeStyle = '#e63946'; ctx.lineWidth = 3;
            ctx.beginPath(); ctx.moveTo(x0 - 4, yOf(warnDepth)); ctx.lineTo(x0 + w + 4, yOf(warnDepth)); ctx.stroke();
            // escala à esquerda
            ctx.fillStyle = '#cbd5e1'; ctx.font = '16px Segoe UI, Arial'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
            for (let d = 0; d <= maxDepth + 1e-9; d += 5) {
                const y = yOf(d), major = d % 10 === 0;
                ctx.strokeStyle = major ? '#e2e8f0' : '#64748b'; ctx.lineWidth = 2;
                ctx.beginPath(); ctx.moveTo(x0 - (major ? 12 : 6), y); ctx.lineTo(x0, y); ctx.stroke();
                if (major) ctx.fillText(String(d), x0 - 16, y);
            }
            ctx.strokeStyle = '#415a77'; ctx.lineWidth = 3; ctx.strokeRect(x0, y0, w, h);
            // nível atual: linha e seta à direita do tubo
            if (active) {
                const yd = yOf(depth);
                ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 3;
                ctx.beginPath(); ctx.moveTo(x0, yd); ctx.lineTo(x0 + w, yd); ctx.stroke();
                ctx.fillStyle = '#ffffff';
                ctx.beginPath(); ctx.moveTo(x0 + w + 3, yd); ctx.lineTo(x0 + w + 15, yd - 8); ctx.lineTo(x0 + w + 15, yd + 8); ctx.closePath(); ctx.fill();
            }
            // valor e tendência à direita
            ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
            const tx = x0 + w + 24;
            ctx.font = 'bold 30px Consolas, Menlo, monospace'; ctx.fillStyle = '#00b4d8';
            ctx.fillText(active ? depth.toFixed(1) : '–', tx, 74);
            ctx.font = '17px Segoe UI, Arial'; ctx.fillStyle = '#9ca3af';
            ctx.fillText('m', tx, 96);
            if (active) {
                const still = Math.abs(vz) < 0.02;
                ctx.font = 'bold 17px Segoe UI, Arial';
                ctx.fillStyle = still ? '#9ca3af' : (vz > 0 ? '#48cae4' : '#a7f3d0');
                ctx.fillText(still ? '= estável' : (vz > 0 ? '▼ desce' : '▲ sobe'), tx, 126);
                if (!still) { ctx.font = '15px Consolas, monospace'; ctx.fillText(Math.abs(vz).toFixed(2) + ' m/s', tx, 146); }
            }
        }

        // Horizonte artificial: céu/mar rodados pelo roll e deslocados pelo pitch, com o símbolo fixo do veículo
        // Método: rodar o desenho por menos o roll e deslocá-lo conforme o pitch.
        // Obtemos: um horizonte artificial como o de um avião, onde se vê a inclinação do robô.
        function drawHorizon(id, roll, pitch, active) {
            const cv = document.getElementById(id);
            if (!cv) return;
            const ctx = cv.getContext('2d');
            ctx.setTransform(cv.width / 300, 0, 0, cv.width / 300, 0, 0);   // desenho em coordenadas lógicas 300x168
            const W = 300, H = 168, r = H / 2 - 8, cx = r + 14, cy = H / 2;
            ctx.clearRect(0, 0, W, H);
            ctx.save();
            ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.clip();
            ctx.translate(cx, cy);
            ctx.rotate(-(active ? roll : 0) * DEG);
            const off = Math.max(-r, Math.min(r, (active ? pitch : 0) * (r / 30.0)));   // 30° de pitch = meio raio
            ctx.fillStyle = '#1e6091'; ctx.fillRect(-2 * r, -2 * r + off, 4 * r, 2 * r);
            ctx.fillStyle = '#6b4226'; ctx.fillRect(-2 * r, off, 4 * r, 2 * r);
            ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 3;
            ctx.beginPath(); ctx.moveTo(-2 * r, off); ctx.lineTo(2 * r, off); ctx.stroke();
            ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(255,255,255,0.7)';
            [-20, -10, 10, 20].forEach(p => { const y = off - p * (r / 30.0), w = (Math.abs(p) === 10 ? 14 : 24);
                ctx.beginPath(); ctx.moveTo(-w, y); ctx.lineTo(w, y); ctx.stroke(); });
            ctx.restore();
            ctx.strokeStyle = '#415a77'; ctx.lineWidth = 4;
            ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.stroke();
            ctx.strokeStyle = '#facc15'; ctx.lineWidth = 5;                                  // símbolo do veículo
            ctx.beginPath(); ctx.moveTo(cx - 34, cy); ctx.lineTo(cx - 10, cy); ctx.lineTo(cx, cy + 9); ctx.lineTo(cx + 10, cy); ctx.lineTo(cx + 34, cy); ctx.stroke();
            const sg = (v) => (v >= 0 ? '+' : '') + v.toFixed(1) + '°';
            const tx = cx + r + 22;
            ctx.textAlign = 'left';
            ctx.font = '18px Segoe UI, Arial'; ctx.fillStyle = '#9ca3af';
            ctx.fillText('Roll', tx, cy - 30); ctx.fillText('Pitch', tx, cy + 26);
            ctx.font = 'bold 28px Consolas, Menlo, monospace'; ctx.fillStyle = '#f3f4f6';
            ctx.fillText(active ? sg(roll) : '–', tx, cy - 4); ctx.fillText(active ? sg(pitch) : '–', tx, cy + 52);
        }

        // Obtemos: atualiza os quatro instrumentos do mini-cockpit da 3ª Pessoa e o cabeçalho do robô.
        function updateCockpitPanel(f) {
            const me = timeline.robots[currentRobot];
            const s = stateAt(me, f);
            const act = isActive(me, f);
            document.getElementById('tpk-dot').style.background = me.color;
            setText('tpk-name', me.label || currentRobot);
            document.getElementById('tpk-status').className = 'status ' + (act ? 'on' : 'off');
            setText('tpk-status', act ? 'ATIVO' : 'INATIVO');
            setText('tpk-time', fmtElapsed(f * dt_step));

            // escalas iguais às da página Cockpit original (velocidade, viragem, profundidade)
            // velocidade vertical (derivada da profundidade, positiva = a descer) para a seta de tendência
            const vz = (valueAt(me.D, Math.min(numSteps - 1, f + 0.5)) - valueAt(me.D, Math.max(0, f - 0.5))) / dt_step;
            drawDepthBar('tpk-depth', s.D, 35, 32, vz, act);
            drawNeedleGauge('tpk-spd', { value: act ? s.spd : null, min: -0.5, max: 3.0, minor: 0.25, ticks: [0, 1, 2, 3],
                zones: [{ from: -0.5, to: 0.0, color: '#9b2226' }],
                color: '#52b788', text: s.spd.toFixed(2), unit: 'm/s', textColor: '#52b788' });
            drawNeedleGauge('tpk-turn', { value: act ? s.turn : null, min: -45, max: 45, minor: 7.5, ticks: [-45, 0, 45],
                zones: [{ from: -45, to: -30, color: '#7c2d12' }, { from: 30, to: 45, color: '#7c2d12' }],
                color: '#f4a261', text: (s.turn >= 0 ? '+' : '') + s.turn.toFixed(1), unit: '°/s', textColor: '#f4a261' });
            drawHorizon('tpk-horizon', s.roll, s.pitch, act);
        }

        // Em ecrãs baixos, reduz o radar e o mini-cockpit à mesma escala para caberem sem se sobreporem
        function fitThirdPersonOverlays() {
            const panel = document.getElementById('panel-pov'), ra = document.getElementById('tp-radar'), ck = document.getElementById('tp-cockpit');
            if (!panel || !ra || !ck || panel.clientHeight === 0) return;
            const margin = 16;                                  // margens (não encolhem com a escala)
            const sc = Math.min(1.0, (panel.clientHeight - 3 * margin) / (ra.offsetHeight + ck.offsetHeight));
            ra.style.transform = ck.style.transform = (sc < 1.0 ? 'scale(' + sc.toFixed(3) + ')' : 'none');
        }
        window.addEventListener('resize', fitThirdPersonOverlays);

        // Método: ciclo com requestAnimationFrame, independente do Plotly.
        // Obtemos: radar fluido (cerca de 60 imagens por segundo) e mini-cockpit a 10 atualizações por segundo.
        function thirdPersonOverlayLoop(nowMs) {
            if (currentView !== 'pov') { radarLoopId = null; return; }
            const f = curF();
            drawRadar(f, nowMs);
            if (nowMs - lastStatsUpdate > 100) { lastStatsUpdate = nowMs; updateCockpitPanel(f); }   // ~10 atualizações/s
            radarLoopId = requestAnimationFrame(thirdPersonOverlayLoop);
        }

        function startThirdPersonOverlay() {
            fitThirdPersonOverlays();
            if (radarLoopId === null) radarLoopId = requestAnimationFrame(thirdPersonOverlayLoop);
        }

        // ---------- Cockpit ----------
        // Painel de instrumentos em SVG, alimentado pela mesma linha temporal das outras vistas. Mostra o que interessa
        // para navegar: atitude e eixos do corpo, rumo e rota, velocidade medida contra estimada, vizinhos no
        // referencial do corpo (azimute, elevação, distância, TCPA, CPA), profundidade e saltos de posição.
        // Cores partilhadas por todos os painéis do Cockpit.
        const COCKPIT_COLORS = { cy: '#27c3dd', ink: '#e6edf7', ink2: '#9db0cc', mut: '#6d83a6', line: '#1d2f4d', red: '#ff5a67', green: '#3fbf86', amber: '#e8a838', blue: '#4ea1ff' };
        // Obtemos: número com casas decimais, traço quando não existe, e sem o sinal de menos em zero (-0.0).
        const formatNumber = (x, d) => { if (x === null || x === undefined || isNaN(x)) return '–'; const t = x.toFixed(d); return (t.charAt(0) === '-' && parseFloat(t) === 0) ? t.substring(1) : t; };
        const fmt1 = (x) => formatNumber(x, 1);
        const fmt2 = (x) => formatNumber(x, 2);
        // Método: resto da divisão por 360 corrigido para positivo.
        // Obtemos: um ângulo entre 0 e 360 (rumo).
        const wrap360 = (a) => ((a % 360) + 360) % 360;
        // Método: deslocar 180, resto da divisão por 360, voltar a deslocar.
        // Obtemos: um ângulo entre -180 e 180 (diferenças de rumo).
        const wrap180 = (a) => (((a + 180) % 360) + 360) % 360 - 180;
        const depthPathCache = {};

        // Estado completo de um robô no instante f (posição, atitude, velocidades estimadas e medidas, incerteza)
        // Método: stateAt() mais as velocidades (estimada NED, medida no corpo) e a incerteza do filtro.
        // Obtemos: tudo o que o Cockpit precisa de um robô num instante.
        function fullStateAt(r, f) {
            const s = stateAt(r, f);
            s.vN = valueAt(r.vN, f); s.vE = valueAt(r.vE, f); s.vD = valueAt(r.vD, f);
            s.u = valueAt(r.u, f); s.v = valueAt(r.v, f); s.w = valueAt(r.w, f); s.sig = valueAt(r.sig, f);
            return s;
        }
        // Matriz de rotação corpo para NED (colunas = eixos x, y, z do corpo em NED, a mesma de bodyAxesNED)
        // Obtemos: a matriz de rotação corpo para NED, com as colunas de bodyAxesNED.
        function rotationMatrix(s) {
            const ax = bodyAxesNED(s);
            return [[ax[0][0], ax[1][0], ax[2][0]], [ax[0][1], ax[1][1], ax[2][1]], [ax[0][2], ax[1][2], ax[2][2]]];
        }
        // Método: fórmula padrão de ângulos de Euler ZYX para quaternião (metades dos ângulos).
        // Obtemos: o quaternião (w, x, y, z), que é o formato em que o robô regista a orientação no CSV.
        function quaternionOf(s) {
            const cr = Math.cos(s.roll * DEG / 2), sr = Math.sin(s.roll * DEG / 2);
            const cp = Math.cos(s.pitch * DEG / 2), sp = Math.sin(s.pitch * DEG / 2);
            const yw = wrap180(s.yaw);   // o yaw da linha temporal vem desenrolado (pode passar de 360°): q com w positivo
            const cy = Math.cos(yw * DEG / 2), sy = Math.sin(yw * DEG / 2);
            return { w: cr * cp * cy + sr * sp * sy, x: sr * cp * cy - cr * sp * sy, y: cr * sp * cy + sr * cp * sy, z: cr * cp * sy - sr * sp * cy };
        }
        // Obtemos: o texto SVG de uma linha (usado em todos os painéis do Cockpit).
        function svgLine(a, b, c, w, dash) {
            return '<line x1="' + a[0].toFixed(1) + '" y1="' + a[1].toFixed(1) + '" x2="' + b[0].toFixed(1) + '" y2="' + b[1].toFixed(1) +
                '" stroke="' + c + '" stroke-width="' + w + '"' + (dash ? ' stroke-dasharray="3 3"' : '') + ' stroke-linecap="round"/>';
        }
        // Obtemos: a moldura HTML de um painel do Cockpit (título e conteúdo).
        function panelHtml(cls, title, body) { return '<div class="ck-pn ' + cls + '"><h4>' + title + '</h4>' + body + '</div>'; }

        // Método: projetar em 2D (perspetiva fixa, com Este para a direita e Norte para dentro) um casco simples
        // rodado pela matriz de rotação. Roll, pitch e yaw aparecem em destaque, cada um com a sua explicação.
        // Obtemos: a atitude do robô desenhada e explicada.
        function attitudePanel(s) {
            const M = rotationMatrix(s), q = quaternionOf(s);
            const A = 35 * DEG, E = 26 * DEG, cx = 150, cy = 118, S = 54;
            const rotateByBody = (p) => [M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2], M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2], M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2]];
            const projectNed = (p) => {   // p em NED: ecrã com Este para a direita, Norte para dentro, Cima = -D
                const x = p[1], y = p[0], z = -p[2];
                const xr = x * Math.cos(A) - y * Math.sin(A), yr = x * Math.sin(A) + y * Math.cos(A);
                return [cx + S * xr, cy - S * (z * Math.cos(E) + yr * Math.sin(E))];
            };
            const bodyToScreen = (p) => projectNed(rotateByBody(p));
            const tail = [[-1.5, -.35, -.25], [-1.5, .35, -.25], [-1.5, .35, .25], [-1.5, -.35, .25]];
            const mid = [[.6, -.35, -.25], [.6, .35, -.25], [.6, .35, .25], [.6, -.35, .25]];
            const nose = [1.5, 0, 0];
            let g = '<svg viewBox="0 0 300 236" role="img">';
            [['N', [2.1, 0, 0]], ['E', [0, 2.1, 0]], ['D', [0, 0, 2.1]]].forEach(function (a) {
                const o = projectNed([0, 0, 0]), e = projectNed(a[1]);
                g += svgLine(o, e, '#4b6288', 1, true) + '<text x="' + (e[0] + 4) + '" y="' + (e[1] + 4) + '" font-size="10" fill="' + COCKPIT_COLORS.mut + '" font-family="monospace">' + a[0] + '</text>';
            });
            [tail, mid].forEach(function (face) { for (let k = 0; k < 4; k++) g += svgLine(bodyToScreen(face[k]), bodyToScreen(face[(k + 1) % 4]), '#7f93b5', 1.4); });
            for (let k = 0; k < 4; k++) g += svgLine(bodyToScreen(tail[k]), bodyToScreen(mid[k]), '#7f93b5', 1.4) + svgLine(bodyToScreen(mid[k]), bodyToScreen(nose), '#9db0cc', 1.4);
            [['x', [2.2, 0, 0], COCKPIT_COLORS.red], ['y', [0, 1.5, 0], COCKPIT_COLORS.green], ['z', [0, 0, 1.5], COCKPIT_COLORS.blue]].forEach(function (a) {
                const o = bodyToScreen([0, 0, 0]), e = bodyToScreen(a[1]);
                g += svgLine(o, e, a[2], 2.4) + '<circle cx="' + e[0].toFixed(1) + '" cy="' + e[1].toFixed(1) + '" r="3" fill="' + a[2] + '"/><text x="' + (e[0] + 6).toFixed(1) + '" y="' + (e[1] + 4).toFixed(1) + '" font-size="12" font-weight="700" fill="' + a[2] + '" font-family="monospace">' + a[0] + '</text>';
            });
            g += '</svg>';
            const att = function (name, val, sub, color) {
                return '<div class="ck-at"><div class="k">' + name + '</div><div class="v" style="color:' + color + '">' + val + '</div><div class="d">' + sub + '</div></div>';
            };
            return panelHtml('ck-c3', 'Atitude e eixos do corpo', g +
                '<div class="ck-att3">' +
                att('Rolamento (roll)', fmt1(s.roll) + '°', 'Inclinação para os lados, em torno do eixo x. Positivo: o lado direito desce.', COCKPIT_COLORS.red) +
                att('Arfagem (pitch)', fmt1(s.pitch) + '°', 'Nariz para cima ou para baixo, em torno do eixo y. Positivo: nariz para cima.', COCKPIT_COLORS.green) +
                att('Guinada (yaw)', fmt1(wrap180(s.yaw)) + '°', 'Para onde aponta o nariz, em torno do eixo z (rumo ' + fmt1(wrap360(s.yaw)) + '°). Positivo: nariz para a direita.', COCKPIT_COLORS.blue) +
                '</div>' +
                '<div class="ck-ro"><span>quaternion q = (<b>' + q.w.toFixed(3) + '</b>, ' + q.x.toFixed(3) + ', ' + q.y.toFixed(3) + ', ' + q.z.toFixed(3) + ')</span></div>');
        }

        // Método: o rumo (heading) vem do quaternião; a rota (course) é a direção da velocidade estimada, com
        // atan2(vE, vN), e só se mostra acima de 0.3 m por segundo; a deriva é rota menos rumo.
        // Obtemos: a bússola com o nariz (rumo), a direção em que realmente anda (rota) e o desvio entre os dois.
        function compassPanel(s) {
            const spd = Math.hypot(s.vN, s.vE);
            const cog = spd > 0.3 ? wrap360(Math.atan2(s.vE, s.vN) / DEG) : null;   // rota sobre o fundo, só quando anda
            const cx = 110, cy = 104, R = 78;
            const polar = (a, rr) => [cx + rr * Math.sin(a * DEG), cy - rr * Math.cos(a * DEG)];
            let g = '<svg viewBox="0 0 220 208" role="img"><circle cx="' + cx + '" cy="' + cy + '" r="' + R + '" fill="#0a1220" stroke="#27406a" stroke-width="1.5"/>';
            for (let a = 0; a < 360; a += 10) {
                const m = a % 90 === 0, p1 = polar(a, R), p2 = polar(a, R - (m ? 12 : (a % 30 === 0 ? 8 : 4)));
                g += svgLine(p1, p2, m ? '#9db0cc' : '#4b6288', m ? 1.8 : 1);
            }
            [['N', 0], ['E', 90], ['S', 180], ['O', 270]].forEach(function (c) {
                const p = polar(c[1], R - 24);
                g += '<text x="' + p[0].toFixed(1) + '" y="' + (p[1] + 5).toFixed(1) + '" text-anchor="middle" font-size="14" font-weight="700" fill="' + (c[0] === 'N' ? COCKPIT_COLORS.red : COCKPIT_COLORS.ink) + '">' + c[0] + '</text>';
            });
            if (cog !== null) {
                const p = polar(cog, R - 8);
                g += '<line x1="' + cx + '" y1="' + cy + '" x2="' + p[0].toFixed(1) + '" y2="' + p[1].toFixed(1) + '" stroke="' + COCKPIT_COLORS.amber + '" stroke-width="2.4" stroke-dasharray="5 3"/><circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="4.5" fill="' + COCKPIT_COLORS.amber + '"/>';
            }
            const tip = polar(s.yaw, R - 6), l = polar(s.yaw + 150, 16), rr = polar(s.yaw - 150, 16);
            g += '<polygon points="' + tip[0].toFixed(1) + ',' + tip[1].toFixed(1) + ' ' + l[0].toFixed(1) + ',' + l[1].toFixed(1) + ' ' + cx + ',' + cy + ' ' + rr[0].toFixed(1) + ',' + rr[1].toFixed(1) + '" fill="' + COCKPIT_COLORS.cy + '" opacity=".92"/></svg>';
            const beta = cog === null ? null : wrap180(cog - wrap360(s.yaw));
            return panelHtml('ck-c3', 'Bússola: rumo e rota', g +
                '<div class="ck-ro"><span style="color:' + COCKPIT_COLORS.cy + '">&#9632; rumo <b>' + fmt1(wrap360(s.yaw)) + '°</b></span><span style="color:' + COCKPIT_COLORS.amber + '">&#9679; rota <b>' + (cog === null ? 'parado' : fmt1(cog) + '°') + '</b></span></div>' +
                '<div class="ck-ro"><span>deriva <b>' + (beta === null ? '–' : fmt1(beta) + '°') + '</b></span><span>velocidade <b>' + fmt2(spd) + ' m/s</b></span></div>');
        }

        // Método: duas séries na mesma janela de 5 minutos: a velocidade medida (twist, no corpo) e a estimada a
        // partir das posições suavizadas. Os eixos ajustam-se ao máximo e mínimo da janela.
        // Obtemos: a comparação entre o que o robô diz que anda e o que as posições dizem.
        function speedPanel(r, f, s) {
            const W = 300, H = 116, l = 30, rr = 6, t = 8, b = 18, pw = W - l - rr, ph = H - t - b;
            const a = Math.max(0, Math.floor(f) - 150), z = Math.min(numSteps - 1, Math.floor(f) + 150);
            const okIdx = (k) => k >= r.k0 && k <= r.k1;
            let lo = -0.3, hi = 1.0;
            for (let k = a; k <= z; k++) if (okIdx(k)) { lo = Math.min(lo, r.u[k], r.spd[k]); hi = Math.max(hi, r.u[k], r.spd[k]); }
            const X = (k) => l + pw * (k - a) / Math.max(1, z - a), Y = (x) => t + ph * (1 - (x - lo) / (hi - lo));
            const path = function (arr) {
                let d = '', pen = false;
                for (let k = a; k <= z; k++) { if (!okIdx(k)) { pen = false; continue; } d += (pen ? 'L' : 'M') + X(k).toFixed(1) + ' ' + Y(arr[k]).toFixed(1) + ' '; pen = true; }
                return d;
            };
            let g = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img">';
            [lo, 0, hi].forEach(function (x) {
                g += '<line x1="' + l + '" x2="' + (l + pw) + '" y1="' + Y(x).toFixed(1) + '" y2="' + Y(x).toFixed(1) + '" stroke="' + (x === 0 ? '#4b6288' : COCKPIT_COLORS.line) + '"/><text x="' + (l - 4) + '" y="' + (Y(x) + 3.5).toFixed(1) + '" text-anchor="end" font-size="9.5" fill="' + COCKPIT_COLORS.mut + '" font-family="monospace">' + x.toFixed(1) + '</text>';
            });
            g += '<path d="' + path(r.spd) + '" fill="none" stroke="' + COCKPIT_COLORS.amber + '" stroke-width="1.8" stroke-dasharray="4 3"/><path d="' + path(r.u) + '" fill="none" stroke="' + COCKPIT_COLORS.cy + '" stroke-width="2"/>';
            g += '<line x1="' + X(f) + '" x2="' + X(f) + '" y1="' + t + '" y2="' + (t + ph) + '" stroke="' + COCKPIT_COLORS.ink + '" stroke-width="1.2"/>';
            g += '<text x="' + l + '" y="' + (H - 4) + '" font-size="9.5" fill="' + COCKPIT_COLORS.mut + '" font-family="monospace">−5 min</text><text x="' + (l + pw) + '" y="' + (H - 4) + '" text-anchor="end" font-size="9.5" fill="' + COCKPIT_COLORS.mut + '" font-family="monospace">+5 min</text></svg>';
            return panelHtml('ck-c3', 'Velocidade: medida e estimada', g +
                '<div class="ck-ro"><span style="color:' + COCKPIT_COLORS.cy + '">&#9472; medida (twist) <b>' + fmt2(s.u) + ' m/s</b></span><span style="color:' + COCKPIT_COLORS.amber + '">&#9548; das posições <b>' + fmt2(s.spd) + ' m/s</b></span></div>' +
                '<div class="ck-ro"><span>diferença <b>' + fmt2(s.spd - s.u) + ' m/s</b></span><span>velocidade lateral <b>' + fmt2(s.v) + '</b></span><span>velocidade vertical <b>' + fmt2(s.w) + '</b></span></div>');
        }

        // Método: o desenho completo da profundidade é calculado uma só vez por robô (depthPathCache) e depois só se
        // move o cursor. Os triângulos são os saltos de posição detetados no registo bruto.
        // Obtemos: a profundidade ao longo da missão, o estado (superfície ou imersão) e atalhos para os saltos.
        function depthPanel(r, id, f, s) {
            const W = 300, H = 130, l = 28, rr = 6, t = 8, b = 16, pw = W - l - rr, ph = H - t - b;
            const X = (k) => l + pw * k / (numSteps - 1), Y = (d) => t + ph * (Math.max(-2, Math.min(34, d)) + 2) / 36;
            if (!depthPathCache[id]) {
                let d = '';
                for (let k = Math.max(0, Math.ceil(r.k0)); k <= Math.min(numSteps - 1, Math.floor(r.k1)); k += 2) d += (d ? 'L' : 'M') + X(k).toFixed(1) + ' ' + Y(r.D[k]).toFixed(1) + ' ';
                depthPathCache[id] = d;
            }
            let g = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img">';
            [0, 10, 20, 30].forEach(function (d) {
                g += '<line x1="' + l + '" x2="' + (l + pw) + '" y1="' + Y(d).toFixed(1) + '" y2="' + Y(d).toFixed(1) + '" stroke="' + (d === 0 ? '#4b6288' : COCKPIT_COLORS.line) + '"/><text x="' + (l - 4) + '" y="' + (Y(d) + 3.5).toFixed(1) + '" text-anchor="end" font-size="9.5" fill="' + COCKPIT_COLORS.mut + '" font-family="monospace">' + d + '</text>';
            });
            g += '<path d="' + depthPathCache[id] + '" fill="none" stroke="' + r.color + '" stroke-width="2"/>';
            (r.jumps || []).forEach(function (e) {
                const x = X(e.t / dt_step);
                g += '<polygon points="' + x.toFixed(1) + ',' + (t + ph + 1) + ' ' + (x - 4).toFixed(1) + ',' + (t + ph + 9) + ' ' + (x + 4).toFixed(1) + ',' + (t + ph + 9) + '" fill="' + COCKPIT_COLORS.red + '"/>';
            });
            g += '<line x1="' + X(f).toFixed(1) + '" x2="' + X(f).toFixed(1) + '" y1="' + t + '" y2="' + (t + ph) + '" stroke="' + COCKPIT_COLORS.ink + '" stroke-width="1.2"/></svg>';
            const phase = s.D < 3 ? 'à superfície' : 'em imersão';
            const ev = (r.jumps || []).map(function (e) {
                return '<button class="ck-btn" onclick="jumpToMissionTime(' + e.t + ')">' + fmtElapsed(e.t) + ' · salto de ' + e.m.toFixed(1) + ' m' + (e.sub ? ' (imerso)' : '') + '</button>';
            }).join(' ');
            return panelHtml('ck-c3', 'Profundidade e eventos de navegação', g +
                '<div class="ck-ro"><span>profundidade (Down) <b>' + fmt1(s.D) + ' m</b></span><span>fase <b>' + phase + '</b></span></div>' +
                '<div class="ck-ro"><span>incerteza horizontal declarada (σ) <b>' + (s.sig * 100).toFixed(1) + ' cm</b></span></div>' +
                '<div class="ck-ro"><span>saltos no registo bruto:</span>' + (ev || '<b>nenhum</b>') + '</div>');
        }
        // Obtemos: salta a reprodução para um instante (segundos desde o início da missão).
        function jumpToMissionTime(relSec) {
            currentTime = t0_base + relSec;
            document.getElementById('time-slider').value = currentTime;
            onSliderChange(currentTime);
        }

        // Método: para cada vizinho, (1) vetor de posição relativa em NED; (2) rodá-lo para o corpo com toBody;
        // (3) azimute = atan2(y, x) e elevação = atan2(-z, distância horizontal); (4) velocidade relativa
        // rv e aproximação = menos (d·rv) a dividir pela distância; (5) tempo do ponto mais próximo
        // TCPA = menos (d·rv) a dividir por (rv·rv), limitado a 60 s, e CPA = distância nesse instante
        // (o mesmo cálculo de proximity.py).
        // Obtemos: o radar polar dos vizinhos no referencial do corpo e a tabela com distância, aproximação,
        // TCPA e CPA, com alerta de cor pela menor distância.
        function neighboursPanel(id, f, s) {
            const cx = 150, cy = 150, R0 = 128, rmap = (d) => R0 * Math.sqrt(Math.min(d, 400) / 400);
            const danger = timeline.collision_threshold_m, warn = timeline.proximity_warning_m;
            let g = '<svg viewBox="0 0 300 300" role="img">';
            g += '<circle cx="' + cx + '" cy="' + cy + '" r="' + rmap(danger).toFixed(1) + '" fill="' + COCKPIT_COLORS.red + '" opacity=".2"/>';
            [[danger, COCKPIT_COLORS.red, true], [warn, COCKPIT_COLORS.amber, true], [100, '#3b5280', false], [300, '#3b5280', false]].forEach(function (c) {
                g += '<circle cx="' + cx + '" cy="' + cy + '" r="' + rmap(c[0]).toFixed(1) + '" fill="none" stroke="' + c[1] + '" stroke-width="1"' + (c[2] ? ' stroke-dasharray="4 3"' : '') + '/><text x="' + (cx + 4) + '" y="' + (cy - rmap(c[0]) - 2).toFixed(1) + '" font-size="9.5" fill="' + COCKPIT_COLORS.ink2 + '" font-family="monospace">' + c[0] + ' m</text>';
            });
            g += '<line x1="' + cx + '" y1="' + (cy + R0) + '" x2="' + cx + '" y2="' + (cy - R0) + '" stroke="' + COCKPIT_COLORS.red + '" stroke-width="1.2"/><line x1="' + (cx - R0) + '" y1="' + cy + '" x2="' + (cx + R0) + '" y2="' + cy + '" stroke="' + COCKPIT_COLORS.green + '" stroke-width="1.2"/>';
            g += '<text x="' + (cx + 5) + '" y="' + (cy - R0 - 2) + '" font-size="11" font-weight="700" fill="' + COCKPIT_COLORS.red + '" font-family="monospace">x proa</text><text x="296" y="' + (cy - 5) + '" text-anchor="end" font-size="11" font-weight="700" fill="' + COCKPIT_COLORS.green + '" font-family="monospace">y estibordo</text>';
            g += '<polygon points="' + cx + ',' + (cy - 9) + ' ' + (cx + 6) + ',' + (cy + 7) + ' ' + (cx - 6) + ',' + (cy + 7) + '" fill="' + timeline.robots[id].color + '" stroke="' + COCKPIT_COLORS.ink + '" stroke-width="1"/>';
            let rows = '', minD = null;
            robotKeys.forEach(function (o) {
                if (o === id) return;
                const ro = timeline.robots[o];
                const tag = '<b style="color:' + ro.color + '">&#9632;</b> ' + o;
                if (!isActive(ro, f)) {
                    rows += '<tr><td>' + tag + '</td><td colspan="6" style="color:' + COCKPIT_COLORS.mut + '">sem dados (ativo de ' + fmtElapsed(ro.k0 * dt_step) + ' a ' + fmtElapsed(ro.k1 * dt_step) + ')</td></tr>';
                    return;
                }
                const t = fullStateAt(ro, f);
                const dN = t.N - s.N, dE = t.E - s.E, dD = t.D - s.D;
                const b = toBody(s, dN, dE, dD);
                const hor = Math.hypot(b.x, b.y), rng = Math.hypot(b.x, b.y, b.z);
                const az = Math.atan2(b.y, b.x) / DEG, el = Math.atan2(-b.z, hor) / DEG;
                const rv = [t.vN - s.vN, t.vE - s.vE, t.vD - s.vD], d = [dN, dE, dD];
                const vv = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2], dot = d[0] * rv[0] + d[1] * rv[1] + d[2] * rv[2];
                const clos = -dot / Math.max(rng, 1e-6);
                // TCPA e CPA como em proximity.py: movimento linear, horizonte de 60 s
                let tcpaTxt, cpa = rng;
                if (vv < 1e-8) tcpaTxt = 'estável';
                else {
                    const tc = -dot / vv;
                    if (tc <= 0) tcpaTxt = 'a afastar';
                    else {
                        const th = Math.min(tc, 60.0);
                        cpa = Math.hypot(d[0] + rv[0] * th, d[1] + rv[1] * th, d[2] + rv[2] * th);
                        tcpaTxt = tc > 60.0 ? '> 60 s' : fmtElapsed(tc);
                    }
                }
                if (minD === null || rng < minD) minD = rng;
                const p = [cx + rmap(rng) * Math.sin(az * DEG), cy - rmap(rng) * Math.cos(az * DEG)];
                g += '<line x1="' + cx + '" y1="' + cy + '" x2="' + p[0].toFixed(1) + '" y2="' + p[1].toFixed(1) + '" stroke="' + ro.color + '" stroke-width="1" stroke-dasharray="2 3" opacity=".7"/>' +
                    '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="6" fill="' + ro.color + '" stroke="' + COCKPIT_COLORS.ink + '" stroke-width="1.5"/>' +
                    '<text x="' + (p[0] + 9).toFixed(1) + '" y="' + (p[1] + 4).toFixed(1) + '" font-size="11" font-weight="600" fill="' + COCKPIT_COLORS.ink + '">' + o.replace('Robot ', 'R') + ' · ' + rng.toFixed(0) + ' m</text>';
                rows += '<tr><td>' + tag + '</td><td class="n">' + fmt1(az) + '°</td><td class="n">' + fmt1(el) + '°</td><td class="n">' + fmt1(rng) + '</td><td class="n">' + (clos >= 0 ? '+' : '') + fmt2(clos) +
                    '</td><td class="n">' + tcpaTxt + '</td><td class="n">' + fmt1(cpa) + '</td></tr>';
            });
            g += '</svg>';
            const cls = minD === null ? 'none' : (minD < danger ? 'crit' : (minD < warn ? 'warn' : 'safe'));
            const txt = minD === null ? 'Sem outros robots ativos' : (minD < danger ? 'Perigo de colisão' : (minD < warn ? 'Aviso de proximidade' : 'Sem risco de colisão'));
            return panelHtml('ck-c12', 'Vizinhos no referencial do corpo',
                '<div class="ck-alert ' + cls + '"><span>' + txt + '</span><span style="font-weight:600;letter-spacing:0;text-transform:none">' + (minD === null ? 'sem dados para comparar' : 'distância mínima ' + minD.toFixed(1) + ' m') + '</span></div>' +
                '<div class="ck-nb"><div>' + g + '</div><div class="ck-tw"><table><thead><tr><th>Vizinho</th><th class="n">Azimute</th><th class="n">Elevação</th><th class="n">Distância (m)</th><th class="n">Aproximação (m/s)</th><th class="n">Tempo até ao CPA</th><th class="n">CPA (m)</th></tr></thead><tbody>' + rows + '</tbody></table></div></div>' +
                '<div class="ck-ro"><span>Δ(corpo) = Rᵀ · (p(vizinho) − p(este)) &nbsp; azimute = atan2(y, x) &nbsp; elevação = atan2(−z, √(x²+y²)) &nbsp; CPA = ponto de maior aproximação, TCPA = tempo até lá</span></div>');
        }

        // Obtemos: reconstrói os painéis do separador Cockpit para o robô selecionado no instante f.
        function updateCockpitView(f) {
            const root = document.getElementById('cockpit-root');
            const r = timeline.robots[currentRobot];
            if (!root || !r) return;
            const head = '<div class="ck-head ck-c12"><b>Cockpit · ' + currentRobot + '</b>';
            if (!isActive(r, f)) {
                root.innerHTML = head + '</div><div class="ck-pn ck-c12"><h4>' + currentRobot + ' sem dados neste instante</h4>' +
                    '<p style="color:' + COCKPIT_COLORS.ink2 + ';margin-bottom:10px">Este robot tem dados de ' + fmtElapsed(r.k0 * dt_step) + ' a ' + fmtElapsed(r.k1 * dt_step) + '. Escolha um instante dentro desta janela.</p>' +
                    '<button class="ck-btn" onclick="jumpToMissionTime(' + (r.k0 * dt_step + 2).toFixed(1) + ')">Ir para o início do ' + currentRobot + '</button></div>';
                return;
            }
            const s = fullStateAt(r, f);
            root.innerHTML = head + '<span>Posição NED: Norte <b>' + fmt1(s.N) + '</b> m · Este <b>' + fmt1(s.E) + '</b> m · Profundidade (Down) <b>' + fmt1(s.D) + '</b> m</span></div>' +
                attitudePanel(s) + compassPanel(s) + speedPanel(r, f, s) + depthPanel(r, currentRobot, f, s) + neighboursPanel(currentRobot, f, s);
        }

        // ---------- Previsão da trajetória (Vista Global e Vista 3ª Pessoa) ----------
        // Segmento de reta com a velocidade estimada (NED) do instante atual, durante PRED_H segundos.
        // Se a reta ia sair da água, termina à superfície.
        const PRED_H = 60.0;
        let predictOn = true;
        // Obtemos: o atalho P (liga ou desliga as linhas de previsão de 60 s).
        function togglePredict() {
            predictOn = !predictOn;
            document.getElementById('btn-predict').classList.toggle('active', predictOn);
            updateFrame(curF());
        }

        // ---------- Painel de bordo da Vista 1ª Pessoa ----------
        // Por cima da câmara 3D: janela do submarino, fita de rumo, profundidade, velocidade e vizinho mais próximo.
        let hudOn = true;
        const HUD_SPAN = 35;     // metade do campo de rumos da fita [graus]
        const HUD_PX = 8;        // pixels por grau na fita
        const HUD_NAMES = { 0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SO', 270: 'O', 315: 'NO' };

        // Método: para cada outro robô ativo, distância 3D, azimute e elevação (rodando o vetor para o corpo com
        // toBody), rumo absoluto e aproximação = menos (d·rv) a dividir pela distância; guarda-se o mais próximo.
        // Obtemos: o vizinho mais próximo com tudo o que o painel de bordo mostra.
        function nearestNeighbour(id, f) {
            const me = timeline.robots[id];
            if (!me || !isActive(me, f)) return null;
            const s = fullStateAt(me, f);
            let best = null;
            robotKeys.forEach(function (o) {
                if (o === id) return;
                const ro = timeline.robots[o];
                if (!isActive(ro, f)) return;
                const t = fullStateAt(ro, f);
                const dN = t.N - s.N, dE = t.E - s.E, dD = t.D - s.D;
                const rng = Math.hypot(dN, dE, dD);
                if (best !== null && rng >= best.rng) return;
                const b = toBody(s, dN, dE, dD);
                const dot = dN * (t.vN - s.vN) + dE * (t.vE - s.vE) + dD * (t.vD - s.vD);
                best = {
                    id: o, color: ro.color, rng: rng,
                    az: Math.atan2(b.y, b.x) / DEG, el: Math.atan2(-b.z, Math.hypot(b.x, b.y)) / DEG,
                    brg: wrap360(Math.atan2(dE, dN) / DEG), clos: -dot / Math.max(rng, 1e-6)
                };
            });
            return best;
        }

        // Método: desenhar em SVG uma régua de rumos que desliza (HUD_PX píxeis por grau), com o rumo atual ao centro
        // e um losango na direção do vizinho (uma seta no bordo se estiver fora do campo).
        // Obtemos: a fita de rumo, como a de um avião, a mostrar para onde fica o vizinho.
        function hudTape(hdg, nb) {
            const W = 2 * HUD_SPAN * HUD_PX, H = 74, cx = W / 2;
            let g = '<svg viewBox="0 0 ' + W + ' ' + H + '" width="' + W + '" height="' + H + '">';
            g += '<rect x="0" y="14" width="' + W + '" height="32" rx="4" fill="rgba(6,18,34,.66)" stroke="' + COCKPIT_COLORS.cy + '" stroke-opacity=".55"/>';
            for (let k = Math.ceil(hdg - HUD_SPAN); k <= Math.floor(hdg + HUD_SPAN); k++) {
                if (k % 5 !== 0) continue;
                const x = (cx + (k - hdg) * HUD_PX).toFixed(1), a = wrap360(k), major = (a % 10 === 0);
                g += '<line x1="' + x + '" y1="46" x2="' + x + '" y2="' + (major ? 36 : 41) + '" stroke="' + COCKPIT_COLORS.ink2 + '" stroke-width="1.2"/>';
                if (HUD_NAMES[a] !== undefined) g += '<text x="' + x + '" y="32" text-anchor="middle" font-size="14" font-weight="800" fill="' + COCKPIT_COLORS.cy + '">' + HUD_NAMES[a] + '</text>';
                else if (major) g += '<text x="' + x + '" y="32" text-anchor="middle" font-size="11" fill="' + COCKPIT_COLORS.ink2 + '" font-family="monospace">' + a + '</text>';
            }
            g += '<polygon points="' + (cx - 7) + ',3 ' + (cx + 7) + ',3 ' + cx + ',14" fill="' + COCKPIT_COLORS.amber + '"/>';
            g += '<text x="' + cx + '" y="68" text-anchor="middle" font-size="17" font-weight="700" fill="' + COCKPIT_COLORS.ink + '" font-family="monospace">RUMO ' + fmt1(hdg) + '°</text>';
            if (nb) {
                const dl = wrap180(nb.brg - hdg);
                if (Math.abs(dl) <= HUD_SPAN) {
                    const x = cx + dl * HUD_PX;
                    g += '<polygon points="' + x.toFixed(1) + ',17 ' + (x + 7).toFixed(1) + ',24 ' + x.toFixed(1) + ',31 ' + (x - 7).toFixed(1) + ',24" fill="' + nb.color + '" stroke="' + COCKPIT_COLORS.ink + '" stroke-width="1.3"/>';
                } else if (dl < 0) {
                    g += '<polygon points="5,24 16,17 16,31" fill="' + nb.color + '" stroke="' + COCKPIT_COLORS.ink + '" stroke-width="1"/>';
                } else {
                    g += '<polygon points="' + (W - 5) + ',24 ' + (W - 16) + ',17 ' + (W - 16) + ',31" fill="' + nb.color + '" stroke="' + COCKPIT_COLORS.ink + '" stroke-width="1"/>';
                }
            }
            return g + '</svg>';
        }

        // Método: reconstruir o HTML do painel com os valores do instante f (fullStateAt do robô e nearestNeighbour).
        // Obtemos: fita de rumo, caixas com profundidade, arfagem, rolamento, velocidades e taxa de viragem, e a
        // caixa do vizinho mais próximo (amarela abaixo do aviso, vermelha a piscar abaixo do limiar de colisão).
        function updateFirstPersonHud(f) {
            const root = document.getElementById('fp-hud'), dyn = document.getElementById('hud-dyn');
            if (!root || !dyn) return;
            root.classList.toggle('off', !hudOn);
            if (!hudOn) { dyn.innerHTML = ''; return; }
            const r = timeline.robots[currentRobot];
            if (!r || !isActive(r, f)) {
                dyn.innerHTML = '<div class="hud-msg">' + currentRobot + ' sem dados neste instante<br><span>dados de ' + fmtElapsed(r.k0 * dt_step) + ' a ' + fmtElapsed(r.k1 * dt_step) + '</span></div>';
                return;
            }
            const s = fullStateAt(r, f), nb = nearestNeighbour(currentRobot, f);
            const danger = timeline.collision_threshold_m, warn = timeline.proximity_warning_m;
            const lv = nb === null ? 'none' : (nb.rng < danger ? 'crit' : (nb.rng < warn ? 'warn' : 'safe'));
            const kv = function (k, v) { return '<div class="hud-k">' + k + '</div><div class="hud-v">' + v + '</div>'; };
            let h = '<div class="hud-tape">' + hudTape(wrap360(s.yaw), nb) + '</div>';
            h += '<div class="hud-box hud-l"><div class="hud-t">' + currentRobot + '</div>' + kv('Profundidade', fmt1(s.D) + ' m') +
                kv('Arfagem (pitch)', fmt1(s.pitch) + '°') + kv('Rolamento (roll)', fmt1(s.roll) + '°') + '</div>';
            h += '<div class="hud-box hud-r">' + kv('Velocidade estimada', fmt2(s.spd) + ' m/s') + kv('Velocidade medida', fmt2(s.u) + ' m/s') +
                kv('Taxa de viragem', fmt1(s.turn) + ' °/s') + '</div>';
            if (nb === null) h += '<div class="hud-nb none">Sem outros robots ativos neste instante</div>';
            else h += '<div class="hud-nb ' + lv + '"><b style="color:' + nb.color + '">&#9670;</b> Vizinho mais próximo: <b>' + nb.id + '</b> &nbsp; ' +
                '<b>' + nb.rng.toFixed(1) + ' m</b> &nbsp; azimute <b>' + (nb.az >= 0 ? '+' : '') + fmt1(nb.az) + '°</b> &nbsp; elevação <b>' + (nb.el >= 0 ? '+' : '') + fmt1(nb.el) + '°</b>' +
                ' &nbsp; aproximação <b>' + (nb.clos >= 0 ? '+' : '') + fmt2(nb.clos) + ' m/s</b>' +
                (lv === 'crit' ? ' &nbsp; <b>PERIGO DE COLISÃO</b>' : (lv === 'warn' ? ' &nbsp; <b>ATENÇÃO</b>' : '')) + '</div>';
            dyn.innerHTML = h;
        }

        // Obtemos: o atalho H (mostra ou esconde o painel de bordo).
        function toggleHud() {
            hudOn = !hudOn;
            document.getElementById('btn-hud').innerHTML = 'Painel de bordo: ' + (hudOn ? 'sim' : 'não') + ' (H)';
            updateFrame(curF());
        }

        // Obtemos: passa a ver o mar pelo robô mais próximo do atual (os olhos do vizinho).
        function firstPersonNearest() {
            const nb = nearestNeighbour(currentRobot, curF());
            if (nb) setRobot(nb.id);
        }

        // ---------- Separador 5: encontros e relatório da missão ----------
        // Os números vêm de Python (mission_report.py): posições suavizadas, grelha de 1 s, distância 3D.
        let reportBuilt = false;
        const REPORT_LEVELS = { perigo: ['crit', 'Perigo'], atencao: ['warn', 'Atenção'], aproximacao: ['apr', 'Aproximação'] };
        const pairLabel = (p) => p.a + ' e ' + p.b;
        const reportTime = (x) => fmtElapsed(x);

        // Obtemos: salta para um instante do relatório e abre a Vista Global para ver o encontro.
        function reportGoTo(rel) {
            const target = Math.max(t0_base, Math.min(t_max, t0_base + rel));
            document.getElementById('time-slider').value = target;
            onSliderChange(target);
            setView('global');
        }

        // Método: juntar os episódios de todos os pares e ordenar pelo instante inicial.
        // Obtemos: a lista cronológica de encontros da missão.
        function allEncounters() {
            const out = [];
            timeline.report.pairs.forEach(function (p) {
                p.episodes.forEach(function (e) { out.push(Object.assign({ a: p.a, b: p.b }, e)); });
            });
            out.sort(function (x, y) { return x.t0 - y.t0; });
            return out;
        }

        // Obtemos: a etiqueta colorida do nível do encontro (perigo, atenção ou aproximação).
        function levelTag(lv) { const c = REPORT_LEVELS[lv]; return '<span class="rp-lv ' + c[0] + '">' + c[1] + '</span>'; }

        // Método: ler timeline.report (calculado em mission_report.py) e escrever as tabelas em HTML uma só vez.
        // Obtemos: o separador 5 com resumo, encontros por par, saltos de posição e botões de exportação.
        function buildReport() {
            const R = timeline.report, T = R.thresholds, enc = allEncounters(), og = timeline.origin;
            const root = document.getElementById('report-root');
            if (!root) return;
            let best = null;
            R.pairs.forEach(function (p) { if (p.d_min !== null && (best === null || p.d_min < best.d_min)) best = p; });
            const nDanger = enc.filter(function (e) { return e.level === 'perigo'; }).length;
            const nWarn = enc.filter(function (e) { return e.level === 'atencao'; }).length;
            const nJumpEnc = enc.filter(function (e) { return e.jump_near; }).length;
            let h = '<div class="ck-head"><b>Encontros e relatório da missão</b><span>Origem NED: <b>' + og.text + '</b></span></div>';

            // Resumo
            const lvTop = best === null ? 'none' : (best.d_min < T.danger ? 'crit' : (best.d_min < T.warning ? 'warn' : 'safe'));
            h += '<div class="ck-alert ' + lvTop + '"><span>' + (best === null ? 'Sem pares de robots com dados em simultâneo' :
                'Menor distância da missão: ' + best.d_min.toFixed(1) + ' m (' + pairLabel(best) + ', aos ' + reportTime(best.t_min) + ')') + '</span>' +
                '<span style="font-weight:600;letter-spacing:0;text-transform:none">' + enc.length + ' encontros abaixo de ' + T.encounter + ' m · ' +
                nWarn + ' de atenção (abaixo de ' + T.warning + ' m) · ' + nDanger + ' de perigo (abaixo de ' + T.danger + ' m) · ' + R.jumps.length + ' saltos de posição</span></div>';

            if (collision !== null) {
                h += '<div class="ck-alert crit"><span>COLISÃO: ' + collisionText() + '</span><span style="font-weight:600;letter-spacing:0;text-transform:none">abaixo de ' + (timeline.contact_distance_m || 2.0) + ' m' + (freezeOn ? ' · os dois robots ficam parados' : ' · dados reais') + '</span></div>';
            }

            // Robots
            h += panelHtml('', 'Robots e janelas de dados', '<div class="ck-tw"><table><thead><tr><th>Robot</th><th class="n">Dados de</th><th class="n">até</th><th class="n">Duração</th><th class="n">Amostras</th><th class="n">Profundidade máxima (m)</th><th class="n">Saltos de posição</th></tr></thead><tbody>' +
                R.robots.map(function (o) {
                    return '<tr><td><b style="color:' + timeline.robots[o.id].color + '">&#9632;</b> ' + o.label + '</td><td class="n">' + reportTime(o.start) + '</td><td class="n">' + reportTime(o.end) + '</td><td class="n">' + reportTime(o.end - o.start) +
                        '</td><td class="n">' + o.n + '</td><td class="n">' + o.max_depth.toFixed(1) + '</td><td class="n">' + o.n_jumps + '</td></tr>';
                }).join('') + '</tbody></table></div>');

            // Resumo por par
            h += panelHtml('', 'Resumo por par de robots', '<div class="ck-tw"><table><thead><tr><th>Par</th><th class="n">Janela comum</th><th class="n">Distância mínima (m)</th><th class="n">Quando</th><th class="n">Tempo abaixo de ' + T.encounter + ' m</th><th class="n">abaixo de ' + T.warning + ' m</th><th class="n">abaixo de ' + T.danger + ' m</th><th></th></tr></thead><tbody>' +
                R.pairs.map(function (p) {
                    if (p.overlap === null) return '<tr><td>' + pairLabel(p) + '</td><td colspan="7" style="color:' + COCKPIT_COLORS.mut + '">sem dados em simultâneo</td></tr>';
                    return '<tr><td>' + pairLabel(p) + '</td><td class="n">' + reportTime(p.overlap[0]) + ' a ' + reportTime(p.overlap[1]) + '</td><td class="n"><b>' + p.d_min.toFixed(1) + '</b></td><td class="n">' + reportTime(p.t_min) +
                        '</td><td class="n">' + p.below.w60.toFixed(0) + ' s</td><td class="n">' + p.below.w30.toFixed(0) + ' s</td><td class="n">' + p.below.w10.toFixed(0) + ' s</td>' +
                        '<td><button class="ck-btn" onclick="reportGoTo(' + (p.t_min - 10).toFixed(1) + ')">Ir para o mínimo</button></td></tr>';
                }).join('') + '</tbody></table></div>');

            // Encontros
            h += panelHtml('', 'Encontros: períodos com distância abaixo de ' + T.encounter + ' m', enc.length === 0 ? '<p style="color:' + COCKPIT_COLORS.ink2 + '">Nenhum encontro.</p>' :
                '<div class="ck-tw"><table><thead><tr><th class="n">#</th><th>Par</th><th class="n">Início</th><th class="n">Fim</th><th class="n">Duração</th><th class="n">Distância mínima (m)</th><th class="n">No instante</th><th>Nível</th><th>Observação</th><th></th></tr></thead><tbody>' +
                enc.map(function (e, i) {
                    return '<tr class="rp-row" data-t0="' + e.t0 + '" data-t1="' + e.t1 + '"><td class="n">' + (i + 1) + '</td><td>' + pairLabel(e) + '</td><td class="n">' + reportTime(e.t0) + '</td><td class="n">' + reportTime(e.t1) + '</td><td class="n">' + (e.t1 - e.t0 + 1).toFixed(0) + ' s</td><td class="n"><b>' + e.d_min.toFixed(1) +
                        '</b></td><td class="n">' + reportTime(e.t_min) + '</td><td>' + levelTag(e.level) + '</td><td style="color:' + COCKPIT_COLORS.amber + '">' + (e.jump_near ? 'coincide com salto de posição: confirmar' : '') +
                        '</td><td><button class="ck-btn" onclick="reportGoTo(' + (e.t_min - 10).toFixed(1) + ')">Ver na vista global</button></td></tr>';
                }).join('') + '</tbody></table></div>');

            // Saltos
            h += panelHtml('', 'Saltos de posição no registo bruto', R.jumps.length === 0 ? '<p style="color:' + COCKPIT_COLORS.ink2 + '">Nenhum salto detetado.</p>' :
                '<div class="ck-tw"><table><thead><tr><th>Robot</th><th class="n">Instante</th><th class="n">Salto (m)</th><th class="n">Profundidade (m)</th><th>Estado</th><th></th></tr></thead><tbody>' +
                R.jumps.map(function (j) {
                    return '<tr><td><b style="color:' + timeline.robots[j.robot].color + '">&#9632;</b> ' + j.robot + '</td><td class="n">' + reportTime(j.t) + '</td><td class="n"><b>' + j.m.toFixed(1) + '</b></td><td class="n">' + j.depth.toFixed(1) +
                        '</td><td>' + (j.depth > 3 ? 'imerso' : 'à superfície') + '</td><td><button class="ck-btn" onclick="reportGoTo(' + (j.t - 10).toFixed(1) + ')">Ver na vista global</button></td></tr>';
                }).join('') + '</tbody></table></div>');

            // Exportar e notas
            h += panelHtml('', 'Exportar e como ler', '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px"><button onclick="reportExportMd()">Descarregar relatório (.md)</button><button onclick="reportExportCsv()">Descarregar encontros (.csv)</button></div>' +
                '<p class="rp-note">Distância = distância 3D (Norte, Este e profundidade) entre as posições suavizadas (filtro de Kalman com suavizador RTS), calculada de 1 em 1 s na janela em que os dois robots têm dados. ' +
                'Um encontro é um período seguido com distância abaixo de ' + T.encounter + ' m; o nível é dado pela distância mínima desse período (perigo abaixo de ' + T.danger + ' m, atenção abaixo de ' + T.warning + ' m). ' +
                'Um salto é uma mudança de posição no registo bruto que implicaria mais de 2,5 m/s: um AUV não consegue isso, por isso é um defeito do registo. Um encontro que coincide com um salto (até 5 s de margem) deve ser confirmado antes de ser tratado como real. ' +
                'Os botões «Ver na vista global» saltam para 10 s antes do instante.</p>');
            root.innerHTML = h;
            reportBuilt = true;
        }

        // Obtemos: destaca a linha do encontro que está a decorrer no instante atual.
        function updateReportView(f) {
            if (!reportBuilt) buildReport();
            const t = f * dt_step;
            document.querySelectorAll('#report-root tr.rp-row').forEach(function (tr) {
                tr.classList.toggle('now', t >= parseFloat(tr.dataset.t0) && t <= parseFloat(tr.dataset.t1));
            });
        }

        // Método: criar um Blob em memória e um link temporário com download.
        // Obtemos: guardar um ficheiro no computador sem servidor.
        function reportDownload(name, text, mime) {
            const blob = new Blob([text], { type: mime });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = name;
            document.body.appendChild(a); a.click();
            setTimeout(function () { document.body.removeChild(a); URL.revokeObjectURL(url); }, 500);
        }

        // Obtemos: o relatório em CSV (uma linha por encontro) para abrir em Excel.
        function reportExportCsv() {
            const NL = String.fromCharCode(10), BOM = String.fromCharCode(0xFEFF);
            const rows = [['n', 'robot_a', 'robot_b', 'inicio_s', 'fim_s', 'duracao_s', 'distancia_minima_m', 'instante_minimo_s', 'nivel', 'coincide_com_salto'].join(';')];
            allEncounters().forEach(function (e, i) {
                rows.push([i + 1, e.a, e.b, e.t0.toFixed(0), e.t1.toFixed(0), (e.t1 - e.t0 + 1).toFixed(0), e.d_min.toFixed(2), e.t_min.toFixed(0), REPORT_LEVELS[e.level][1], e.jump_near ? 'sim' : 'nao'].join(';'));
            });
            reportDownload('encontros_tomsa.csv', BOM + rows.join(NL) + NL, 'text/csv;charset=utf-8');
        }

        // Obtemos: o relatório em Markdown, pronto para colar no relatório do trabalho.
        function reportExportMd() {
            const NL = String.fromCharCode(10), R = timeline.report, T = R.thresholds, enc = allEncounters();
            const L = [];
            L.push('# Relatório da missão TOMSA', '', 'Origem NED: ' + timeline.origin.text, '');
            L.push('Limiares: perigo abaixo de ' + T.danger + ' m, atenção abaixo de ' + T.warning + ' m, encontro abaixo de ' + T.encounter + ' m.', '');
            L.push('## Robots', '', '| Robot | Dados de | até | Duração | Profundidade máxima (m) | Saltos de posição |', '|---|---|---|---|---|---|');
            R.robots.forEach(function (o) { L.push('| ' + o.label + ' | ' + reportTime(o.start) + ' | ' + reportTime(o.end) + ' | ' + reportTime(o.end - o.start) + ' | ' + o.max_depth.toFixed(1) + ' | ' + o.n_jumps + ' |'); });
            L.push('', '## Resumo por par', '', '| Par | Distância mínima (m) | Quando | Abaixo de ' + T.encounter + ' m | Abaixo de ' + T.warning + ' m | Abaixo de ' + T.danger + ' m |', '|---|---|---|---|---|---|');
            R.pairs.forEach(function (p) {
                if (p.overlap === null) L.push('| ' + pairLabel(p) + ' | sem dados em simultâneo | | | | |');
                else L.push('| ' + pairLabel(p) + ' | ' + p.d_min.toFixed(1) + ' | ' + reportTime(p.t_min) + ' | ' + p.below.w60.toFixed(0) + ' s | ' + p.below.w30.toFixed(0) + ' s | ' + p.below.w10.toFixed(0) + ' s |');
            });
            L.push('', '## Encontros', '', '| # | Par | Início | Fim | Distância mínima (m) | No instante | Nível | Observação |', '|---|---|---|---|---|---|---|---|');
            enc.forEach(function (e, i) {
                L.push('| ' + (i + 1) + ' | ' + pairLabel(e) + ' | ' + reportTime(e.t0) + ' | ' + reportTime(e.t1) + ' | ' + e.d_min.toFixed(1) + ' | ' + reportTime(e.t_min) + ' | ' + REPORT_LEVELS[e.level][1] + ' | ' + (e.jump_near ? 'coincide com salto de posição' : '') + ' |');
            });
            L.push('', '## Saltos de posição no registo bruto', '', '| Robot | Instante | Salto (m) | Profundidade (m) | Estado |', '|---|---|---|---|---|');
            R.jumps.forEach(function (j) { L.push('| ' + j.robot + ' | ' + reportTime(j.t) + ' | ' + j.m.toFixed(1) + ' | ' + j.depth.toFixed(1) + ' | ' + (j.depth > 3 ? 'imerso' : 'à superfície') + ' |'); });
            L.push('', 'Distância 3D entre posições suavizadas (RTS), de 1 em 1 s. Tempos desde o início da missão.');
            reportDownload('relatorio_missao_tomsa.md', L.join(NL) + NL, 'text/markdown;charset=utf-8');
        }

        // ---------- Gráfico 2D de distâncias: move o cursor de tempo ----------
        // Método: mover só a forma shapes[1] (o cursor vertical) do gráfico de distâncias.
        // Obtemos: o cursor de tempo a acompanhar a reprodução sem refazer o gráfico.
        function updateProxView(f) {
            const gd = document.getElementById('plot-prox');
            if (!gd || !gd.layout || !gd.layout.shapes || gd.layout.shapes.length < 2) return;
            const tRel = f * dt_step;
            Plotly.relayout(gd, { 'shapes[1].x0': tRel, 'shapes[1].x1': tRel });
        }


        // ---------- Colisão ----------
        // Método: para cada par de robôs ativos, percorrer a grelha de tempo e guardar o primeiro instante em que a
        // distância 3D fica abaixo de contact_distance_m, com a distância nesse instante.
        // Obtemos: collision (ou null). Se existir, o Play pára nesse instante, aparece a faixa COLISÃO e os dois robôs
        // ficam parados (as séries deles repetem o valor do instante da colisão). O botão volta aos dados reais.
        let collision = null;
        let freezeOn = true;
        const collisionOrig = {};
        const FREEZE_KEYS = ['N', 'E', 'D', 'roll', 'pitch', 'yaw', 'spd', 'turn', 'dist', 'vN', 'vE', 'vD', 'u', 'v', 'w', 'sig'];
        const ZERO_KEYS = ['spd', 'turn', 'vN', 'vE', 'vD', 'u', 'v', 'w'];

        function pairDistance(ra, rb, k) {
            const dn = ra.N[k] - rb.N[k], de = ra.E[k] - rb.E[k], dd = ra.D[k] - rb.D[k];
            return Math.sqrt(dn * dn + de * de + dd * dd);
        }

        function detectCollision() {
            const lim = timeline.contact_distance_m || 2.0;
            let best = null;
            for (let i = 0; i < robotKeys.length; i++) {
                for (let j = i + 1; j < robotKeys.length; j++) {
                    const ra = timeline.robots[robotKeys[i]], rb = timeline.robots[robotKeys[j]];
                    for (let k = 0; k < numSteps; k++) {
                        if (!(isActive(ra, k) && isActive(rb, k))) continue;
                        if (pairDistance(ra, rb, k) < lim) {
                            if (best === null || k < best.k) best = { k: k, a: robotKeys[i], b: robotKeys[j] };
                            break;
                        }
                    }
                }
            }
            if (best === null) return;
            const ra = timeline.robots[best.a], rb = timeline.robots[best.b];
            collision = { k: best.k, t: t0_base + best.k * dt_step, a: best.a, b: best.b, d: pairDistance(ra, rb, best.k) };
            document.getElementById('btn-freeze').style.display = '';
            applyCollisionFreeze();
        }

        // Obtemos: os dois robôs da colisão parados (freezeOn) ou com os dados reais (guardados em collisionOrig).
        function applyCollisionFreeze() {
            if (!collision) return;
            const kc = collision.k;
            [collision.a, collision.b].forEach(id => {
                const r = timeline.robots[id];
                if (!collisionOrig[id]) collisionOrig[id] = {};
                FREEZE_KEYS.forEach(key => {
                    if (!collisionOrig[id][key]) collisionOrig[id][key] = r[key];
                    const src = collisionOrig[id][key];
                    if (!freezeOn) { r[key] = src; return; }
                    const out = src.slice(0, kc + 1);
                    const hold = ZERO_KEYS.indexOf(key) >= 0 ? 0 : src[kc];
                    for (let k = kc + 1; k < src.length; k++) out.push(hold);
                    r[key] = out;
                });
            });
            const b = document.getElementById('btn-freeze');
            b.textContent = freezeOn ? 'Robots parados (C)' : 'Dados reais (C)';
            b.classList.toggle('active', !freezeOn);
        }

        function toggleFreeze() {
            if (!collision) return;
            freezeOn = !freezeOn;
            applyCollisionFreeze();
            updateFrame(curF());
        }

        // Obtemos: texto da colisão para a faixa e para o relatório.
        function collisionText() {
            return collision.a + ' e ' + collision.b + ' aos ' + fmtElapsed(collision.t - t0_base) + ': <b>' + collision.d.toFixed(1) + ' m</b>';
        }

        // ---------- Aviso de proximidade (Vista Global e Vista 3ª Pessoa) ----------
        // Amarelo: dois robôs ativos a menos de proximity_warning_m. Vermelho a piscar: menos de collision_threshold_m.
        // Método: para cada par de robôs ativos, distância 3D = raiz de (dN2 + dE2 + dD2).
        // Obtemos: faixa amarela (abaixo do aviso) ou vermelha a piscar (abaixo do limiar de colisão).
        function updateProximityBanner(f) {
            const box = document.getElementById('proximity-banner');
            if (!box) return;
            if (currentView !== 'global' && currentView !== 'pov' && currentView !== 'fp') { box.style.display = 'none'; return; }
            box.style.top = (currentView === 'fp' && hudOn) ? '124px' : '28px';   // na 1ª pessoa fica por baixo da fita de rumo
            const act = robotKeys.filter(id => isActive(timeline.robots[id], f));
            const near = [];
            for (let i = 0; i < act.length; i++) {
                for (let j = i + 1; j < act.length; j++) {
                    const a = stateAt(timeline.robots[act[i]], f), b = stateAt(timeline.robots[act[j]], f);
                    const d = Math.sqrt((a.N - b.N) * (a.N - b.N) + (a.E - b.E) * (a.E - b.E) + (a.D - b.D) * (a.D - b.D));
                    if (d < timeline.proximity_warning_m) near.push({ a: act[i], b: act[j], d: d });
                }
            }
            const col = (collision !== null && currentTime >= collision.t) ? collision : null;
            if (near.length === 0 && col === null) { box.style.display = 'none'; return; }
            near.sort((p, q) => p.d - q.d);
            const danger = col !== null || near[0].d < timeline.collision_threshold_m;
            box.className = danger ? 'danger' : 'warning';
            const outros = col === null ? near : near.filter(p => !(p.a === col.a && p.b === col.b));
            box.innerHTML = '<div class="title">&#9888; ' + (col !== null ? 'COLISÃO' : (danger ? 'PERIGO DE COLISÃO' : 'ATENÇÃO: ROBOTS PRÓXIMOS')) + '</div>' +
                (col !== null ? '<div>' + collisionText() + (freezeOn ? ' · robots parados' : '') + '</div>' : '') +
                outros.map(p => '<div>' + p.a + ' e ' + p.b + ': <b>' + p.d.toFixed(1) + ' m</b></div>').join('');
            box.style.display = 'block';
        }

        // Método: despachar para a função da vista atual.
        // Obtemos: um único ponto de entrada para redesenhar tudo quando o tempo, o robô ou a vista mudam.
        function updateFrame(f) {
            updateProximityBanner(f);
            if (currentView === 'global') updateScene('plot-global', f, false);
            else if (currentView === 'pov') updateScene('plot-pov', f, true);
            else if (currentView === 'fp') { updateScene('plot-fp', f, false, true); updateFirstPersonHud(f); }
            else if (currentView === 'cockpit') updateCockpitView(f);
            else if (currentView === 'prox') updateProxView(f);
            else if (currentView === 'report') updateReportView(f);
        }

        // ---------- Marcas de proximidade na barra do tempo ----------
        // Cada marca é um período em que dois robôs ativos estão a menos de proximity_warning_m (amarelo) ou a menos de
        // collision_threshold_m (vermelho). Ao clicar numa marca, o tempo salta para 10 s antes do início do período.
        // Obtemos: salta para um instante absoluto, limitado ao início e ao fim da missão.
        function jumpToTime(t) {
            const target = Math.max(t0_base, Math.min(t_max, t));
            document.getElementById('time-slider').value = target;
            onSliderChange(target);
        }

        // Método: percorrer a grelha de tempo para cada par e cada limiar; um episódio começa quando a
        // distância passa abaixo do limiar e acaba quando sobe. Guarda-se a distância mínima e o instante.
        // Obtemos: os blocos amarelos e vermelhos por baixo da barra do tempo (clique salta para 10 s antes).
        function buildProximityMarks() {
            const box = document.getElementById('prox-marks');
            if (!box) return;
            box.innerHTML = '';
            const span = t_max - t0_base;
            if (!(span > 0)) return;
            const warnM = timeline.proximity_warning_m || 30.0;
            const dangerM = timeline.collision_threshold_m || 10.0;
            const marks = [];
            for (let i = 0; i < robotKeys.length; i++) {
                for (let j = i + 1; j < robotKeys.length; j++) {
                    const ra = timeline.robots[robotKeys[i]], rb = timeline.robots[robotKeys[j]];
                    [[warnM, 'warning'], [dangerM, 'danger']].forEach(([limit, kind]) => {
                        let start = -1, minD = Infinity, kMin = 0;
                        for (let k = 0; k <= numSteps; k++) {
                            let d = Infinity;
                            if (k < numSteps && isActive(ra, k) && isActive(rb, k)) {
                                const dn = ra.N[k] - rb.N[k], de = ra.E[k] - rb.E[k], dd = ra.D[k] - rb.D[k];
                                d = Math.sqrt(dn * dn + de * de + dd * dd);
                            }
                            if (d < limit) {
                                if (start < 0) { start = k; minD = Infinity; }
                                if (d < minD) { minD = d; kMin = k; }
                            } else if (start >= 0) {
                                marks.push({ kind: kind, a: robotKeys[i], b: robotKeys[j], limit: limit, k0: start, k1: k - 1, minD: minD, kMin: kMin });
                                start = -1;
                            }
                        }
                    });
                }
            }
            marks.sort((p, q) => (p.kind === q.kind ? p.k0 - q.k0 : (p.kind === 'warning' ? -1 : 1)));   // o vermelho fica por cima
            marks.forEach(m => {
                const tStart = t0_base + m.k0 * dt_step, tEnd = t0_base + m.k1 * dt_step, tMin = t0_base + m.kMin * dt_step;
                const el = document.createElement('div');
                el.className = 'prox-mark ' + m.kind;
                el.style.setProperty('--c', String(((tStart + tEnd) / 2 - t0_base) / span));
                el.style.setProperty('--w', String((tEnd - tStart) / span));
                el.dataset.kind = m.kind;
                el.dataset.pair = m.a + ' e ' + m.b;
                el.dataset.t0 = String(tStart);
                el.dataset.t1 = String(tEnd);
                el.dataset.minDist = m.minD.toFixed(1);
                // (sem barras invertidas: este JavaScript vive dentro de um texto Python; NL = quebra de linha)
                el.title = [
                    (m.kind === 'danger' ? 'PERIGO DE COLISÃO' : 'ATENÇÃO: robots próximos') + ' (menos de ' + m.limit.toFixed(0) + ' m)',
                    m.a + ' e ' + m.b,
                    'De ' + fmtElapsed(tStart - t0_base) + ' a ' + fmtElapsed(tEnd - t0_base),
                    'Distância mínima: ' + m.minD.toFixed(1) + ' m (aos ' + fmtElapsed(tMin - t0_base) + ')',
                    'Clique para ir para 10 s antes'
                ].join(String.fromCharCode(10));
                el.addEventListener('click', () => jumpToTime(tStart - 10.0));
                box.appendChild(el);
            });
        }
        buildProximityMarks();
        detectCollision();

        // ---------- controlos ----------
        // Obtemos: troca de painel (separador), marca o botão ativo e inicia o radar quando a vista é a 3ª Pessoa.
        function setView(view) {
            currentView = view;
            document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-buttons button').forEach(b => {
                if (b.id.startsWith('btn-global') || b.id.startsWith('btn-pov') || b.id.startsWith('btn-cockpit') || b.id.startsWith('btn-prox') || b.id.startsWith('btn-report') || b.id.startsWith('btn-fp'))
                    b.classList.remove('active');
            });
            const p = document.getElementById('panel-' + view);
            if (p) p.classList.add('active');
            const btn = document.getElementById('btn-' + view);
            if (btn) btn.classList.add('active');
            const targetPlot = document.getElementById('plot-' + view);
            if (targetPlot && window.Plotly) Plotly.Plots.resize(targetPlot);
            updateFrame(curF());
            if (view === 'pov') startThirdPersonOverlay();
        }

        // Obtemos: muda o robô selecionado, liga o seguimento e atualiza a cor e o tamanho dos marcadores.
        function setRobot(robotId) {
            currentRobot = robotId;
            if (!followOn) setFollow(true);      // ao escolher um robô, a câmara da Vista Global vai ter com ele e segue-o
            robotKeys.forEach((id, i) => {
                const b = document.getElementById('btn-robot-' + i);
                if (b) b.classList.toggle('active', id === robotId);
            });
            const gd = document.getElementById('plot-global');
            if (gd && gd.data) {
                robotKeys.forEach((id, i) => {
                    const sel = (id === currentRobot);
                    Plotly.restyle(gd, { 'marker.size': [sel ? 8 : 6], 'marker.line.color': [sel ? '#ffffff' : timeline.robots[id].color] }, [3 + 3 * i]);
                    Plotly.restyle(gd, { color: [hullColor(id)] }, [4 + 3 * i]);
                });
            }
            updateFrame(curF());
        }

        // Obtemos: define a velocidade de reprodução (Ultra percorre a missão toda em cerca de 2 s).
        function setSpeed(val) {
            speed = val;
            const map = { 'spd-slow': 0.25, 'spd-norm': 1.0, 'spd-fast': 5.0, 'spd-10': 10.0, 'spd-20': 20.0, 'spd-50': 50.0 };
            Object.keys(map).forEach(k => document.getElementById(k).classList.toggle('active', val === map[k]));
            document.getElementById('spd-ultra').classList.toggle('active', val >= 500.0);
        }

        // Obtemos: arranca ou para o ciclo de reprodução.
        function togglePlay() {
            isPlaying = !isPlaying;
            document.getElementById('btn-play').innerHTML = isPlaying ? '&#9208; Pausa' : '&#9654; Play';
            if (isPlaying) {
                lastFrameTimestamp = null;
                animFrameId = requestAnimationFrame(runPlaybackLoop);
            } else if (animFrameId) {
                cancelAnimationFrame(animFrameId);
                animFrameId = null;
            }
        }

        // Obtemos: volta ao instante inicial (tecla R).
        function resetTrajectory() {
            currentTime = t0_base;
            document.getElementById('time-slider').value = t0_base;
            onSliderChange(t0_base);
        }

        // Obtemos: ao arrastar a barra do tempo, atualiza o relógio e redesenha.
        function onSliderChange(val) {
            currentTime = parseFloat(val);
            document.getElementById('time-display').innerText = fmtElapsed(currentTime - t0_base);
            updateFrame(curF());
        }

        // Método: requestAnimationFrame; o tempo avança dtSec vezes a velocidade escolhida (dtSec limitado a 0.25 s
        // para não saltar depois de uma pausa) e os gráficos só se refazem cerca de 30 vezes por segundo.
        // Obtemos: reprodução fluida mesmo com 50 vezes a velocidade.
        function runPlaybackLoop(timestamp) {
            if (!isPlaying) { lastFrameTimestamp = null; return; }
            if (!lastFrameTimestamp) lastFrameTimestamp = timestamp;
            const dtSec = Math.min(0.25, (timestamp - lastFrameTimestamp) / 1000.0);
            lastFrameTimestamp = timestamp;

            let effectiveSpeed = speed;
            if (speed >= 500.0) effectiveSpeed = (t_max - t0_base) / 2.0;   // missão completa em ~2 segundos

            const prevTime = currentTime;
            currentTime += dtSec * effectiveSpeed;
            if (currentTime > t_max) currentTime = t0_base;
            if (collision !== null && prevTime < collision.t && currentTime >= collision.t) {
                // o Play pára no instante da colisão; ao carregar em Play segue dali
                currentTime = collision.t;
                document.getElementById('time-slider').value = currentTime;
                document.getElementById('time-display').innerText = fmtElapsed(currentTime - t0_base);
                updateFrame(curF());
                togglePlay();
                return;
            }

            document.getElementById('time-slider').value = currentTime;
            document.getElementById('time-display').innerText = fmtElapsed(currentTime - t0_base);

            if (timestamp - lastRender >= 33) {   // ~30 atualizações/s dos gráficos
                lastRender = timestamp;
                updateFrame(curF());
            }
            animFrameId = requestAnimationFrame(runPlaybackLoop);
        }

        // Atalhos de teclado
        window.addEventListener('keydown', (e) => {
            if (e.target && e.target.tagName === 'INPUT') return;
            if (e.key >= '1' && e.key <= '9' && robotKeys[parseInt(e.key) - 1]) setRobot(robotKeys[parseInt(e.key) - 1]);
            if (e.key.toLowerCase() === 'v') {
                if (currentView === 'global') setView('pov');
                else if (currentView === 'pov') setView('fp');
                else if (currentView === 'fp') setView('cockpit');
                else if (currentView === 'cockpit') setView('prox');
                else if (currentView === 'prox') setView('report');
                else setView('global');
            }
            if (e.key.toLowerCase() === 'r') resetTrajectory();
            if (e.key.toLowerCase() === 'f') toggleFollow();
            if (e.key.toLowerCase() === 'g') overviewCamera();
            if (e.key === '+' || e.key === '=') tpZoom(1.0 / 1.2);
            if (e.key === '-' || e.key === '_') tpZoom(1.2);
            if (e.key.toLowerCase() === 'p') togglePredict();
            if (e.key.toLowerCase() === 'c') toggleFreeze();
            if (e.key.toLowerCase() === 'h') toggleHud();
            if (e.key === ' ') { e.preventDefault(); togglePlay(); }
        });

        // Inicialização com parâmetros de URL: ?t=segundos_desde_o_inicio&view=pov&robot=2&play=1 (ou #t=...)
        // Obtemos: abrir já num instante, vista e robô escolhidos pelo endereço (útil para tirar capturas de ecrã).
        function initFromUrl() {
            const params = new URLSearchParams(window.location.search);
            const hash = window.location.hash;
            let rel = 0;
            if (params.has('t')) rel = parseFloat(params.get('t'));
            else if (hash && hash.startsWith('#t=')) rel = parseFloat(hash.substring(3));
            if (params.has('robot')) setRobot(robotKeys[parseInt(params.get('robot')) - 1] || currentRobot);
            if (params.has('view')) setView(params.get('view'));
            if (params.has('speed')) setSpeed(parseFloat(params.get('speed')));
            currentTime = t0_base + rel;
            document.getElementById('time-slider').value = currentTime;
            onSliderChange(currentTime);
            if ((params.get('play') === '1' || hash === '#play') && !isPlaying) togglePlay();
        }
        let initDone = false;
        function initOnce() {
            if (initDone) return;
            initDone = true;
            installThirdPersonMouse();
            document.getElementById('tp-range-down').addEventListener('click', () => changeRadarRange(-1));
            document.getElementById('tp-range-up').addEventListener('click', () => changeRadarRange(+1));
            document.getElementById('tp-radar-range').innerText = radarRange + ' m';
            initFromUrl();
        }
        window.addEventListener('load', () => setTimeout(initOnce, 300));
    </script>
</body>
</html>"""

        html = template.replace("__DIV_GLOBAL__", div_global) \
                       .replace("__DIV_POV__", div_pov) \
                       .replace("__DIV_FP__", div_fp) \
                       .replace("__DIV_COCKPIT__", div_cockpit) \
                       .replace("__DIV_PROX__", div_prox) \
                       .replace("__DIV_REPORT__", div_report) \
                       .replace("__ORIGIN_TXT__", origin_txt) \
                       .replace("__DANGER_TXT__", f"{danger_m:g}") \
                       .replace("__MIN_T__", str(min_t)) \
                       .replace("__MAX_T__", str(max_t)) \
                       .replace("__T_EVAL__", str(t_eval)) \
                       .replace("__ACTIVE_ROBOT__", self.active_robot) \
                       .replace("__DANGER_M__", f"{danger_m:g}") \
                       .replace("__TIMELINE_JSON__", timeline_json)

        return html

    def export_standalone_html(self, output_file: str) -> str:
        """Gera e guarda o ficheiro HTML autónomo que funciona 100% offline."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        html_content = self.build_dashboard_html()
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_file

    def launch_server(self, open_browser: bool = True):
        """Inicia o servidor HTTP multithreaded local."""
        dashboard_ref = self
        html_cache = self.build_dashboard_html()

        class DashboardHTTPHandler(SimpleHTTPRequestHandler):
            """Servidor mínimo: responde à página principal com o HTML já construído (html_cache)."""
            def do_GET(self):
                if self.path == "/" or self.path.startswith("/?"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(html_cache.encode("utf-8"))
                else:
                    super().do_GET()

            def log_message(self, format, *args):
                pass  # Silencia logs de requisições rotineiras

        server = ThreadingHTTPServer((self.host, self.port), DashboardHTTPHandler)
        url = f"http://{self.host}:{self.port}"
        print(f"\n[SERVIDOR DASHBOARD WEB] Ativo em: {url}")
        print("  - Teclas 1, 2, 3: Alternar robô ativo")
        print("  - Tecla V: Alternar entre as Vistas (Global 3D / 3ª Pessoa / Cockpit / CPA)")
        print("  - Tecla R: Reiniciar replay das trajetórias")
        print("  - Barra de espaço: Play / Pausa")

        if open_browser:
            threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nA encerrar servidor web...")
            server.server_close()