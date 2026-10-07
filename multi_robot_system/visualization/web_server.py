"""Servidor Web Multithreaded e Dashboard Interativo em Plotly.

Implementa a interface gráfica central do sistema com:
1. Comutação instantânea entre as 3 Vistas: Global 3D, POV 1ª Pessoa e Cockpit de Instrumentos.
2. Controlo temporal com slider, Play/Pausa e vários níveis de velocidade (até Ultra Rápido em ~2s).
3. Teclas de atalho integradas:
   - '1', '2', '3': Seleciona o robô ativo.
   - 'v': Alterna ciclicamente entre as 3 vistas.
   - 'r': Reinicia a reprodução das trajetórias (reset ao início).
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
from multi_robot_system.visualization.plotly_views import (
    create_global_3d_figure,
    create_pov_camera_figure,
    create_proximity_timeline_figure,
    get_scene_geometry
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
        states = self.dataset.get_state_at_time(t_eval)
        cpa_results = self.dataset.evaluate_cpa_at_time(t_eval, collision_threshold_m=10.0)

        # Gera as 4 figuras Plotly com IDs estáticos garantidos
        fig_global = create_global_3d_figure(self.dataset, t_eval, self.active_robot)
        fig_pov = create_pov_camera_figure(self.dataset, t_eval, self.active_robot)
        active_st = states.get(self.active_robot, list(states.values())[0])
        fig_cockpit = create_cockpit_figure(active_st, cpa_results, t_eval - self.dataset.t0_mission)
        fig_prox = create_proximity_timeline_figure(self.dataset, t_eval, collision_threshold_m=10.0)

        div_global = pio.to_html(fig_global, div_id="plot-global", full_html=False, include_plotlyjs="cdn")
        div_pov = pio.to_html(fig_pov, div_id="plot-pov", full_html=False, include_plotlyjs=False)
        div_cockpit = pio.to_html(fig_cockpit, div_id="plot-cockpit", full_html=False, include_plotlyjs=False)
        div_prox = pio.to_html(fig_prox, div_id="plot-prox", full_html=False, include_plotlyjs=False)

        min_t = float(self.t_start)
        max_t = float(self.t_end)

        # Amostragem da linha temporal COMPLETA da missão para animação no cliente (sem latência de rede)
        dt_sample = 2.0  # passo temporal [s]; o JS interpola linearmente entre amostras
        t_arr = np.arange(min_t, max_t + 1e-6, dt_sample)
        sorted_robots = sorted(self.dataset.trajectories.keys())

        def _unwrap_deg(a):
            return np.degrees(np.unwrap(np.radians(a)))

        timeline_dict = {
            "dt_step": dt_sample,
            "t0_base": min_t,
            "t_max": max_t,
            "num_steps": len(t_arr),
            "scene": get_scene_geometry(self.dataset),
            "robots": {}
        }

        for r_id in sorted_robots:
            traj = self.dataset.trajectories[r_id]
            ts = traj.timestamps
            f = lambda y: np.interp(t_arr, ts, y)  # fora do intervalo do robô mantém a 1ª/última posição
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
            }
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
        }
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
        input[type=range] {
            flex: 1;
            accent-color: #00b4d8;
            height: 6px;
            cursor: pointer;
        }
        .time-label { font-family: monospace; font-size: 14px; color: #48cae4; min-width: 90px; font-weight: bold; }
        
        .main-container {
            padding: 16px 24px;
            height: calc(100vh - 125px);
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
        kbd {
            background: #374151;
            padding: 2px 6px;
            border-radius: 4px;
            color: #f9fafb;
            font-size: 11px;
            font-family: monospace;
        }
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
            <button id="btn-pov" onclick="setView('pov')">2. Vista POV (1ª Pessoa)</button>
            <button id="btn-cockpit" onclick="setView('cockpit')">3. Cockpit / Controlo</button>
            <button id="btn-prox" onclick="setView('prox')">4. Distâncias CPA 2D</button>
            
            <span style="font-size: 12px; color: #9ca3af; margin-left: 12px; margin-right: 4px;">ROBÔ:</span>
            <button id="btn-r1" class="active" onclick="setRobot('Robot 1')">Robot 1</button>
            <button id="btn-r4" onclick="setRobot('Robot 4')">Robot 4</button>
            <button id="btn-r7" onclick="setRobot('Robot 7')">Robot 7</button>
        </div>
    </header>

    <div class="controls-bar">
        <button id="btn-play" onclick="togglePlay()">&#9654; Play</button>
        <button id="btn-reset" class="danger" onclick="resetTrajectory()">&#8634; Reiniciar (R)</button>
        
        <div class="slider-box">
            <span style="font-size: 12px; color: #9ca3af;">Tempo:</span>
            <input type="range" id="time-slider" min="__MIN_T__" max="__MAX_T__" step="0.5" value="__T_EVAL__" oninput="onSliderChange(this.value)">
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
        <div id="panel-global" class="view-panel active">
            __DIV_GLOBAL__
        </div>
        <div id="panel-pov" class="view-panel">
            __DIV_POV__
        </div>
        <div id="panel-cockpit" class="view-panel">
            __DIV_COCKPIT__
        </div>
        <div id="panel-prox" class="view-panel">
            __DIV_PROX__
        </div>
    </div>

    <div class="help-banner">
        <div>
            <b>Atalhos de Teclado:</b>
            <kbd>1</kbd>, <kbd>2</kbd>, <kbd>3</kbd>: Selecionar Submarino |
            <kbd>V</kbd>: Alternar Vista |
            <kbd>R</kbd>: Reiniciar Trajetória |
            <kbd>Espaço</kbd>: Play / Pausa
        </div>
        <div>
            Limiar de Alarme de Colisão: <b>&lt; 10 metros</b> | Filtro: <b>RTS Smoother Ativo</b>
        </div>
    </div>

    <script>
        const timeline = __TIMELINE_JSON__;
        const dt_step = timeline.dt_step;
        const t0_base = timeline.t0_base;
        const t_max = timeline.t_max;
        const numSteps = timeline.num_steps;
        const robotKeys = Object.keys(timeline.robots).sort();
        const DEG = Math.PI / 180.0;
        const FOV_F = 1.0 / Math.tan(35.0 * DEG);   // distância focal para FOV horizontal de 70 graus (u em [-1, 1])
        const HALF_V = 0.5625;                       // meia-altura do ecrã (16:9)

        let currentView = 'global';
        let currentRobot = '__ACTIVE_ROBOT__';
        let isPlaying = false;
        let speed = 1.0;
        let currentTime = t0_base;
        let animFrameId = null;
        let lastFrameTimestamp = null;
        let lastRender = 0;
        let lastActiveKey = '';

        // ---------- utilitários de interpolação ----------
        function curF() {
            return Math.min(numSteps - 1, Math.max(0, (currentTime - t0_base) / dt_step));
        }
        function at(arr, f) {
            const i = Math.floor(f);
            if (i >= arr.length - 1) return arr[arr.length - 1];
            const w = f - i;
            return arr[i] * (1 - w) + arr[i + 1] * w;
        }
        function isActive(r, f) { return f >= r.k0 && f <= r.k1; }
        function stateAt(r, f) {
            return {
                E: at(r.E, f), N: at(r.N, f), D: at(r.D, f),
                yaw: at(r.yaw, f), pitch: at(r.pitch, f), roll: at(r.roll, f),
                spd: at(r.spd, f), turn: at(r.turn, f)
            };
        }
        // Vetor no referencial do corpo (x frente, y direita, z baixo): R^T (ZYX) * (dN, dE, dD)
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
        function fmtElapsed(sec) {
            const m = Math.floor(sec / 60);
            return m + 'm ' + (sec - 60 * m).toFixed(1) + 's';
        }

        function computeSubmarineWireframe(e, n, d, yawDeg, pitchDeg, sc) {
            const yaw = yawDeg * DEG, pitch = pitchDeg * DEG;
            const k = sc || 1.0;
            const length = 8.0 * k, wingspan = 3.5 * k, height = 2.5 * k;
            const fe = Math.sin(yaw) * Math.cos(pitch);
            const fn = Math.cos(yaw) * Math.cos(pitch);
            const fd = -Math.sin(pitch);
            const re = Math.cos(yaw), rn = -Math.sin(yaw), rd = 0.0;
            const ue = Math.sin(yaw) * Math.sin(pitch);
            const un = Math.cos(yaw) * Math.sin(pitch);
            const ud = -Math.cos(pitch);
            const nose_e = e + length * fe, nose_n = n + length * fn, nose_d = d + length * fd;
            const stern_e = e - 0.5 * length * fe, stern_n = n - 0.5 * length * fn, stern_d = d - 0.5 * length * fd;
            const port_e = stern_e - 0.5 * wingspan * re, port_n = stern_n - 0.5 * wingspan * rn, port_d = stern_d - 0.5 * wingspan * rd;
            const stbd_e = stern_e + 0.5 * wingspan * re, stbd_n = stern_n + 0.5 * wingspan * rn, stbd_d = stern_d + 0.5 * wingspan * rd;
            const rud_e = stern_e + height * ue, rud_n = stern_n + height * un, rud_d = stern_d + height * ud;
            return {
                x: [nose_e, stern_e, port_e, stbd_e, stern_e, rud_e, nose_e],
                y: [nose_n, stern_n, port_n, stbd_n, stern_n, rud_n, nose_n],
                z: [nose_d, stern_d, port_d, stbd_d, stern_d, rud_d, nose_d]
            };
        }

        // ---------- Cena 3D (Vista Global e Vista POV partilham o mesmo código) ----------
        const lastKeys = {};
        let frameCount = 0;
        function updateScene(gdId, f, pov) {
            const gd = document.getElementById(gdId);
            if (!gd || !gd.data) return;
            frameCount++;
            const hull = pov ? 3.0 : 1.0;           // na 1ª pessoa os cascos dos outros robôs são desenhados maiores
            const idx = [], X = [], Y = [], Z = [];
            const i0 = Math.floor(f);
            const flags = [];
            const mkIdx = [], mkOpacity = [], mkText = [], wfOpacity = [], wfIdx = [];

            robotKeys.forEach((id, i) => {
                const r = timeline.robots[id];
                const s = stateAt(r, f);
                const act = isActive(r, f);
                const self = pov && id === currentRobot;   // o próprio casco não se vê por dentro
                flags.push(act ? 1 : 0);

                // Rasto progressivo: só a parte já percorrida pelo robô
                const sIdx = Math.max(0, Math.floor(r.k0));
                const eIdx = Math.max(sIdx, Math.min(i0, Math.ceil(r.k1)));
                const tx = r.E.slice(sIdx, eIdx + 1), ty = r.N.slice(sIdx, eIdx + 1), tz = r.D.slice(sIdx, eIdx + 1);
                if (act) { tx.push(s.E); ty.push(s.N); tz.push(s.D); }
                idx.push(2 + 3 * i); X.push(tx); Y.push(ty); Z.push(tz);

                idx.push(3 + 3 * i); X.push([s.E]); Y.push([s.N]); Z.push([s.D]);
                const g = computeSubmarineWireframe(s.E, s.N, s.D, s.yaw, s.pitch, hull);
                idx.push(4 + 3 * i); X.push(g.x); Y.push(g.y); Z.push(g.z);

                mkIdx.push(3 + 3 * i); mkOpacity.push(self ? 0.0 : (act ? 1.0 : 0.25));
                mkText.push([self ? '' : (act ? ('<b>' + id + '</b><br>' + s.spd.toFixed(1) + ' m/s') : ('<b>' + id + '</b><br>inativo'))]);
                wfIdx.push(4 + 3 * i); wfOpacity.push(self ? 0.0 : (act ? 1.0 : 0.25));
            });
            Plotly.restyle(gd, { x: X, y: Y, z: Z }, idx);

            const key = flags.join('') + (pov ? currentRobot : '');
            if (key !== lastKeys[gdId] || (frameCount % 8) === 0) {
                lastKeys[gdId] = key;
                Plotly.restyle(gd, { 'marker.opacity': mkOpacity, text: mkText }, mkIdx);
                Plotly.restyle(gd, { opacity: wfOpacity }, wfIdx);
            }

            if (pov) setFirstPersonCamera(gd, f);
        }

        // Câmara em 1ª pessoa: olho no nariz do robô ativo, a olhar para a proa, com o 'up' do corpo.
        // Coordenadas da câmara do Plotly = (E - cE, N - cN, -(D - cD)) / span (eixo Z invertido).
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

        // ---------- Cockpit ----------
        function updateCockpitView(f) {
            const gd = document.getElementById('plot-cockpit');
            if (!gd || !gd.data) return;
            const actR = timeline.robots[currentRobot];
            if (!actR) return;
            const s = stateAt(actR, f);
            const active = isActive(actR, f);
            const yaw = ((s.yaw % 360.0) + 360.0) % 360.0;

            let minDist = null;
            const rs = [], thetas = [], texts = [], colors = [];
            robotKeys.forEach(id => {
                if (id === currentRobot) return;
                const r = timeline.robots[id];
                if (!isActive(r, f) || !active) return;
                const t = stateAt(r, f);
                const dN = t.N - s.N, dE = t.E - s.E, dD = t.D - s.D;
                const dist = Math.sqrt(dN * dN + dE * dE + dD * dD);
                if (minDist === null || dist < minDist) minDist = dist;
                const psi = s.yaw * DEG;
                const xb = Math.cos(psi) * dN + Math.sin(psi) * dE;
                const yb = -Math.sin(psi) * dN + Math.cos(psi) * dE;
                thetas.push(((Math.atan2(yb, xb) / DEG) % 360.0 + 360.0) % 360.0);
                rs.push(dist);
                texts.push(id + '<br>Dist: ' + dist.toFixed(1) + 'm');
                colors.push(dist < 10.0 ? '#e63946' : '#52b788');
            });

            const risk = (minDist !== null && minDist < 10.0);
            const statusColor = risk ? '#e63946' : '#2a9d8f';
            const statusMsg = risk ? 'PERIGO DE COLISÃO (< 10m)' : (minDist === null ? 'SEM OUTROS ROBÔS ATIVOS' : 'ROTA SEGURA');
            const sg = (v) => (v >= 0 ? '+' : '') + v.toFixed(1);

            Plotly.restyle(gd, {
                value: [yaw],
                'gauge.bar.color': [actR.color],
                title: [{ text: 'Roll: ' + sg(s.roll) + '° | Pitch: ' + sg(s.pitch) + '°' + (active ? '' : ' [INATIVO]'), font: { color: '#e0e1dd', size: 14 } }]
            }, [0]);
            Plotly.restyle(gd, { value: [s.D] }, [1]);
            Plotly.restyle(gd, { value: [s.spd] }, [2]);
            Plotly.restyle(gd, { r: [rs], theta: [thetas], hovertext: [texts], text: [texts.map(t => t.split('<br>')[0])], 'marker.color': [colors] }, [3]);
            Plotly.restyle(gd, { value: [s.turn] }, [4]);
            Plotly.restyle(gd, {
                value: [minDist === null ? 0 : minDist],
                'number.font.color': [statusColor],
                title: [{
                    text: '<b>' + statusMsg + '</b><br><span style="font-size:0.8em;color:#e0e1dd">Menor distância: ' + (minDist === null ? '-' : minDist.toFixed(1) + ' m') + '</span>',
                    font: { size: 16, color: statusColor }
                }]
            }, [5]);
        }

        // ---------- Gráfico 2D de distâncias: move o cursor de tempo ----------
        function updateProxView(f) {
            const gd = document.getElementById('plot-prox');
            if (!gd || !gd.layout || !gd.layout.shapes || gd.layout.shapes.length < 2) return;
            const tRel = f * dt_step;
            Plotly.relayout(gd, { 'shapes[1].x0': tRel, 'shapes[1].x1': tRel });
        }

        function updateFrame(f) {
            if (currentView === 'global') updateScene('plot-global', f, false);
            else if (currentView === 'pov') updateScene('plot-pov', f, true);
            else if (currentView === 'cockpit') updateCockpitView(f);
            else if (currentView === 'prox') updateProxView(f);
        }

        // ---------- controlos ----------
        function setView(view) {
            currentView = view;
            document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-buttons button').forEach(b => {
                if (b.id.startsWith('btn-global') || b.id.startsWith('btn-pov') || b.id.startsWith('btn-cockpit') || b.id.startsWith('btn-prox'))
                    b.classList.remove('active');
            });
            const p = document.getElementById('panel-' + view);
            if (p) p.classList.add('active');
            const btn = document.getElementById('btn-' + view);
            if (btn) btn.classList.add('active');
            const targetPlot = document.getElementById('plot-' + view);
            if (targetPlot && window.Plotly) Plotly.Plots.resize(targetPlot);
            updateFrame(curF());
        }

        function setRobot(robotId) {
            currentRobot = robotId;
            document.getElementById('btn-r1').classList.toggle('active', robotId === 'Robot 1');
            document.getElementById('btn-r4').classList.toggle('active', robotId === 'Robot 4');
            document.getElementById('btn-r7').classList.toggle('active', robotId === 'Robot 7');
            const gd = document.getElementById('plot-global');
            if (gd && gd.data) {
                robotKeys.forEach((id, i) => {
                    const sel = (id === currentRobot);
                    Plotly.restyle(gd, { 'marker.size': [sel ? 8 : 6], 'marker.line.color': [sel ? '#ffffff' : timeline.robots[id].color] }, [3 + 3 * i]);
                    Plotly.restyle(gd, { 'line.color': [sel ? '#ffffff' : timeline.robots[id].color] }, [4 + 3 * i]);
                });
            }
            updateFrame(curF());
        }

        function setSpeed(val) {
            speed = val;
            const map = { 'spd-slow': 0.25, 'spd-norm': 1.0, 'spd-fast': 5.0, 'spd-10': 10.0, 'spd-20': 20.0, 'spd-50': 50.0 };
            Object.keys(map).forEach(k => document.getElementById(k).classList.toggle('active', val === map[k]));
            document.getElementById('spd-ultra').classList.toggle('active', val >= 500.0);
        }

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

        function resetTrajectory() {
            currentTime = t0_base;
            document.getElementById('time-slider').value = t0_base;
            onSliderChange(t0_base);
        }

        function onSliderChange(val) {
            currentTime = parseFloat(val);
            document.getElementById('time-display').innerText = fmtElapsed(currentTime - t0_base);
            updateFrame(curF());
        }

        function runPlaybackLoop(timestamp) {
            if (!isPlaying) { lastFrameTimestamp = null; return; }
            if (!lastFrameTimestamp) lastFrameTimestamp = timestamp;
            const dtSec = Math.min(0.25, (timestamp - lastFrameTimestamp) / 1000.0);
            lastFrameTimestamp = timestamp;

            let effectiveSpeed = speed;
            if (speed >= 500.0) effectiveSpeed = (t_max - t0_base) / 2.0;   // missão completa em ~2 segundos

            currentTime += dtSec * effectiveSpeed;
            if (currentTime > t_max) currentTime = t0_base;

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
            if (e.key === '1') setRobot('Robot 1');
            if (e.key === '2') setRobot('Robot 4');
            if (e.key === '3') setRobot('Robot 7');
            if (e.key.toLowerCase() === 'v') {
                if (currentView === 'global') setView('pov');
                else if (currentView === 'pov') setView('cockpit');
                else if (currentView === 'cockpit') setView('prox');
                else setView('global');
            }
            if (e.key.toLowerCase() === 'r') resetTrajectory();
            if (e.key === ' ') { e.preventDefault(); togglePlay(); }
        });

        // Inicialização com parâmetros de URL: ?t=segundos_desde_o_inicio&view=pov&robot=2&play=1 (ou #t=...)
        function initFromUrl() {
            const params = new URLSearchParams(window.location.search);
            const hash = window.location.hash;
            let rel = 0;
            if (params.has('t')) rel = parseFloat(params.get('t'));
            else if (hash && hash.startsWith('#t=')) rel = parseFloat(hash.substring(3));
            if (params.has('robot')) setRobot(['Robot 1', 'Robot 4', 'Robot 7'][parseInt(params.get('robot')) - 1] || currentRobot);
            if (params.has('view')) setView(params.get('view'));
            if (params.has('speed')) setSpeed(parseFloat(params.get('speed')));
            currentTime = t0_base + rel;
            document.getElementById('time-slider').value = currentTime;
            onSliderChange(currentTime);
            if ((params.get('play') === '1' || hash === '#play') && !isPlaying) togglePlay();
        }
        let initDone = false;
        function initOnce() { if (initDone) return; initDone = true; initFromUrl(); }
        window.addEventListener('load', () => setTimeout(initOnce, 300));
    </script>
</body>
</html>"""

        html = template.replace("__DIV_GLOBAL__", div_global) \
                       .replace("__DIV_POV__", div_pov) \
                       .replace("__DIV_COCKPIT__", div_cockpit) \
                       .replace("__DIV_PROX__", div_prox) \
                       .replace("__MIN_T__", str(min_t)) \
                       .replace("__MAX_T__", str(max_t)) \
                       .replace("__T_EVAL__", str(t_eval)) \
                       .replace("__ACTIVE_ROBOT__", self.active_robot) \
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
        print("  - Tecla V: Alternar entre as 3 Vistas (Global 3D / POV / Cockpit)")
        print("  - Tecla R: Reiniciar replay das trajetórias")
        print("  - Barra de espaço: Play / Pausa")

        if open_browser:
            threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nA encerrar servidor web...")
            server.server_close()
