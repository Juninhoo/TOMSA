"""Visualizador 3D (Plotly) do ponto de vista de cada robô.

Mostra os outros robôs e as suas trajetórias vistos pelo robô escolhido
(x frente, y direita, z cima), com barra do tempo e teclas 1, 2, 3 para mudar de robô.
Gera um ficheiro HTML autónomo.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

from src.config_loader import load_config
from src.data_loader import MissionDataset, load_mission_dataset


def build_body_view_figure(
    dataset: MissionDataset,
    stride_traj: int = 5,
    num_slider_steps: int = 60
) -> Tuple[go.Figure, str]:
    """Esta função constrói a figura 3D com os botões de mudança de robô e a barra do tempo."""
    # Robôs e grelha de tempo comum, com posições e orientações interpoladas
    robots = sorted(list(dataset.trajectories.keys()))
    if not dataset.has_overlap:
        raise ValueError("O dataset não possui janela temporal sobreposta comum.")

    t_start = dataset.common_start
    t_end = dataset.common_end
    t_common = np.arange(t_start, t_end + 1e-5, 1.0)
    n_steps = len(t_common)

    data_sync = {}
    for r_id in robots:
        traj = dataset.trajectories[r_id]
        ned_interp = np.empty((n_steps, 3))
        for dim in range(3):
            ned_interp[:, dim] = np.interp(t_common, traj.timestamps, traj.pos_ned[dim])

        rots = dataset._slerps[r_id](t_common)
        data_sync[r_id] = {
            "pos": ned_interp,
            "rot": rots.as_matrix(),
            "color": traj.color,
            "label": traj.label
        }

    def _get_body_coords(obs_id: str, tgt_id: str, step: int) -> np.ndarray:
        """Posição do robô tgt_id vista pelo robô obs_id no passo de tempo step."""
        p_obs = data_sync[obs_id]["pos"][step]
        r_b = data_sync[obs_id]["rot"][step]
        p_tgt = data_sync[tgt_id]["pos"][step]
        return r_b.T @ (p_tgt - p_obs)

    def _get_body_trajectory(obs_id: str, tgt_id: str, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trajetória completa do robô tgt_id vista pelo robô obs_id (z para cima)."""
        p_obs = data_sync[obs_id]["pos"][step]
        r_b = data_sync[obs_id]["rot"][step]
        pos_decimated = data_sync[tgt_id]["pos"][::stride_traj]
        pts = (r_b.T @ (pos_decimated - p_obs).T).T
        return pts[:, 0], pts[:, 1], -pts[:, 2]

    def _get_view_payload(obs_id: str, step: int = 0) -> Dict[str, list]:
        """Coordenadas de todos os traços para o observador obs_id no passo step (usado pelos botões e pela barra do tempo)."""
        targets = [r for r in robots if r != obs_id]
        x_list = [[0]]
        y_list = [[0]]
        z_list = [[0]]
        text_list = [[f"Observador ({obs_id})"]]

        for tgt in targets:
            c = _get_body_coords(obs_id, tgt, step)
            x_list.append([float(c[0])])
            y_list.append([float(c[1])])
            z_list.append([float(-c[2])])
            text_list.append([f"{tgt} ({data_sync[tgt]['label']})"])

        x_list.append([0, 30])
        y_list.append([0, 0])
        z_list.append([0, 0])
        text_list.append(None)

        for tgt in targets:
            xt, yt, zt = _get_body_trajectory(obs_id, tgt, step)
            x_list.append(xt.tolist())
            y_list.append(yt.tolist())
            z_list.append(zt.tolist())
            text_list.append(None)

        return {"x": x_list, "y": y_list, "z": z_list, "text": text_list}

    default_obs = robots[0]
    fig = go.Figure()

    # Traços: observador, outros robôs, seta da proa e trajetórias relativas
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=7, color="#E9C46A", symbol="diamond"),
        text=[f"Observador ({default_obs})"],
        textposition="top center",
        name=f"Observador ({default_obs})"
    ))

    targets_default = [r for r in robots if r != default_obs]
    d0 = _get_view_payload(default_obs, 0)

    for idx, tgt in enumerate(targets_default, start=1):
        fig.add_trace(go.Scatter3d(
            x=d0["x"][idx], y=d0["y"][idx], z=d0["z"][idx],
            mode="markers+text",
            marker=dict(size=9, color=data_sync[tgt]["color"]),
            text=[f"{tgt}"],
            textposition="top center",
            name=f"Posição {tgt}"
        ))

    fig.add_trace(go.Scatter3d(
        x=[0, 30], y=[0, 0], z=[0, 0],
        mode="lines+text",
        line=dict(color="#FFE57F", width=6),
        text=[None, "Proa (+X)"],
        textposition="top right",
        name="Vetor de Proa (Heading)"
    ))

    start_tr_idx = 1 + len(targets_default) + 1
    for idx, tgt in enumerate(targets_default):
        tr_trace_idx = start_tr_idx + idx
        fig.add_trace(go.Scatter3d(
            x=d0["x"][tr_trace_idx], y=d0["y"][tr_trace_idx], z=d0["z"][tr_trace_idx],
            mode="lines",
            line=dict(color=data_sync[tgt]["color"], width=3),
            opacity=0.7,
            name=f"Trajetória Relativa {tgt}"
        ))

    # Botões para escolher o robô observador e barra do tempo
    buttons = []
    for k, obs in enumerate(robots):
        vdata = _get_view_payload(obs, 0)
        key_num = k + 1
        buttons.append(dict(
            label=f"[{key_num}] Visão {obs}",
            method="update",
            args=[
                vdata,
                {
                    "title.text": f"TOMSA 2026 - Vista em 1ª Pessoa: {obs} (Observador Body Frame)",
                    "scene.camera": {
                        "eye": {"x": -0.85, "y": 0.0, "z": 0.35},
                        "center": {"x": 1.2, "y": 0.0, "z": 0.0},
                        "up": {"x": 0.0, "y": 0.0, "z": 1.0}
                    }
                }
            ]
        ))

    updatemenus = [dict(
        type="buttons",
        direction="left",
        x=0.01, y=1.12,
        pad={"r": 8, "t": 8},
        showactive=True,
        buttons=buttons
    )]

    slider_steps = []
    step_stride = max(1, n_steps // num_slider_steps)
    for step_k in range(0, n_steps, step_stride):
        sdata = _get_view_payload(default_obs, step_k)
        t_sec = int(t_common[step_k] - t_common[0])
        slider_steps.append({
            "method": "update",
            "label": f"{t_sec}s",
            "args": [
                {
                    "x": sdata["x"],
                    "y": sdata["y"],
                    "z": sdata["z"]
                }
            ]
        })

    sliders = [dict(
        active=0,
        steps=slider_steps,
        x=0.05, y=-0.05,
        len=0.9,
        currentvalue={"prefix": "Tempo de Replay da Missão: "}
    )]

    # Teclas 1, 2, 3 para mudar de observador (JavaScript acrescentado à página)
    post_script = """
    document.addEventListener('keydown', function(event) {
        var buttons = document.querySelectorAll('.updatemenu-button');
        if (!buttons || buttons.length < 3) return;
        if (event.key === '1') {
            buttons[0].click();
        } else if (event.key === '2') {
            buttons[1].click();
        } else if (event.key === '3') {
            buttons[2].click();
        }
    });
    """

    fig.update_layout(
        title=f"TOMSA 2026 - Vista em 1ª Pessoa: {default_obs} (Observador Body Frame)",
        template="plotly_dark",
        updatemenus=updatemenus,
        sliders=sliders,
        scene=dict(
            xaxis=dict(title="Frente / Proa (m) [+X Body]", gridcolor="#444"),
            yaxis=dict(title="Estibordo / Direita (m) [+Y Body]", gridcolor="#444"),
            zaxis=dict(title="Vertical / Cima (m) [-Z Down]", gridcolor="#444"),
            aspectmode="data",
            camera=dict(
                eye=dict(x=-0.85, y=0.0, z=0.35),
                center=dict(x=1.2, y=0.0, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0)
            )
        ),
        margin=dict(l=10, r=10, b=30, t=70)
    )

    return fig, post_script


def launch_interactive_3d(
    config_path: str = "config/config.yaml",
    export_html_path: str = "output/mission_body_view_3d.html",
    open_browser: bool = True
) -> go.Figure:
    """Carrega a missão, gera a figura, grava o HTML e (opcionalmente) abre-o no navegador."""
    dataset, _ = load_mission_dataset(config_path)
    fig, post_script = build_body_view_figure(dataset, stride_traj=5, num_slider_steps=60)

    if export_html_path:
        out_path = Path(export_html_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path), post_script=post_script, include_plotlyjs="cdn")
        print(f"Visualizador 3D exportado com sucesso para: {out_path.resolve()}")

    if open_browser:
        fig.show(post_script=post_script)

    return fig
