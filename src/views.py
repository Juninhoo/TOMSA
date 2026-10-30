"""Figuras Matplotlib da missão.

Vista Global (todos os robôs no mesmo referencial), vista 3D, Vista Local centrada
num robô (como um radar) e gráficos de azimute, elevação e distância ao longo do tempo.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.frames import transform_to_local_robot_frame, calculate_relative_aer_body
from src.data_loader import MissionDataset, RobotTrajectory


def plot_global_view(
    dataset: MissionDataset,
    current_time: Optional[float] = None,
    frame: str = "NED",
    ax: Optional[plt.Axes] = None,
    show_uncertainty: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """Esta função desenha a Vista Global: a trajetória de todos os robôs vista de cima,
    com a posição, a direção e (opcionalmente) a elipse de incerteza de cada um no instante dado.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    else:
        fig = ax.figure

    # Origem da missão e instante a mostrar
    ax.scatter([0], [0], color="black", marker="P", s=120, zorder=6, label="Origem da Missão (0,0)")

    t = current_time if current_time is not None else dataset.common_start
    current_states = dataset.get_state_at(t)

    for r_id, traj in dataset.trajectories.items():
        if frame == "NED":
            # Eixo horizontal = Este, eixo vertical = Norte
            x_plot = traj.pos_ned[1]
            y_plot = traj.pos_ned[0]
            x_label = "Este / East [m]"
            y_label = "Norte / North [m]"
        else:
            x_plot = traj.pos_enu[0]
            y_plot = traj.pos_enu[1]
            x_label = "East [m]"
            y_label = "North [m]"

        # Trajetória completa e ponto de partida
        ax.plot(x_plot, y_plot, color=traj.color, alpha=0.45, linewidth=1.5, label=f"Trajetória {traj.label}")

        ax.scatter(x_plot[0], y_plot[0], color=traj.color, marker="o", facecolors="none",
                   edgecolors=traj.color, s=70, linewidths=1.5, zorder=4)

        # Posição atual, seta da direção (rumo medido a partir do Norte), nome e elipse de incerteza
        if r_id in current_states:
            st = current_states[r_id]
            if frame == "NED":
                cx = st["pos_ned"][1]
                cy = st["pos_ned"][0]
                heading_rad = np.radians(st["heading_deg"])
                dx = 18.0 * np.sin(heading_rad)
                dy = 18.0 * np.cos(heading_rad)
            else:
                cx = st["pos_enu"][0]
                cy = st["pos_enu"][1]
                heading_rad = np.radians(st["heading_deg"])
                dx = 18.0 * np.sin(heading_rad)
                dy = 18.0 * np.cos(heading_rad)

            ax.scatter([cx], [cy], color=traj.color, s=110, zorder=5, edgecolors="black", linewidths=1.2)
            ax.arrow(cx, cy, dx, dy, head_width=6, head_length=8, fc=traj.color, ec="black", zorder=6)
            ax.text(cx + 8, cy + 8, f"{traj.label}\n({st['heading_deg']:.1f}°)", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=traj.color, alpha=0.85), zorder=7)

            if show_uncertainty:
                sig_n = st["pos_std_ned"][0]
                sig_e = st["pos_std_ned"][1]
                ellipse = patches.Ellipse((cx, cy), width=2*sig_e, height=2*sig_n,
                                          angle=0, fc=traj.color, alpha=0.2, ec=traj.color)
                ax.add_patch(ellipse)

    ax.set_title(f"Vista Global da Missão (Aguçadoura) - Referencial {frame}\nInstante: t = {t - dataset.common_start:.1f} s",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.axis("equal")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    return fig, ax


def plot_3d_global_view(
    dataset: MissionDataset,
    current_time: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    elev: Optional[float] = 25.0,
    azim: Optional[float] = -55.0
) -> Tuple[plt.Figure, plt.Axes]:
    """Desenha a vista 3D das trajetórias, com a posição e a direção de cada robô no instante dado."""
    if ax is None:
        fig = plt.figure(figsize=(9, 8), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    t = current_time if current_time is not None else dataset.common_start
    current_states = dataset.get_state_at(t)

    # Origem, depois trajetória 3D, posição e direção de cada robô
    ax.scatter([0], [0], [0], color="black", marker="P", s=100, zorder=6, label="Origem Missão (0,0,0)")

    for r_id, traj in dataset.trajectories.items():
        east = traj.pos_ned[1]
        north = traj.pos_ned[0]
        altitude = -traj.pos_ned[2]

        ax.plot(east, north, altitude, color=traj.color, alpha=0.55, linewidth=1.6, label=traj.label)

        ax.scatter([east[0]], [north[0]], [altitude[0]], color=traj.color, marker="o", s=40, alpha=0.7)

        if r_id in current_states:
            st = current_states[r_id]
            cx = st["pos_ned"][1]
            cy = st["pos_ned"][0]
            cz = -st["pos_ned"][2]

            ax.scatter([cx], [cy], [cz], color=traj.color, s=120, edgecolors="black", linewidths=1.2, zorder=7)

            from src.frames import quaternion_to_rotation_matrix
            R_b2ned = quaternion_to_rotation_matrix(*st["quat"])
            fwd_body = np.array([1.0, 0.0, 0.0])
            fwd_ned = R_b2ned @ fwd_body
            fwd_east = fwd_ned[1] * 22.0
            fwd_north = fwd_ned[0] * 22.0
            fwd_alt = -fwd_ned[2] * 22.0

            ax.quiver(cx, cy, cz, fwd_east, fwd_north, fwd_alt, color=traj.color, arrow_length_ratio=0.3,
                      linewidth=2.0, zorder=8)

            ax.text(cx + 5, cy + 5, cz + 2, f"{traj.label}\n({st['heading_deg']:.0f}°)", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=traj.color, alpha=0.85), zorder=9)

    ax.set_title(f"Vista Espacial Tridimensional (3D)\nInstante: t = {t - dataset.common_start:.1f} s",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Este / East [m]", fontsize=9)
    ax.set_ylabel("Norte / North [m]", fontsize=9)
    ax.set_zlabel("Altitude / Cota [m]", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    if elev is not None and azim is not None:
        ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    return fig, ax


def plot_local_view(
    dataset: MissionDataset,
    selected_robot_id: str,
    current_time: Optional[float] = None,
    max_range_m: float = 250.0,
    collision_threshold_m: float = 30.0,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Esta função desenha a Vista Local: o robô selecionado no centro, com a proa para cima,
    anéis de distância, zona de perigo e os outros robôs vistos a partir dele.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    else:
        fig = ax.figure

    t = current_time if current_time is not None else dataset.common_start
    states = dataset.get_state_at(t)

    if selected_robot_id not in states:
        ax.text(0, 0, f"Robô {selected_robot_id} sem dados no instante {t:.1f}s", ha="center")
        return fig, ax

    sel_state = states[selected_robot_id]
    sel_traj = dataset.trajectories[selected_robot_id]
    p_sel = sel_state["pos_ned"]
    q_sel = sel_state["quat"]

    # Anéis de distância, zona de perigo e eixos do robô (frente para cima, direita para a direita)
    range_steps = np.arange(50.0, max_range_m + 1.0, 50.0)
    for r in range_steps:
        circle = patches.Circle((0, 0), r, fill=False, color="gray", linestyle=":", alpha=0.5)
        ax.add_patch(circle)
        ax.text(0, r + 2, f"{int(r)} m", color="gray", fontsize=7, ha="center")

    alert_circle = patches.Circle((0, 0), collision_threshold_m, fill=True, facecolor="red", alpha=0.12,
                                  linestyle="--", edgecolor="red", label=f"Alerta Proximidade ({collision_threshold_m:.0f}m)")
    ax.add_patch(alert_circle)

    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)

    ax.scatter([0], [0], color=sel_traj.color, s=140, marker="^", edgecolors="black", linewidths=1.5, zorder=6,
               label=f"{sel_traj.label} (Origem Local)")
    ax.arrow(0, 0, 0, 25.0, head_width=7, head_length=9, fc=sel_traj.color, ec="black", zorder=7)
    ax.text(0, -18, f"PROA (0°)\nHeading: {sel_state['heading_deg']:.1f}°", ha="center", fontsize=8, fontweight="bold")

    ax.text(max_range_m * 0.85, 0, "Estibordo (+y)\n[Direita]", ha="center", va="center", color="#555", fontsize=8)
    ax.text(-max_range_m * 0.85, 0, "Bombordo (-y)\n[Esquerda]", ha="center", va="center", color="#555", fontsize=8)
    ax.text(0, -max_range_m * 0.85, "Ré (-x)\n[Atrás]", ha="center", va="center", color="#555", fontsize=8)

    # Outros robôs, convertidos para as coordenadas do robô selecionado, com linha de visada e etiqueta
    for r_id, st in states.items():
        if r_id == selected_robot_id:
            continue

        p_neigh = st["pos_ned"]
        p_body = transform_to_local_robot_frame(p_neigh, p_sel, q_sel)
        x_fwd = p_body[0]
        y_stbd = p_body[1]

        bearing_rel, elev_rel, dist_rel = calculate_relative_aer_body(p_body)
        neigh_traj = dataset.trajectories[r_id]

        is_collision = dist_rel <= collision_threshold_m
        marker_color = "red" if is_collision else neigh_traj.color

        ax.scatter([y_stbd], [x_fwd], color=marker_color, s=120, edgecolors="black", linewidths=1.2, zorder=5,
                   label=f"{neigh_traj.label} (dist: {dist_rel:.1f}m)")

        ax.plot([0, y_stbd], [0, x_fwd], color=neigh_traj.color, linestyle="--", linewidth=1.2, alpha=0.7)

        status_txt = "ALERTA COLISÃO!" if is_collision else f"Dist: {dist_rel:.1f} m\nBrg: {bearing_rel:+.1f}°"
        ax.text(y_stbd + 5, x_fwd + 5, f"{neigh_traj.label}\n{status_txt}", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=marker_color, alpha=0.9), zorder=7)

    ax.set_xlim(-max_range_m, max_range_m)
    ax.set_ylim(-max_range_m, max_range_m)
    ax.set_xlabel("Estibordo / Starboard (+y) [m]  <-- Bombordo / Port (-y)", fontsize=10)
    ax.set_ylabel("Proa / Forward (+x) [m]  <-- Ré / Stern (-x)", fontsize=10)
    ax.set_title(f"Vista Local (Body Frame de {sel_traj.label})\nInstante: t = {t - dataset.common_start:.1f} s",
                 fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    return fig, ax


def plot_relative_aer_timeseries(
    dataset: MissionDataset,
    reference_robot_id: str,
    dt: float = 2.0
) -> Tuple[plt.Figure, np.ndarray]:
    """Desenha a evolução do azimute, da elevação e da distância dos outros robôs vistos pelo robô de referência."""
    t_min = dataset.common_start
    t_max = dataset.common_end
    t_grid = np.arange(t_min, t_max, dt)
    t_rel_min = (t_grid - t_min) / 60.0

    ref_traj = dataset.trajectories[reference_robot_id]
    other_ids = [r_id for r_id in dataset.trajectories.keys() if r_id != reference_robot_id]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=150)

    for other_id in other_ids:
        other_traj = dataset.trajectories[other_id]
        ranges = []
        bearings = []
        elevations = []

        for t in t_grid:
            states = dataset.get_state_at(t)
            if reference_robot_id in states and other_id in states:
                p_ref = states[reference_robot_id]["pos_ned"]
                q_ref = states[reference_robot_id]["quat"]
                p_other = states[other_id]["pos_ned"]

                p_body = transform_to_local_robot_frame(p_other, p_ref, q_ref)
                brg, el, rng = calculate_relative_aer_body(p_body)
                ranges.append(rng)
                bearings.append(brg)
                elevations.append(el)
            else:
                ranges.append(np.nan)
                bearings.append(np.nan)
                elevations.append(np.nan)

        axes[0].plot(t_rel_min, ranges, color=other_traj.color, label=f"Distância a {other_traj.label}")
        axes[1].plot(t_rel_min, bearings, color=other_traj.color, label=f"Marcação (Bearing) a {other_traj.label}")
        axes[2].plot(t_rel_min, elevations, color=other_traj.color, label=f"Elevação Relativa a {other_traj.label}")

    axes[0].set_ylabel("Slant Range [m]", fontsize=9)
    axes[0].set_title(f"Métricas AER no Referencial do Robô de Referência: {ref_traj.label}", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].set_ylabel("Relative Bearing [°]", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].set_ylabel("Relative Elevation [°]", fontsize=9)
    axes[2].set_xlabel("Tempo Decorrido na Missão [minutos]", fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig, axes


def export_all_report_figures(
    dataset: MissionDataset,
    output_dir: str = "docs/figures",
    selected_robot_id: str = "Robot 1"
) -> List[str]:
    """Gera e grava em PDF e PNG as figuras do relatório (Vista Global, Vista Local e gráficos AER)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # Vista Global, Vista Local e gráficos AER
    fig_glob, _ = plot_global_view(dataset, current_time=dataset.common_start + 600.0, frame="NED")
    f_glob = out_path / "global_view_trajectories.pdf"
    fig_glob.savefig(f_glob, bbox_inches="tight")
    fig_glob.savefig(out_path / "global_view_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close(fig_glob)
    generated_files.append(str(f_glob))

    fig_loc, _ = plot_local_view(dataset, selected_robot_id=selected_robot_id,
                                 current_time=dataset.common_start + 600.0)
    f_loc = out_path / "local_view_body_frame.pdf"
    fig_loc.savefig(f_loc, bbox_inches="tight")
    fig_loc.savefig(out_path / "local_view_body_frame.png", dpi=300, bbox_inches="tight")
    plt.close(fig_loc)
    generated_files.append(str(f_loc))

    fig_aer, _ = plot_relative_aer_timeseries(dataset, reference_robot_id=selected_robot_id)
    f_aer = out_path / "aer_relative_profiles.pdf"
    fig_aer.savefig(f_aer, bbox_inches="tight")
    fig_aer.savefig(out_path / "aer_relative_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig_aer)
    generated_files.append(str(f_aer))

    return generated_files
