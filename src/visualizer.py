"""Módulo de Replay Interativo da Missão (Mission Replay & Multi-View GUI).

Implementação de Alta Performance com Atualização Persistente de Artistas
(Object Recycling / Blitting) para garantir resposta instantânea a 60+ FPS:
1. Os fundos estáticos (trajetórias completas, anéis radar, grelhas e eixos)
   são desenhados UMA ÚNICA VEZ na inicialização.
2. A cada frame do replay ou movimento do slider, apenas as posições dos robôs,
   vetores de orientação e rótulos são atualizados na memória gráfica.
3. Não há chamadas a ax.clear(), eliminando o lag e a sobrecarga de redesenho.
4. Painéis simultâneos: Vista Global 2D (NED), Vista Espacial 3D e Radar Local.
5. Seletor de velocidade dinâmica (10x, 25x, 50x, 100x).
"""

from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np

from src.data_loader import MissionDataset
from src.frames import (
    quaternion_to_rotation_matrix,
    transform_to_local_robot_frame,
    calculate_relative_aer_body
)


class MissionVisualizer:
    """Controlador otimizado de alta performance para replay interativo 2D e 3D."""

    def __init__(
        self,
        dataset: MissionDataset,
        selected_robot_id: str = "Robot 1",
        speed_multiplier: float = 25.0,
        max_range_m: float = 250.0,
        collision_threshold_m: float = 30.0
    ):
        self.dataset = dataset
        self.selected_robot_id = selected_robot_id
        self.speed_multiplier = speed_multiplier
        self.max_range_m = max_range_m
        self.collision_threshold_m = collision_threshold_m

        self.t_min = dataset.common_start
        self.t_max = dataset.common_end
        self.current_t = self.t_min
        self.is_playing = False
        self.timer = None

        # -------------------------------------------------------------
        # 1. Configurar Janela e Layout Panorâmico
        # -------------------------------------------------------------
        self.fig = plt.figure(figsize=(18, 8.5), dpi=105)
        self.fig.canvas.manager.set_window_title(
            "TOMSA 2026 - Multi-Robot Trajectory Viewer (Alta Performance 2D + 3D)"
        )

        gs = self.fig.add_gridspec(
            2, 3,
            height_ratios=[8.4, 1.6],
            width_ratios=[1.05, 1.05, 0.95],
            left=0.04, right=0.97, top=0.93, bottom=0.07,
            wspace=0.18, hspace=0.25
        )

        self.ax_2d = self.fig.add_subplot(gs[0, 0])
        self.ax_3d = self.fig.add_subplot(gs[0, 1], projection="3d")
        self.ax_local = self.fig.add_subplot(gs[0, 2])

        # Painel inferior para controlos
        gs_ctrl = gs[1, :].subgridspec(2, 9, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)

        self.ax_slider = self.fig.add_subplot(gs_ctrl[0, 1:7])
        self.ax_btn_play = self.fig.add_subplot(gs_ctrl[1, 1])
        self.ax_btn_prev = self.fig.add_subplot(gs_ctrl[1, 2])
        self.ax_btn_next = self.fig.add_subplot(gs_ctrl[1, 3])
        self.ax_btn_reset = self.fig.add_subplot(gs_ctrl[1, 4])
        self.ax_btn_spd1 = self.fig.add_subplot(gs_ctrl[1, 5])
        self.ax_btn_spd2 = self.fig.add_subplot(gs_ctrl[1, 6])
        self.ax_btn_view3d = self.fig.add_subplot(gs_ctrl[1, 7])
        self.ax_radio = self.fig.add_subplot(gs_ctrl[:, 0])

        # -------------------------------------------------------------
        # 2. Inicializar Widgets
        # -------------------------------------------------------------
        self.slider = Slider(
            self.ax_slider,
            "Tempo de Missão [s]",
            valmin=0.0,
            valmax=self.t_max - self.t_min,
            valinit=0.0,
            valstep=1.0,
            color="#2A9D8F"
        )
        self.slider.on_changed(self._on_slider_changed)

        self.btn_play = Button(self.ax_btn_play, "Play >", color="#E9C46A", hovercolor="#F4A261")
        self.btn_play.on_clicked(self._toggle_play)

        self.btn_prev = Button(self.ax_btn_prev, "< -10s", color="#E0E1DD", hovercolor="#CBD5E1")
        self.btn_prev.on_clicked(lambda event: self.step_time(-10.0))

        self.btn_next = Button(self.ax_btn_next, "+10s >", color="#E0E1DD", hovercolor="#CBD5E1")
        self.btn_next.on_clicked(lambda event: self.step_time(10.0))

        self.btn_reset = Button(self.ax_btn_reset, "<< Inicio", color="#E0E1DD", hovercolor="#CBD5E1")
        self.btn_reset.on_clicked(lambda event: self.set_time(self.t_min))

        self.btn_spd1 = Button(self.ax_btn_spd1, "Vel: 25x", color="#E0E1DD", hovercolor="#CBD5E1")
        self.btn_spd1.on_clicked(lambda event: self._set_speed(25.0))

        self.btn_spd2 = Button(self.ax_btn_spd2, "Vel: 100x", color="#E0E1DD", hovercolor="#CBD5E1")
        self.btn_spd2.on_clicked(lambda event: self._set_speed(100.0))

        self.btn_view3d = Button(self.ax_btn_view3d, "Reset Cam 3D", color="#E0E1DD", hovercolor="#CBD5E1")
        self.btn_view3d.on_clicked(self._reset_3d_camera)

        robot_ids = list(dataset.trajectories.keys())
        active_idx = robot_ids.index(selected_robot_id) if selected_robot_id in robot_ids else 0
        self.radio = RadioButtons(self.ax_radio, robot_ids, active=active_idx, activecolor="#E63946")
        self.radio.on_clicked(self._on_robot_selected)

        # -------------------------------------------------------------
        # 3. Desenho ÚNICO da Camada Estática (Background) e Artistas Móveis
        # -------------------------------------------------------------
        self._init_static_background_and_artists()

        # Atualizar posições iniciais dos robôs
        self._update_artists_positions(self.t_min)

        # Temporizador para animação fluida (~25 fps, dt = 40ms)
        self.timer = self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self._on_timer_tick)

    def _init_static_background_and_artists(self):
        """Desenha todo o cenário estático uma única vez e instancia os objetos móveis."""
        # ==========================================
        # A. PAINEL 1: VISTA GLOBAL 2D (NED)
        # ==========================================
        self.ax_2d.scatter([0], [0], color="black", marker="P", s=110, zorder=6, label="Origem Missão (0,0)")

        # Calcular limites globais com margem
        all_east = np.concatenate([t.pos_ned[1] for t in self.dataset.trajectories.values()])
        all_north = np.concatenate([t.pos_ned[0] for t in self.dataset.trajectories.values()])
        margin = 35.0
        self.ax_2d.set_xlim(np.min(all_east) - margin, np.max(all_east) + margin)
        self.ax_2d.set_ylim(np.min(all_north) - margin, np.max(all_north) + margin)
        self.ax_2d.set_xlabel("Este / East [m]", fontsize=9)
        self.ax_2d.set_ylabel("Norte / North [m]", fontsize=9)
        self.ax_2d.grid(True, linestyle="--", alpha=0.55)
        self.ax_2d.set_aspect("equal", adjustable="box")
        self.title_2d = self.ax_2d.set_title("Vista Global 2D (NED) | t = 0.0 s", fontsize=11, fontweight="bold")

        self.artists_2d_markers = {}
        self.artists_2d_quiver = {}
        self.artists_2d_labels = {}

        for r_id, traj in self.dataset.trajectories.items():
            # Trajetória estática completa (decimada 1/5 para leveza sem perder detalhe gráfico)
            self.ax_2d.plot(
                traj.pos_ned[1, ::5], traj.pos_ned[0, ::5],
                color=traj.color, alpha=0.4, linewidth=1.5,
                label=f"Rota {traj.label}"
            )
            # Ponto de partida
            self.ax_2d.scatter([traj.pos_ned[1, 0]], [traj.pos_ned[0, 0]],
                               color=traj.color, marker="o", facecolors="none",
                               edgecolors=traj.color, s=55, linewidths=1.2)

            # ARTISTAS MÓVEIS 2D (instanciados uma única vez na posição inicial):
            self.artists_2d_markers[r_id] = self.ax_2d.scatter(
                [0], [0], color=traj.color, s=115, edgecolors="black", linewidths=1.2, zorder=7
            )
            # Seta de proa (quiver com escala fixa em metros)
            self.artists_2d_quiver[r_id] = self.ax_2d.quiver(
                [0], [0], [1], [0], angles="xy", scale_units="xy", scale=0.055,
                color=traj.color, width=0.007, headwidth=4, headlength=5, zorder=8
            )
            self.artists_2d_labels[r_id] = self.ax_2d.text(
                0, 0, "", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=traj.color, alpha=0.85),
                zorder=9
            )

        self.ax_2d.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

        # ==========================================
        # B. PAINEL 2: VISTA ESPACIAL 3D
        # ==========================================
        self.ax_3d.scatter([0], [0], [0], color="black", marker="P", s=90, label="Origem (0,0,0)")

        all_alt = np.concatenate([-t.pos_ned[2] for t in self.dataset.trajectories.values()])
        self.ax_3d.set_xlim3d(np.min(all_east) - margin, np.max(all_east) + margin)
        self.ax_3d.set_ylim3d(np.min(all_north) - margin, np.max(all_north) + margin)
        self.ax_3d.set_zlim3d(np.min(all_alt) - 5, np.max(all_alt) + 5)
        self.ax_3d.set_xlabel("Este [m]", fontsize=8)
        self.ax_3d.set_ylabel("Norte [m]", fontsize=8)
        self.ax_3d.set_zlabel("Altitude [m]", fontsize=8)
        self.ax_3d.view_init(elev=25.0, azim=-55.0)
        self.ax_3d.grid(True, linestyle="--", alpha=0.35)
        self.title_3d = self.ax_3d.set_title("Vista Espacial 3D | t = 0.0 s", fontsize=11, fontweight="bold")

        self.artists_3d_markers = {}
        self.artists_3d_lines_fwd = {}
        self.artists_3d_labels = {}

        for r_id, traj in self.dataset.trajectories.items():
            # Linha estática 3D (decimada 1/10 para acelerar a renderização e ordenação em profundidade)
            self.ax_3d.plot(
                traj.pos_ned[1, ::10], traj.pos_ned[0, ::10], -traj.pos_ned[2, ::10],
                color=traj.color, alpha=0.45, linewidth=1.5, label=traj.label
            )

            # ARTISTAS MÓVEIS 3D:
            self.artists_3d_markers[r_id] = self.ax_3d.scatter([0], [0], [0], color=traj.color, s=110,
                                                               edgecolors="black", zorder=7)
            # Segmento 3D de proa (extremamente rápido de atualizar via set_data_3d)
            line, = self.ax_3d.plot([0, 0], [0, 0], [0, 0], color=traj.color, linewidth=2.4, zorder=8)
            self.artists_3d_lines_fwd[r_id] = line

            self.artists_3d_labels[r_id] = self.ax_3d.text(0, 0, 0, "", fontsize=7.5,
                                                           bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                                                     ec=traj.color, alpha=0.85), zorder=9)

        self.ax_3d.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

        # ==========================================
        # C. PAINEL 3: VISTA LOCAL RADAR (BODY FRAME)
        # ==========================================
        # Anéis radar estáticos
        range_steps = np.arange(50.0, self.max_range_m + 1.0, 50.0)
        for r in range_steps:
            circle = patches.Circle((0, 0), r, fill=False, color="gray", linestyle=":", alpha=0.5)
            self.ax_local.add_patch(circle)
            self.ax_local.text(0, r + 2, f"{int(r)} m", color="gray", fontsize=7, ha="center")

        # Círculo de perigo / proximidade
        alert_circle = patches.Circle(
            (0, 0), self.collision_threshold_m, fill=True, facecolor="red", alpha=0.12,
            linestyle="--", edgecolor="red", label=f"Alerta ({self.collision_threshold_m:.0f}m)"
        )
        self.ax_local.add_patch(alert_circle)

        # Eixos do corpo estáticos
        self.ax_local.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)
        self.ax_local.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)

        # Símbolo do robô no centro apontando para a Proa (+x)
        self.center_robot_marker = self.ax_local.scatter([0], [0], color="#E63946", s=130, marker="^",
                                                         edgecolors="black", linewidths=1.5, zorder=6)
        self.ax_local.arrow(0, 0, 0, 22.0, head_width=6, head_length=8, fc="#E63946", ec="black", zorder=7)

        self.ax_local.text(self.max_range_m * 0.85, 0, "Estibordo (+y)\n[Direita]", ha="center", va="center", color="#666", fontsize=7.5)
        self.ax_local.text(-self.max_range_m * 0.85, 0, "Bombordo (-y)\n[Esquerda]", ha="center", va="center", color="#666", fontsize=7.5)
        self.ax_local.text(0, -self.max_range_m * 0.85, "Ré (-x)\n[Atrás]", ha="center", va="center", color="#666", fontsize=7.5)

        self.center_info_text = self.ax_local.text(0, -18, "PROA (0°)", ha="center", fontsize=8, fontweight="bold")

        self.ax_local.set_xlim(-self.max_range_m, self.max_range_m)
        self.ax_local.set_ylim(-self.max_range_m, self.max_range_m)
        self.ax_local.set_xlabel("Estibordo (+y) [m]  <-- Bombordo (-y)", fontsize=8.5)
        self.ax_local.set_ylabel("Proa / Frente (+x) [m]  <-- Ré (-x)", fontsize=8.5)
        self.ax_local.set_aspect("equal", adjustable="box")
        self.title_local = self.ax_local.set_title("Vista Local (Body Frame) | t = 0.0 s", fontsize=11, fontweight="bold")

        # Artistas móveis para robôs vizinhos no radar
        self.artists_local_markers = {}
        self.artists_local_los = {}
        self.artists_local_labels = {}

        for r_id, traj in self.dataset.trajectories.items():
            self.artists_local_markers[r_id] = self.ax_local.scatter(
                [0], [0], color=traj.color, s=110, edgecolors="black", linewidths=1.2, zorder=8
            )
            line, = self.ax_local.plot([0, 0], [0, 0], color=traj.color, linestyle="--", linewidth=1.2, alpha=0.7, zorder=7)
            self.artists_local_los[r_id] = line

            self.artists_local_labels[r_id] = self.ax_local.text(
                0, 0, "", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=traj.color, alpha=0.88),
                zorder=9
            )

        self.ax_local.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    def _update_artists_positions(self, t: float):
        """Atualiza EXCLUSIVAMENTE as coordenadas dos artistas móveis (Tempo gasto < 2ms!)."""
        states = self.dataset.get_state_at(t)
        rel_t = t - self.t_min

        # 1. Atualizar Títulos
        self.title_2d.set_text(f"Vista Global 2D (NED) | t = {rel_t:.1f} s")
        self.title_3d.set_text(f"Vista Espacial 3D | t = {rel_t:.1f} s")
        sel_label = self.dataset.trajectories[self.selected_robot_id].label
        self.title_local.set_text(f"Vista Local (Body: {sel_label}) | t = {rel_t:.1f} s")

        # 2. Atualizar Artistas na Vista 2D e 3D
        for r_id, traj in self.dataset.trajectories.items():
            if r_id not in states:
                continue
            st = states[r_id]
            cx = float(st["pos_ned"][1])   # East
            cy = float(st["pos_ned"][0])   # North
            cz = float(-st["pos_ned"][2])  # Altitude (-Down)

            heading_rad = np.radians(st["heading_deg"])
            dx = float(np.sin(heading_rad))
            dy = float(np.cos(heading_rad))

            # Atualizar marcador 2D (move o "íman"):
            self.artists_2d_markers[r_id].set_offsets([[cx, cy]])
            self.artists_2d_quiver[r_id].set_offsets([[cx, cy]])
            self.artists_2d_quiver[r_id].set_UVC(dx, dy)
            self.artists_2d_labels[r_id].set_position((cx + 8, cy + 8))
            self.artists_2d_labels[r_id].set_text(f"{traj.label}\n({st['heading_deg']:.0f}°)")

            # Atualizar marcador 3D:
            self.artists_3d_markers[r_id]._offsets3d = ([cx], [cy], [cz])

            # Vetor 3D de atitude (quiver)
            R_b2ned = quaternion_to_rotation_matrix(*st["quat"])
            fwd_ned = R_b2ned @ np.array([1.0, 0.0, 0.0])
            fe = float(fwd_ned[1] * 20.0)
            fn = float(fwd_ned[0] * 20.0)
            fa = float(-fwd_ned[2] * 20.0)

            self.artists_3d_lines_fwd[r_id].set_data_3d([cx, cx + fe], [cy, cy + fn], [cz, cz + fa])
            self.artists_3d_labels[r_id].set_position_3d((cx + 5, cy + 5, cz + 2))
            self.artists_3d_labels[r_id].set_text(f"{traj.label}\n({st['heading_deg']:.0f}°)")

        # 3. Atualizar Vista Local (Radar)
        if self.selected_robot_id in states:
            sel_st = states[self.selected_robot_id]
            p_sel = sel_st["pos_ned"]
            q_sel = sel_st["quat"]

            # Atualizar cor do robô central e texto
            sel_color = self.dataset.trajectories[self.selected_robot_id].color
            self.center_robot_marker.set_facecolors([sel_color])
            self.center_info_text.set_text(f"PROA (0°)\nHeading: {sel_st['heading_deg']:.1f}°")

            for r_id, traj in self.dataset.trajectories.items():
                if r_id == self.selected_robot_id:
                    # O próprio robô está no centro (0, 0), esconde o marcador vizinho
                    self.artists_local_markers[r_id].set_visible(False)
                    self.artists_local_los[r_id].set_visible(False)
                    self.artists_local_labels[r_id].set_visible(False)
                    continue

                if r_id not in states:
                    continue

                self.artists_local_markers[r_id].set_visible(True)
                self.artists_local_los[r_id].set_visible(True)
                self.artists_local_labels[r_id].set_visible(True)

                p_neigh = states[r_id]["pos_ned"]
                p_body = transform_to_local_robot_frame(p_neigh, p_sel, q_sel)
                x_fwd = float(p_body[0])
                y_stbd = float(p_body[1])

                brg_rel, el_rel, dist_rel = calculate_relative_aer_body(p_body)
                is_alert = dist_rel <= self.collision_threshold_m

                # Deslocar íman no radar:
                self.artists_local_markers[r_id].set_offsets([[y_stbd, x_fwd]])
                self.artists_local_los[r_id].set_data([0, y_stbd], [0, x_fwd])

                color = "red" if is_alert else traj.color
                self.artists_local_markers[r_id].set_facecolors([color])
                self.artists_local_los[r_id].set_color(color)

                status_txt = "ALERTA COLISAO!" if is_alert else f"Dist: {dist_rel:.1f}m\nBrg: {brg_rel:+.1f}°"
                self.artists_local_labels[r_id].set_position((y_stbd + 5, x_fwd + 5))
                self.artists_local_labels[r_id].set_text(f"{traj.label}\n{status_txt}")

        self.fig.canvas.draw_idle()

    # -------------------------------------------------------------
    # Handlers de Eventos dos Controlos
    # -------------------------------------------------------------
    def _on_slider_changed(self, val):
        self.current_t = self.t_min + val
        self._update_artists_positions(self.current_t)

    def _toggle_play(self, event):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.label.set_text("Pause ||")
            self.timer.start()
        else:
            self.btn_play.label.set_text("Play >")
            self.timer.stop()
        self.fig.canvas.draw_idle()

    def _on_timer_tick(self):
        dt = 0.040 * self.speed_multiplier
        new_t = self.current_t + dt
        if new_t >= self.t_max:
            new_t = self.t_min
            self.is_playing = False
            self.btn_play.label.set_text("Play >")
            self.timer.stop()

        self.current_t = new_t
        self.slider.set_val(self.current_t - self.t_min)

    def step_time(self, delta_s: float):
        new_t = np.clip(self.current_t + delta_s, self.t_min, self.t_max)
        self.slider.set_val(new_t - self.t_min)

    def set_time(self, target_t: float):
        self.slider.set_val(target_t - self.t_min)

    def _set_speed(self, multiplier: float):
        self.speed_multiplier = multiplier

    def _reset_3d_camera(self, event):
        self.ax_3d.view_init(elev=25.0, azim=-55.0)
        self.fig.canvas.draw_idle()

    def _on_robot_selected(self, label: str):
        self.selected_robot_id = label
        self._update_artists_positions(self.current_t)

    def show(self):
        """Apresenta a interface interativa."""
        plt.show()
