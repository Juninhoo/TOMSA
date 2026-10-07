"""Módulo de Carregamento e Validação da Configuração YAML.

Lê e valida os parâmetros da missão: origem de referência, robôs, cores,
frequência de replay, limiares de segurança e caminhos de exportação.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass
class OriginConfig:
    latitude: float
    longitude: float
    altitude: float


@dataclass
class RobotConfig:
    id: str
    file: str
    color: str
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.id


@dataclass
class SpatialConfig:
    neighbor_radius_m: float = 250.0
    collision_threshold_m: float = 30.0
    covariance_sigma: float = 2.0


@dataclass
class ReplayConfig:
    step_s: float = 1.0
    fps: int = 15
    speed_multiplier: float = 5.0


@dataclass
class OutputConfig:
    log_file: str = "output/mission_execution.log"
    export_csv: str = "output/synchronized_trajectories.csv"
    metrics_file: str = "output/mission_metrics.json"
    figures_dir: str = "docs/figures"


@dataclass
class MissionConfig:
    origin: OriginConfig
    frame: str
    robots: List[RobotConfig]
    selected_robot: str
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def get_robot_by_id(self, robot_id: str) -> Optional[RobotConfig]:
        for r in self.robots:
            if r.id == robot_id:
                return r
        return None


def load_config(config_path: str = "config/config.yaml", base_dir: Optional[str] = None) -> MissionConfig:
    """Carrega e valida o ficheiro de configuração YAML da missão."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro de configuração não encontrado: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Validar origem
    orig_data = data.get("origin", {})
    origin = OriginConfig(
        latitude=float(orig_data.get("latitude", 41.45201666666667)),
        longitude=float(orig_data.get("longitude", -8.81322222222222)),
        altitude=float(orig_data.get("altitude", 55.44791126176966))
    )

    frame = str(data.get("frame", "NED")).upper()
    if frame not in ("NED", "ENU"):
        raise ValueError(f"Referencial inválido '{frame}'. Deve ser 'NED' ou 'ENU'.")

    # Validar robôs
    robots_data = data.get("robots", [])
    if not robots_data:
        raise ValueError("A configuração deve especificar pelo menos um robô.")

    robots: List[RobotConfig] = []
    base_path = Path(base_dir) if base_dir else path.parent.parent

    for r in robots_data:
        r_file = r["file"]
        # Resolver caminho relativo ao projeto se necessário
        candidate_file = base_path / r_file
        resolved_file = str(candidate_file if candidate_file.exists() else r_file)

        robots.append(RobotConfig(
            id=str(r["id"]),
            file=resolved_file,
            color=str(r.get("color", "#0000FF")),
            label=str(r.get("label", r["id"]))
        ))

    selected_robot = str(data.get("selected_robot", robots[0].id))
    valid_ids = [r.id for r in robots]
    if selected_robot not in valid_ids:
        raise ValueError(f"selected_robot '{selected_robot}' não encontrado na lista {valid_ids}")

    # Sub-configurações opcionais
    sp_data = data.get("spatial", {})
    spatial = SpatialConfig(
        neighbor_radius_m=float(sp_data.get("neighbor_radius_m", 250.0)),
        collision_threshold_m=float(sp_data.get("collision_threshold_m", 30.0)),
        covariance_sigma=float(sp_data.get("covariance_sigma", 2.0))
    )

    rp_data = data.get("replay", {})
    replay = ReplayConfig(
        step_s=float(rp_data.get("step_s", 1.0)),
        fps=int(rp_data.get("fps", 15)),
        speed_multiplier=float(rp_data.get("speed_multiplier", 5.0))
    )

    out_data = data.get("output", {})
    output = OutputConfig(
        log_file=str(out_data.get("log_file", "output/mission_execution.log")),
        export_csv=str(out_data.get("export_csv", "output/synchronized_trajectories.csv")),
        metrics_file=str(out_data.get("metrics_file", "output/mission_metrics.json")),
        figures_dir=str(out_data.get("figures_dir", "docs/figures"))
    )

    return MissionConfig(
        origin=origin,
        frame=frame,
        robots=robots,
        selected_robot=selected_robot,
        spatial=spatial,
        replay=replay,
        output=output
    )
