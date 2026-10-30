"""Leitura e validação do ficheiro de configuração da missão (config/config.yaml).

Origem das coordenadas, robôs e respetivas cores, limiares de segurança,
parâmetros de reprodução e caminhos dos ficheiros de saída.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass
class OriginConfig:
    """Origem das coordenadas locais (latitude e longitude em graus, altitude em metros)."""
    latitude: float
    longitude: float
    altitude: float


@dataclass
class RobotConfig:
    """Um robô: identificador, ficheiro de dados, cor e nome a mostrar."""
    id: str
    file: str
    color: str
    label: str = ""

    def __post_init__(self):
        """Usa o identificador como nome quando não é indicado."""
        if not self.label:
            self.label = self.id


@dataclass
class SpatialConfig:
    """Parâmetros de análise espacial: raio da vista local, limiar de colisão e escala das elipses."""
    neighbor_radius_m: float = 250.0
    collision_threshold_m: float = 30.0
    covariance_sigma: float = 2.0


@dataclass
class ReplayConfig:
    """Parâmetros da reprodução da missão: passo de tempo, imagens por segundo e aceleração."""
    step_s: float = 1.0
    fps: int = 15
    speed_multiplier: float = 5.0


@dataclass
class OutputConfig:
    """Caminhos dos ficheiros de saída (registo, CSV, métricas e figuras)."""
    log_file: str = "output/mission_execution.log"
    export_csv: str = "output/synchronized_trajectories.csv"
    metrics_file: str = "output/mission_metrics.json"
    figures_dir: str = "docs/figures"


@dataclass
class MissionConfig:
    """Configuração completa da missão."""
    origin: OriginConfig
    frame: str
    robots: List[RobotConfig]
    selected_robot: str
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def get_robot_by_id(self, robot_id: str) -> Optional[RobotConfig]:
        """Devolve a configuração de um robô pelo seu identificador (ou None)."""
        for r in self.robots:
            if r.id == robot_id:
                return r
        return None


def load_config(config_path: str = "config/config.yaml", base_dir: Optional[str] = None) -> MissionConfig:
    """Esta função lê o ficheiro de configuração, verifica os campos obrigatórios
    e devolve um objeto MissionConfig.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro de configuração não encontrado: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Origem das coordenadas
    orig_data = data.get("origin", {})
    origin = OriginConfig(
        latitude=float(orig_data.get("latitude", 41.45201666666667)),
        longitude=float(orig_data.get("longitude", -8.81322222222222)),
        altitude=float(orig_data.get("altitude", 55.44791126176966))
    )

    frame = str(data.get("frame", "NED")).upper()
    if frame not in ("NED", "ENU"):
        raise ValueError(f"Referencial inválido '{frame}'. Deve ser 'NED' ou 'ENU'.")

    # Robôs (o caminho do ficheiro pode ser relativo à pasta do projeto)
    robots_data = data.get("robots", [])
    if not robots_data:
        raise ValueError("A configuração deve especificar pelo menos um robô.")

    robots: List[RobotConfig] = []
    base_path = Path(base_dir) if base_dir else path.parent.parent

    for r in robots_data:
        r_file = r["file"]
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

    # Parâmetros opcionais (com valores por omissão)
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
