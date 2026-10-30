#!/usr/bin/env python3
"""Programa principal do visualizador Matplotlib (linha de comandos).

Conforme as opções: testes, relatórios (métricas, CSV, GeoJSON), figuras,
visualizador 3D Plotly ou janela interativa de reprodução.
"""

import argparse
import sys
from pathlib import Path

from src.config_loader import load_config
from src.data_loader import load_mission_dataset
from src.logger import setup_logger, export_mission_reports
from src.views import export_all_report_figures
from src.extras import (
    plot_3d_mission_trajectories,
    plot_inter_robot_distance_matrix,
    export_trajectories_to_geojson
)


def run_tests():
    """Corre os testes das conversões de coordenadas e para o programa se algum falhar."""
    import subprocess
    print("=" * 60)
    print("A EXECUTAR TESTES UNITÁRIOS DE RIGOR MATEMÁTICO...")
    print("=" * 60)
    cmd = [sys.executable, "-m", "pytest", "tests/test_coordinates.py", "-v"]
    res = subprocess.run(cmd, env={"PYTHONPATH": "."})
    if res.returncode != 0:
        print("ALERTA: Alguns testes falharam!")
        sys.exit(res.returncode)
    print("\nTodos os testes matemáticos foram validados com sucesso!")


def main():
    """Lê as opções da linha de comandos e executa as tarefas pedidas."""
    # Opções disponíveis na linha de comandos
    parser = argparse.ArgumentParser(
        description="TOMSA 2026 - Multi-Robot Trajectory Viewer & Analyzer (INESC TEC Aguçadoura)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Caminho para o ficheiro de configuração YAML (predefinido: config/config.yaml)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Inicia a interface gráfica interativa com Replay, Slider temporal e Vistas Global/Local"
    )
    parser.add_argument(
        "--export-figures",
        action="store_true",
        help="Gera e guarda todas as figuras de alta resolução (PDF/PNG) na pasta docs/figures"
    )
    parser.add_argument(
        "--export-reports",
        action="store_true",
        help="Gera o relatório de métricas JSON, trajetórias sincronizadas CSV e ficheiro GeoJSON"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Executa a suite de testes unitários com pytest"
    )
    parser.add_argument(
        "--plotly", "--web",
        action="store_true",
        help="Inicia o visualizador 3D interativo em primeira pessoa no navegador (Plotly)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Executa validação completa: testes, figuras, relatórios e exportação"
    )

    args = parser.parse_args()

    # Sem opções: executa tudo
    if len(sys.argv) == 1:
        print("Nenhuma opção especificada. A executar validação e exportação de figuras (--all)...")
        args.all = True

    if args.test or args.all:
        run_tests()

    # Carregamento da missão
    logger = setup_logger()
    logger.info("A iniciar TOMSA 2026 Multi-Robot Trajectory System...")
    logger.info(f"A carregar configuração de: {args.config}")

    dataset, config = load_mission_dataset(args.config)
    logger.info(f"Robôs carregados: {list(dataset.trajectories.keys())}")
    logger.info(f"Origem da missão: Lat={config.origin.latitude:.6f}°, Lon={config.origin.longitude:.6f}°, Alt={config.origin.altitude:.1f}m")
    logger.info(f"Duração da janela comum sobreposta: {dataset.common_end - dataset.common_start:.1f} segundos")

    # Relatórios: métricas JSON, CSV sincronizado e GeoJSON
    if args.export_reports or args.all:
        logger.info("A exportar relatórios de telemetria e métricas...")
        m_file, c_file = export_mission_reports(
            dataset=dataset,
            metrics_path=config.output.metrics_file,
            csv_path=config.output.export_csv,
            logger=logger
        )
        geojson_file = export_trajectories_to_geojson(dataset, "output/mission_trajectories.geojson")
        logger.info(f"GeoJSON exportado para: {geojson_file}")

    # Figuras do relatório (2D, 3D, proximidade) e página HTML 3D
    if args.export_figures or args.all:
        logger.info(f"A gerar figuras de alta resolução para {config.output.figures_dir}...")
        figs = export_all_report_figures(dataset, output_dir=config.output.figures_dir,
                                         selected_robot_id=config.selected_robot)
        for f in figs:
            logger.info(f"Figura gerada: {f}")

        f_3d = Path(config.output.figures_dir) / "mission_trajectories_3d.pdf"
        plot_3d_mission_trajectories(dataset, str(f_3d))
        logger.info(f"Figura 3D gerada: {f_3d}")

        f_prox = Path(config.output.figures_dir) / "proximity_analysis.pdf"
        plot_inter_robot_distance_matrix(dataset, str(f_prox), alert_threshold_m=config.spatial.collision_threshold_m)
        logger.info(f"Figura de proximidade gerada: {f_prox}")

        from src.plotly_visualizer import launch_interactive_3d
        launch_interactive_3d(args.config, open_browser=False)
        logger.info("Visualização 3D Web exportada para output/mission_body_view_3d.html")

    # Visualizador 3D no navegador e janela interativa Matplotlib
    if args.plotly:
        logger.info("A lançar visualizador 3D em 1ª Pessoa no navegador (Plotly)...")
        from src.plotly_visualizer import launch_interactive_3d
        launch_interactive_3d(args.config, open_browser=True)

    if args.visualize:
        logger.info("A lançar MissionVisualizer interativo (Matplotlib)...")
        from src.visualizer import MissionVisualizer
        viz = MissionVisualizer(dataset, selected_robot_id=config.selected_robot,
                                speed_multiplier=config.replay.speed_multiplier)
        viz.show()

    logger.info("Processamento concluído com sucesso!")


if __name__ == "__main__":
    main()
