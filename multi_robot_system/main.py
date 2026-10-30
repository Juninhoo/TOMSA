#!/usr/bin/env python3
"""Programa principal do dashboard multi-robô (linha de comandos).

Carrega a missão e, conforme as opções, corre os testes, exporta os dados,
compara os interpoladores, gera o HTML autónomo ou abre o dashboard web.
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, List
import yaml

from multi_robot_system.core.data.odom_loader import load_single_robot_csv, compute_reference_origin
from multi_robot_system.core.data.trajectory import MissionDataset, RobotTrajectory
from multi_robot_system.visualization.web_server import MissionWebDashboard
from multi_robot_system.io_export.geojson_exporter import export_mission_to_geojson
from multi_robot_system.io_export.kml_exporter import export_mission_to_kml
from multi_robot_system.io_export.benchmark_exporter import (
    run_full_interpolator_benchmark,
    generate_latex_table
)


def run_test_suite():
    """Corre os testes automáticos com pytest e para o programa se algum falhar."""
    import subprocess
    print("\n" + "=" * 65)
    print("A EXECUTAR TESTES UNITÁRIOS DE RIGOR MATEMÁTICO (PYTEST)...")
    print("=" * 65)
    cmd = [sys.executable, "-m", "pytest", "multi_robot_system/tests/", "-v"]
    res = subprocess.run(cmd, env={"PYTHONPATH": "."})
    if res.returncode != 0:
        print("\nALERTA: Alguns testes matemáticos falharam!")
        sys.exit(res.returncode)
    print("\nTODOS OS TESTES FORAM VALIDADOS COM SUCESSO (100% APROVAÇÃO)!\n")


# Cores atribuídas aos robôs carregados com --dados
PALETA_CORES = ["#E63946", "#2A9D8F", "#457B9D", "#F4A261", "#9B5DE5", "#FFD166", "#06D6A0", "#EF476F"]


def robots_from_files(paths, robots_config=()):
    """Esta função cria a lista de robôs a partir de ficheiros CSV (ou pastas com CSV).

    O nome vem do ficheiro (robot7_s2.csv -> Robot 7); um robô que já existe na configuração mantém a sua cor.
    """
    ficheiros = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            ficheiros += sorted(pp.glob("*.csv"))
        else:
            ficheiros.append(pp)
    if not ficheiros:
        raise SystemExit("Não encontrei ficheiros CSV em: " + ", ".join(str(p) for p in paths))
    robots, usados = [], set()
    for k, f in enumerate(ficheiros):
        m = re.search(r"robot\D*(\d+)", f.stem, re.IGNORECASE)
        rid = "Robot " + str(int(m.group(1))) if m else f.stem
        if rid in usados:
            rid = rid + " (" + f.stem + ")"
        usados.add(rid)
        cor = next((r["color"] for r in robots_config if r["id"] == rid), PALETA_CORES[k % len(PALETA_CORES)])
        robots.append({"id": rid, "file": str(f), "color": cor, "label": rid})
    return robots


def load_mission(config_path: str, data_paths=None) -> Tuple[MissionDataset, dict]:
    """Esta função carrega a missão completa a partir do ficheiro de configuração.

    Calcula a origem das coordenadas, lê e suaviza os dados de cada robô
    e devolve o conjunto de trajetórias e a configuração.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Ficheiros indicados com --dados: substituem os robôs da configuração e a origem passa a vir dos dados
    if data_paths:
        cfg["robots"] = robots_from_files(data_paths, cfg.get("robots", []))
        cfg["selected_robot"] = cfg["robots"][0]["id"]
        cfg["origin"] = dict(cfg["origin"], mode="data_mean")

    def _resolve(csv_file: str) -> str:
        # Aceita o caminho tal como está ou relativo à pasta multi_robot_system
        return csv_file if Path(csv_file).is_file() else str(Path("multi_robot_system") / csv_file)

    # Origem das coordenadas locais: fixa (configuração) ou calculada a partir dos dados
    orig = cfg["origin"]
    if orig.get("mode", "fixed") == "data_mean":
        lat0, lon0, h0 = compute_reference_origin([_resolve(r["file"]) for r in cfg["robots"]])
    else:
        lat0, lon0, h0 = float(orig["latitude"]), float(orig["longitude"]), float(orig["altitude"])
    cfg["origin_resolved"] = {"latitude": lat0, "longitude": lon0, "altitude": h0,
                              "mode": "data_mean" if orig.get("mode", "fixed") == "data_mean" else "surface_reference"}

    print(f"\n[INICIALIZAÇÃO] Origem de Referência: Lat={lat0:.6f}°, Lon={lon0:.6f}°, Alt={h0:.2f}m")
    trajectories = {}

    # Leitura e suavização dos dados de cada robô
    for rob in cfg["robots"]:
        r_id = rob["id"]
        csv_file = rob["file"]
        if not Path(csv_file).is_file():
            csv_file = str(Path("multi_robot_system") / csv_file)

        print(f"  -> A carregar e suavizar {r_id} ({rob['label']}) a partir de {csv_file}...")
        t0 = time.perf_counter()
        raw_dict = load_single_robot_csv(
            file_path=csv_file,
            robot_id=r_id,
            label=rob["label"],
            color=rob["color"],
            origin_lat=lat0,
            origin_lon=lon0,
            origin_h=h0,
            apply_rts=cfg.get("filter", {}).get("use_rts_smoother", True),
            process_noise_q=cfg.get("filter", {}).get("process_noise_q", 0.05)
        )
        traj = RobotTrajectory(
            robot_id=raw_dict["robot_id"],
            label=raw_dict["label"],
            color=raw_dict["color"],
            timestamps=raw_dict["timestamps"],
            lat=raw_dict["lat"],
            lon=raw_dict["lon"],
            alt=raw_dict["alt"],
            pos_ned=raw_dict["pos_ned"],
            quaternions=raw_dict["quaternions"],
            rot_matrices=raw_dict["rot_matrices"],
            euler_deg=raw_dict["euler_deg"],
            vel_ned=raw_dict["vel_ned"],
            speed_longitudinal=raw_dict["speed_longitudinal"],
            turn_rate_yaw=raw_dict["turn_rate_yaw"],
            cov_diag=raw_dict["cov_diag"],
            vel_body_meas=raw_dict.get("vel_body_meas"),
            sigma_h_decl=raw_dict.get("sigma_h_decl"),
            jump_events=raw_dict.get("jump_events"),
            pos_ned_raw=raw_dict.get("pos_ned_raw")
        )
        trajectories[r_id] = traj
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"     OK ({len(traj.timestamps)} amostras processadas em {dt_ms:.1f}ms, dur={traj.duration_s:.1f}s)")

    dataset = MissionDataset(trajectories, lat0, lon0, h0)
    # Aviso quando os robôs nunca estiveram ativos ao mesmo tempo
    if dataset.has_overlap:
        print(f"[SINCRONIZAÇÃO] Janela temporal comum: {dataset.common_end - dataset.common_start:.1f}s sobreposta.")
    else:
        print("[AVISO] Os robots nunca estiveram ativos ao mesmo tempo: não há janela comum, por isso as distâncias e o CPA entre eles não se aplicam.")
    return dataset, cfg


def main():
    """Lê as opções da linha de comandos e executa as tarefas pedidas."""
    # Opções disponíveis na linha de comandos
    parser = argparse.ArgumentParser(
        description="TOMSA 2026 - Multi-Robot Subsea Trajectory System (INESC TEC Aguçadoura)"
    )
    parser.add_argument(
        "--config", type=str, default="multi_robot_system/config/mission_config.yaml",
        help="Caminho para o ficheiro de configuração YAML"
    )
    parser.add_argument(
        "--dados", nargs="+", metavar="CSV_OU_PASTA",
        help="Ficheiros CSV (ou uma pasta com CSV) a usar no lugar dos da configuração. Ex.: --dados novos/robot1.csv novos/robot4.csv"
    )
    parser.add_argument("--web", action="store_true", help="Inicia o servidor e abre o Dashboard no browser (Plotly)")
    parser.add_argument("--export", action="store_true", help="Exporta ficheiros GeoJSON (QGIS), KML 3D (Google Earth) e CSV sincronizado")
    parser.add_argument("--benchmark", action="store_true", help="Executa o benchmark de comparação dos interpoladores (Linear vs Spline vs RTS)")
    parser.add_argument("--bundle", action="store_true", help="Gera o ficheiro HTML autónomo offline (output/mission_dashboard_bundle.html)")
    parser.add_argument("--test", action="store_true", help="Executa a suite de testes unitários com pytest")
    parser.add_argument("--all", action="store_true", help="Executa validação completa: testes, benchmark, exportação e dashboard")

    args = parser.parse_args()

    # Sem opções: executa tudo
    if len(sys.argv) == 1:
        print("Nenhum argumento especificado. A executar validação completa (--all)...")
        args.all = True

    if args.test or args.all:
        run_test_suite()

    dataset, cfg = load_mission(args.config, args.dados)

    # Exportação: CSV sincronizado, GeoJSON e KML
    if args.export or args.all:
        print("\n" + "=" * 65)
        print("A EXPORTAR DADOS GEOESPACIAIS E TELEMETRIA SINCRONIZADA...")
        print("=" * 65)
        csv_out = cfg["output"].get("export_csv", "output/synchronized_trajectories.csv")
        dataset.export_synchronized_csv(csv_out)
        print(f" [1/3] CSV Sincronizado exportado: {csv_out}")

        geojson_out = cfg["output"].get("geojson_file", "output/mission_trajectories.geojson")
        export_mission_to_geojson(dataset, geojson_out)
        print(f" [2/3] GeoJSON (QGIS) exportado: {geojson_out}")

        kml_out = cfg["output"].get("kml_file", "output/mission_trajectories_3d.kml")
        export_mission_to_kml(dataset, kml_out)
        print(f" [3/3] KML 3D (Google Earth) exportado: {kml_out}")

    # Comparação dos interpoladores e tabela LaTeX
    if args.benchmark or args.all:
        print("\n" + "=" * 65)
        print("A EXECUTAR BENCHMARK DE COMPARATIVO DE INTERPOLADORES...")
        print("=" * 65)
        bench_df = run_full_interpolator_benchmark(dataset)
        print("\nTabela Resumo do Benchmark (Média de Todos os Veículos):")
        agg = bench_df.groupby("method").agg({
            "rmse_m": "mean",
            "max_err_m": "mean",
            "compute_time_ms": "mean"
        })
        print(agg.to_string())

        latex_tab = generate_latex_table(bench_df)
        with open("multi_robot_system/output/interpolator_latex_table.tex", "w", encoding="utf-8") as f:
            f.write(latex_tab)
        print("\nTabela LaTeX guardada em: multi_robot_system/output/interpolator_latex_table.tex")

    # Dashboard: ficheiro HTML autónomo e/ou servidor web
    dashboard = MissionWebDashboard(dataset, cfg)

    if args.bundle or args.all:
        bundle_path = "multi_robot_system/output/mission_dashboard_bundle.html"
        dashboard.export_standalone_html(bundle_path)
        print(f"\n[HTML AUTÓNOMO OFFLINE] Gerado com sucesso: {bundle_path}")

    if args.web or args.all:
        dashboard.launch_server(open_browser=True)


if __name__ == "__main__":
    main()
