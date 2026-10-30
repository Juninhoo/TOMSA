"""Exportação dos resultados da comparação de interpoladores.

Gera um CSV com os erros e tempos de cada método, uma tabela LaTeX para o relatório
e uma figura de comparação.
"""

from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_robot_system.core.data.trajectory import MissionDataset
from multi_robot_system.core.math.interpolators import compare_interpolators_on_trajectory


def run_full_interpolator_benchmark(
    dataset: MissionDataset,
    output_csv_path: str = "output/interpolator_comparison.csv",
    output_fig_path: str = "docs/interpolator_comparison.png"
) -> pd.DataFrame:
    """Esta função compara os interpoladores em todos os robôs e grava o CSV e a figura de resultados."""
    records = []

    for r_id, traj in dataset.trajectories.items():
        bench = compare_interpolators_on_trajectory(traj.timestamps, traj.pos_ned, subsample_factor=6)
        for method, res in bench.items():
            records.append({
                "robot_id": r_id,
                "method": method,
                "continuity": res.get("continuity", "C0"),
                "rmse_m": res["rmse_m"],
                "max_err_m": res["max_err_m"],
                "mean_err_m": res["mean_err_m"],
                "compute_time_ms": res["compute_time_ms"]
            })

    df = pd.DataFrame(records)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    # Figura: interpolação linear vs spline num excerto do Robot 1 (à esquerda) e RMSE médio de cada método (à direita)
    try:
        traj_sample = next(iter(dataset.trajectories.values()))
        t_sub = traj_sample.timestamps[:120]
        pos_sub = traj_sample.pos_ned[:, :120]

        t_dense = np.linspace(t_sub[0], t_sub[-1], 300)

        lin_e = np.interp(t_dense, t_sub, pos_sub[1])
        lin_n = np.interp(t_dense, t_sub, pos_sub[0])

        from scipy.interpolate import CubicSpline
        cs_e = CubicSpline(t_sub, pos_sub[1], bc_type="natural")(t_dense)
        cs_n = CubicSpline(t_sub, pos_sub[0], bc_type="natural")(t_dense)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        fig.patch.set_facecolor("#f8f9fa")

        ax1.plot(pos_sub[1], pos_sub[0], "ko", markersize=3, label="Amostras Originais", alpha=0.6)
        ax1.plot(lin_e, lin_n, "b--", label="Linear (C0)", linewidth=1.5)
        ax1.plot(cs_e, cs_n, "r-", label="Spline Cúbica (C2)", linewidth=1.8)
        ax1.set_title("Comparação Espacial no Plano Horizontal (Norte vs Este)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Este (m)")
        ax1.set_ylabel("Norte (m)")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()

        methods = df["method"].unique()
        avg_rmse = [df[df["method"] == m]["rmse_m"].mean() for m in methods]
        bars = ax2.bar(methods, avg_rmse, color=["#457b9d", "#2a9d8f", "#e63946"], width=0.5)
        ax2.set_title("Erro Médio Quadrático Global (RMSE em Metros)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("RMSE (m)")
        ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f} m", ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        Path(output_fig_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_fig_path)
        plt.close()
    except Exception as e:
        print(f"Nota: Figura gráfica não gerada ({e})")

    return df


def generate_latex_table(df: pd.DataFrame) -> str:
    """Gera o código LaTeX da tabela de erros médios por método, para o relatório."""
    agg = df.groupby("method").agg({
        "continuity": "first",
        "rmse_m": "mean",
        "max_err_m": "mean",
        "compute_time_ms": "mean"
    }).reset_index()

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Método de Interpolação} & \textbf{Continuidade} & \textbf{RMSE Médio (m)} & \textbf{Desvio Máximo (m)} & \textbf{Tempo (ms)} \\",
        r"\midrule"
    ]
    for _, row in agg.iterrows():
        lines.append(
            f"{row['method']} & {row['continuity']} & {row['rmse_m']:.4f} & {row['max_err_m']:.4f} & {row['compute_time_ms']:.2f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Resultados quantitativos do benchmark de interpoladores nos dados da Aguçadoura.}",
        r"\end{table}"
    ])
    return "\n".join(lines)
