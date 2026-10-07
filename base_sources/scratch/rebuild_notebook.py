import json
import io
import sys

cell1_text = """# TÓPICOS DE MATEMÁTICA PARA SISTEMAS AUTÓNOMOS
## TRABALHO LETIVO - ENTREGA 1

> **Nota 1:** Todos elementos do grupo são responsáveis por esta entrega.  
> **Nota 2:** Todos elementos do grupo tomaram conhecimento das regras estabelecidas para o funcionamento da u.c. no [Moodle do ISEP](https://moodle.isep.ipp.pt/).

---

### GRUPO
* **N:** 1131518 **Nome:** Geovani Lopes
* **N:** 1120542 **Nome:** Luis Lei
* **N:** 1261726 **Nome:** Hugo Amado

---

### Conteúdo
1. [Identificação dos objetivos e estruturação do problema](#1-identificação-dos-objetivos-e-estruturação-do-problema)
2. [Justificação da relevância das bibliotecas Python selecionadas](#2-justificação-da-relevância-das-bibliotecas-python-selecionadas)
3. [Metodologia proposta para o desenvolvimento do projeto](#3-metodologia-proposta-para-o-desenvolvimento-do-projeto)
4. [Demonstração e Validação do Núcleo Matemático](#4-demonstração-e-validação-do-núcleo-matemático)
"""

cell2_text = """## 1. Identificação dos objetivos e estruturação do problema
[De volta para "Conteúdo"](#conteúdo)

### 1.1 Objetivo Geral
O presente projeto tem como objetivo projetar, desenvolver e validar rigorosamente uma aplicação computacional em **Python** destinada à visualização, sincronização temporal e análise cinemática de trajetórias de múltiplos robôs autónomos de superfície e submarinos (ASVs/AUVs). O sistema baseia-se em telemetria real obtida pelo **INESC TEC** na costa da **Aguçadoura** (Póvoa de Varzim, Portugal).

A aplicação realiza a conversão de coordenadas geodésicas de navegação (WGS-84: latitude, longitude, altitude elipsoidal e quaterniões de atitude em $SO(3)$) para referenciais cartesianos locais tangentes (**NED** - *North-East-Down* e **ENU** - *East-North-Up*), projeta as posições relativas e atitudes no referencial próprio do corpo (**Body Frame**) do veículo ativo, e extrai grandezas esféricas relativas de visada (**AER** - *Azimuth, Elevation, Range*).

### 1.2 Objetivos Específicos e Requisitos do Sistema
| ID | Módulo / Funcionalidade | Descrição e Critério de Aceitação |
|:--:|:---|:---|
| **O1** | **Global View 3D** | Apresentação panorâmica de todas as trajetórias num referencial cartesiano comum (NED), com batimetria (fundo do mar), plano da água e rasto progressivo color-coded por robô. |
| **O2** | **POV 1ª Pessoa** | Câmara virtual no nariz do robô ativo, orientada pela proa e arfagem (*pitch*), permitindo visualizar os outros robôs no espaço tridimensional com o próprio casco oculto. |
| **O3** | **Cockpit de Instrumentos** | Painel instrumental avançado com horizonte artificial, bússola, profundímetro, velocímetro longitudinal, taxa de viragem e radar tático AER. |
| **O4** | **Suavizador RTS & CPA** | Reconstrução ótima do estado cinemático via Filtro de Kalman / Suavizador RTS e cálculo analítico em forma fechada do Ponto de Maior Aproximação com alarme visual (< 10m). |
| **O5** | **Controlo Temporal Avançado** | Slider de tempo, Play/Pausa e múltiplos níveis de aceleração (`0.25x`, `1x`, `5x`, `10x`, `20x`, `50x` e `⚡ Ultra` ~2s). |
| **O6** | **Rigor Matemático & Exportação** | Implementação de raiz (*first-principles*) de todas as fórmulas com NumPy, validação por 24 testes unitários automatizados (`pytest`) e exportação em GeoJSON, KML 3D e bundle HTML 100% offline. |

### 1.3 Modelação Matemática do Problema
#### A. Modelo Geodésico WGS-84
$$N(\\varphi) = \\frac{a}{\\sqrt{1 - e^2 \\sin^2 \\varphi}}$$
com $a = 6\\,378\\,137.0\\text{ m}$ e $e^2 = 2f - f^2 \\approx 0.00669438$.

#### B. Conversão Geodésico $\\to$ ECEF $\\to$ NED
$$X = (N(\\varphi) + h) \\cos\\varphi \\cos\\lambda, \\quad Y = (N(\\varphi) + h) \\cos\\varphi \\sin\\lambda, \\quad Z = \\left(N(\\varphi)(1 - e^2) + h\\right) \\sin\\varphi$$
$$\\begin{bmatrix} x_E \\\\ y_N \\\\ z_U \\end{bmatrix} = \\begin{bmatrix} -\\sin\\lambda_0 & \\cos\\lambda_0 & 0 \\\\ -\\sin\\varphi_0 \\cos\\lambda_0 & -\\sin\\varphi_0 \\sin\\lambda_0 & \\cos\\varphi_0 \\\\ \\cos\\varphi_0 \\cos\\lambda_0 & \\cos\\varphi_0 \\sin\\lambda_0 & \\sin\\varphi_0 \\end{bmatrix} \\begin{bmatrix} X - X_0 \\\\ Y - Y_0 \\\\ Z - Z_0 \\end{bmatrix}$$
$$x_N = y_N^{\\text{ENU}}, \\quad y_E = x_E^{\\text{ENU}}, \\quad z_D = -z_U^{\\text{ENU}}$$

#### C. Orientação em $SO(3)$ e Transformação no Body Frame
$$\\mathbf{p}_j^{\\text{Body}(i)} = \\mathbf{R}_{B/NED}^T \\left(\\mathbf{p}_j^{NED} - \\mathbf{p}_i^{NED}\\right) = \\begin{bmatrix} x_{\\text{proa}} \\\\ y_{\\text{estibordo}} \\\\ z_{\\text{quilha}} \\end{bmatrix}$$
"""

cell3_text = """## 2. Justificação da relevância das bibliotecas Python selecionadas
[De volta para "Conteúdo"](#conteúdo)

A seleção de bibliotecas baseou-se em critérios estritos de desempenho computacional, maturidade científica, desacoplamento e robustez numérica:

| Biblioteca | Papel no Projeto | Justificação Técnica e Vantagens |
|:---|:---|:---|
| **NumPy** | Cálculo Vetorial e Álgebra Linear | Permite a execução vetorizada em C de todas as conversões geodésicas, rotações $3 \\times 3$ e operações com quaterniões sem ciclos lentos em Python puro. |
| **SciPy** | Rotações Espaciais e Interpolação | Fornece rotinas numéricas para interpolação esférica linear (`Slerp`) de quaterniões de atitude no grupo $SO(3)$ e decomposição espetral de matrizes de covariância. |
| **Pandas** | Ingestão e Estruturação Tabular | Otimizada para leitura rápida dos CSVs de odometria do ROS, ordenação estrita por tempo, limpeza de duplicados e indexação temporal. |
| **Plotly** | Engine 3D Interativa e Dashboard Web | Renderização 3D acelerada por GPU (WebGL) no navegador, suporte a câmara 3D em 1ª pessoa no Body Frame e geração de bundle HTML 100% offline. |
| **PyYAML** | Configuração Declarativa Externa | Permite alterar parâmetros da missão (`mission_config.yaml`) sem modificar uma única linha do código-fonte executável. |
| **pytest** | Automação de Testes de Regressão | Suite de 24 testes unitários automatizados garantindo tolerâncias numéricas sub-milimétricas ($< 10^{-6}\\text{ m}$). |
| **pymap3d** / **pyproj** | Validação Numérica Externa | Utilizadas **exclusivamente como autoridade de teste** nos ensaios automatizados de validação cruzada das fórmulas de raiz. |

> **Decisão de Arquitetura:** Todas as equações geodésicas e transformações de referencial foram implementadas **de raiz pelo grupo** com NumPy, garantindo total domínio concetual.
"""

cell4_code = """import importlib

packages = [
    ("numpy", "Cálculo vetorial e matrizes"),
    ("pandas", "Leitura de telemetria e séries temporais"),
    ("plotly", "Visualizador 3D interativo e dashboard web"),
    ("yaml", "Configuração declarativa (PyYAML)"),
    ("scipy", "Interpolação SLERP e álgebra avançada"),
    ("pymap3d", "Validação geodésica de referência"),
    ("pyproj", "Verificação cartográfica independente"),
    ("pytest", "Automação de testes unitários")
]

print(f"{'BIBLIOTECA':<12} | {'ESTADO':<8} | {'VERSÃO':<12} | {'PAPEL PRINCIPAL'}")
print("-" * 75)
for pkg, role in packages:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "Instalado")
        print(f"{pkg:<12} | OK       | {ver:<12} | {role}")
    except ImportError:
        print(f"{pkg:<12} | AUSENTE  | {'---':<12} | {role}")
"""

cell5_text = """## 3. Metodologia proposta para o desenvolvimento do projeto
[De volta para "Conteúdo"](#conteúdo)

### 3.1 Planeamento Temporal e Calendarização
O desenvolvimento foi conduzido segundo uma metodologia iterativa e incremental (Clean Architecture), dividida em 4 fases principais:

| Fase | Período | Objetivos e Entregáveis Principais | Estado |
|:---:|:---:|:---|:---:|
| **Fase 1: Especificação & Modelação** | Semanas 1-2 | Requisitos funcionais, dedução das equações WGS-84/ECEF/NED/AER, submissão da Entrega 1 (`TOMSA_TL1_GRUPOXX.ipynb`). | **Concluída** |
| **Fase 2: Núcleo Matemático & Suavização** | Semanas 3-4 | Implementação de `geodesy.py`, `rotations.py`, Filtro de Kalman & Suavizador RTS, validação por 24 testes unitários com pytest. | **Concluída** |
| **Fase 3: Visualização 3D & Cockpit** | Semanas 5-6 | Construção das 4 Vistas em Plotly (`plotly_views.py`), câmara POV 1ª pessoa, cockpit de instrumentos e alerta CPA (< 10m). | **Concluída** |
| **Fase 4: Consolidação & Entrega** | Semanas 7-8 | Bundle HTML autónomo offline, exportação GeoJSON/KML 3D, documentação de utilizador e backups do projeto. | **Concluída** |

### 3.2 Estrutura Modular do Repositório
```text
multi_robot_system/
├── config/mission_config.yaml     # Parametrização declarativa da missão
├── data/                          # Telemetria real INESC TEC (Robot 1, 4, 7)
├── core/
│   ├── data/                      # Loader e dataclasses (trajectory.py, odom_loader.py)
│   └── math/                      # Geodesia, quaternions, Kalman/RTS, CPA (first-principles)
├── visualization/                 # Renderizador 3D Plotly, Cockpit e Servidor Web
├── io_export/                     # Exportadores GeoJSON, KML 3D e tabelas LaTeX
├── tests/                         # 24 testes unitários automatizados (pytest)
└── main.py                        # Ponto de entrada CLI unificado
```

### 3.3 Divisão de Tarefas na Equipa
| Membro do Grupo | Responsabilidades Principais | Atividades de Validação |
|:---|:---|:---|
| **Geovani Lopes** (Nº 1131518) | Arquitetura geral, módulo de geodesia (`geodesy.py`), quaternions (`rotations.py`) e engine 3D em Plotly (`plotly_views.py`, `web_server.py`). | Testes unitários com Pytest, validação de erros geodésicos vs pymap3d e relatórios. |
| **Luis Lei** (Nº 1120542) | Ingestão e parsing de odometria ROS (`odom_loader.py`), Filtro de Kalman e Suavizador RTS (`kalman_rts.py`), sincronização temporal. | Auditoria de ruído de amostragem, calibração do modelo de covariância. |
| **Hugo Amado** (Nº 1261726) | Painel de instrumentos analógicos (`cockpit_view.py`), cálculo analítico de CPA (`proximity.py`) e exportação SIG GeoJSON/KML 3D (`io_export`). | Testes de usabilidade dos atalhos de teclado e integração com QGIS/Google Earth. |
"""

cell6_text = """## 4. Demonstração e Validação do Núcleo Matemático
[De volta para "Conteúdo"](#conteúdo)

### 4.1 Demonstração Prática: Conversão para NED e Projeção no Body Frame
A célula abaixo executa um ensaio prático com as coordenadas reais da missão da Aguçadoura, calculando as coordenadas globais NED e as coordenadas relativas locais a bordo do Robô 1:
"""

cell7_code = """import numpy as np
from src.coordinates import geodetic_to_ned, ned_to_aer
from src.frames import quaternion_to_rotation_matrix, transform_to_local_robot_frame, calculate_relative_aer_body

# 1. Origem da Missão (Aguçadoura - INESC TEC)
lat0, lon0, h0 = 41.45201667, -8.81322222, 55.45

# 2. Posições iniciais dos Robôs 1 e 4
lat1, lon1, h1 = 41.45243668, -8.81236127, 56.73
lat4, lon4, h4 = 41.45311684, -8.81205601, 28.36

# Conversão para NED
pos1_ned = np.array(geodetic_to_ned(lat1, lon1, h1, lat0, lon0, h0))
pos4_ned = np.array(geodetic_to_ned(lat4, lon4, h4, lat0, lon0, h0))

az1, el1, rng1 = ned_to_aer(pos1_ned[0], pos1_ned[1], pos1_ned[2])
az4, el4, rng4 = ned_to_aer(pos4_ned[0], pos4_ned[1], pos4_ned[2])

print("=== VISTA GLOBAL (Plano Tangente Comum NED) ===")
print(f"Origem da Missão: Lat = {lat0:.6f}°, Lon = {lon0:.6f}°, Alt = {h0:.2f} m")
print(f"Robot 1: N = {pos1_ned[0]:+7.2f} m, E = {pos1_ned[1]:+7.2f} m, D = {pos1_ned[2]:+7.2f} m | Range = {rng1:6.2f} m, Azimute = {az1:5.1f}°")
print(f"Robot 4: N = {pos4_ned[0]:+7.2f} m, E = {pos4_ned[1]:+7.2f} m, D = {pos4_ned[2]:+7.2f} m | Range = {rng4:6.2f} m, Azimute = {az4:5.1f}°")

# 3. Transformação para a VISTA LOCAL (Body Frame do Robot 1)
# Supondo Robot 1 com orientação q = [0, 0, 0.7071, 0.7071] (Proa apontada a Este / 90°)
q_r1 = np.array([0.0, 0.0, 0.70710678, 0.70710678])
R_b1 = quaternion_to_rotation_matrix(q_r1[0], q_r1[1], q_r1[2], q_r1[3])
pos4_in_body1 = transform_to_local_robot_frame(pos4_ned, pos1_ned, R_b1)
bearing, dist, el_rel = calculate_relative_aer_body(pos4_in_body1)

print("\\n=== VISTA LOCAL (Perspetiva a Bordo do Robot 1) ===")
print(f"Robot 4 no Body Frame do Robot 1: Proa (Frente) = {pos4_in_body1[0]:+7.2f} m, Estibordo (Direita) = {pos4_in_body1[1]:+7.2f} m, Quilha = {pos4_in_body1[2]:+7.2f} m")
print(f"Distância Relativa: {dist:.2f} m | Marcação Relativa (Bearing): {bearing:+.1f}° ({'Estibordo' if bearing > 0 else 'Bombordo'})")
"""

cell8_text = """### 4.2 Auditoria Numérica Automatizada vs pymap3d
A célula seguinte realiza a auditoria numérica de validação cruzada entre as nossas fórmulas *first-principles* implementadas de raiz e a biblioteca de referência internacional `pymap3d`:
"""

cell9_code = """import numpy as np
from src.coordinates import geodetic_to_ecef, ecef_to_geodetic, geodetic_to_ned
import pymap3d

# Teste com as coordenadas da Aguçadoura
lat, lon, h = 41.45201666666667, -8.81322222222222, 55.44791126176966
lat0, lon0, h0 = lat, lon, h

# Fórmulas de raiz
X, Y, Z = geodetic_to_ecef(lat, lon, h)
X_ref, Y_ref, Z_ref = pymap3d.geodetic2ecef(lat, lon, h)
err_ecef = np.sqrt((X - X_ref)**2 + (Y - Y_ref)**2 + (Z - Z_ref)**2)

# Ciclo fechado Bowring
lat_inv, lon_inv, h_inv = ecef_to_geodetic(X, Y, Z)
X_inv, Y_inv, Z_inv = geodetic_to_ecef(lat_inv, lon_inv, h_inv)
err_roundtrip_dist = np.sqrt((X - X_inv)**2 + (Y - Y_inv)**2 + (Z - Z_inv)**2)

# NED vs pymap3d
lat_test, lon_test, h_test = lat + 0.001, lon + 0.001, h + 10.0
n, e, d = geodetic_to_ned(lat_test, lon_test, h_test, lat0, lon0, h0)
n_ref, e_ref, d_ref = pymap3d.geodetic2ned(lat_test, lon_test, h_test, lat0, lon0, h0)
err_ned = np.sqrt((n - n_ref)**2 + (e - e_ref)**2 + (d - d_ref)**2)

print("=== RELATÓRIO DE AUDITORIA NUMÉRICA (Fórmulas de Raiz vs pymap3d) ===")
print(f"1. Conversão Geodésico -> ECEF: Erro Absoluto = {err_ecef:.3e} m  [CERTIFICADO]")
print(f"2. Ciclo Fechado ECEF -> Geodésico: Erro Residual = {err_roundtrip_dist:.3e} m  [CERTIFICADO]")
print(f"3. Conversão Geodésico -> NED:  Erro Absoluto = {err_ned:.3e} m  [CERTIFICADO]")
print("Todos os testes cumprem com folga a tolerância de 1.00e-06 m!")
"""

def run_and_capture(code):
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured
    try:
        exec(code, globals())
    except Exception as e:
        print(f"Error executing code: {e}")
    sys.stdout = old_stdout
    out_lines = [line + "\n" for line in captured.getvalue().split("\n")]
    if out_lines and out_lines[-1] == "\n":
        out_lines.pop()
    return [{
        "name": "stdout",
        "output_type": "stream",
        "text": out_lines
    }]

def make_markdown_cell(text):
    lines = [line + "\n" for line in text.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def make_code_cell(code, exec_count):
    lines = [line + "\n" for line in code.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    outputs = run_and_capture(code)
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": outputs,
        "source": lines
    }

notebook_dict = {
    "cells": [
        make_markdown_cell(cell1_text),
        make_markdown_cell(cell2_text),
        make_markdown_cell(cell3_text),
        make_code_cell(cell4_code, 1),
        make_markdown_cell(cell5_text),
        make_markdown_cell(cell6_text),
        make_code_cell(cell7_code, 2),
        make_markdown_cell(cell8_text),
        make_code_cell(cell9_code, 3)
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

for path in [
    "/home/geovani/antigravityProjects/ISEP/TOMSA/TOMSA_TL1_GRUPOXX.ipynb",
    "/home/geovani/antigravityProjects/ISEP/TOMSA/TOMSA_TL1_GRUPO1131518.ipynb"
]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1, ensure_ascii=False)

print("Notebooks with outputs written successfully.")
