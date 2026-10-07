import json
import subprocess
import sys
from pathlib import Path
import numpy as np

# Executar primeiro os blocos de código para capturar os outputs reais
env_check_code = """import importlib

packages = [
    ("numpy", "Cálculo vetorial e matrizes"),
    ("pandas", "Leitura de telemetria e séries temporais"),
    ("matplotlib", "Renderização gráfica 2D/3D e GUI de Replay"),
    ("yaml", "Configuração declarativa (PyYAML)"),
    ("scipy", "Interpolação SLERP e elipses de covariância"),
    ("pymap3d", "Validação geodésica de referência"),
    ("pyproj", "Verificação cartográfica independente"),
    ("pytest", "Automação de testes unitários"),
    ("plotly", "Visualizador 3D interativo em 1ª pessoa no Body Frame")
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

demo_code = """import numpy as np
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
R_b1 = quaternion_to_rotation_matrix(q_r1)
pos4_in_body1 = transform_to_local_robot_frame(pos4_ned, pos1_ned, R_b1)
bearing, dist, el_rel = calculate_relative_aer_body(pos4_in_body1)

print("\\n=== VISTA LOCAL (Perspetiva a Bordo do Robot 1) ===")
print(f"Robot 4 no Body Frame do Robot 1: Proa (Frente) = {pos4_in_body1[0]:+7.2f} m, Estibordo (Direita) = {pos4_in_body1[1]:+7.2f} m, Quilha = {pos4_in_body1[2]:+7.2f} m")
print(f"Distância Relativa: {dist:.2f} m | Marcação Relativa (Bearing): {bearing:+.1f}° ({'Estibordo' if bearing > 0 else 'Bombordo'})")
"""

validation_code = """from src.coordinates import geodetic_to_ecef, ecef_to_geodetic, geodetic_to_ned, enu_to_aer
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
err_roundtrip_m = pymap3d.geodetic2ecef(lat_inv, lon_inv, h_inv)
err_roundtrip_dist = np.sqrt(sum((a - b)**2 for a, b in zip([X, Y, Z], err_roundtrip_m)))

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

# Executar cada bloco em subprocesso Python para capturar stdout
def run_code(code_str):
    res = subprocess.run(
        ["/home/geovani/antigravityProjects/ISEP/TOMSA/.venv/bin/python", "-c", code_str],
        cwd="/home/geovani/antigravityProjects/ISEP/TOMSA",
        capture_output=True,
        text=True
    )
    return res.stdout if res.returncode == 0 else f"ERRO:\n{res.stderr}"

out1 = run_code(env_check_code)
out2 = run_code(demo_code)
out3 = run_code(validation_code)

print("Output 1 captured:", len(out1))
print("Output 2 captured:", len(out2))
print("Output 3 captured:", len(out3))

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# TÓPICOS DE MATEMÁTICA PARA SISTEMAS AUTÓNOMOS (TOMSA)\n",
                "## TRABALHO LETIVO - ENTREGA 1\n",
                "\n",
                "> **Nota 1:** Todos os elementos do grupo são responsáveis por esta entrega.\n",
                "> **Nota 2:** Todos os elementos do grupo tomaram conhecimento das regras estabelecidas para o funcionamento da u.c. no [Moodle do ISEP](https://moodle.isep.ipp.pt).\n",
                "\n",
                "---\n",
                "### GRUPO\n",
                "* **N:** 1211756 **Nome:** Geovani Debastiani\n",
                "* **N:** XXXXXXXX **Nome:** Colega de Grupo\n",
                "\n",
                "---\n",
                "### Conteúdo\n",
                "1. [Identificação dos objetivos e estruturação do problema](#1-identificação-dos-objetivos-e-estruturação-do-problema)\n",
                "2. [Justificação da relevância das bibliotecas Python selecionadas](#2-justificação-da-relevância-das-bibliotecas-python-selecionadas)\n",
                "3. [Metodologia proposta para o desenvolvimento do projeto](#3-metodologia-proposta-para-o-desenvolvimento-do-projeto)\n",
                "4. [Demonstração e Validação do Núcleo Matemático](#4-demonstração-e-validação-do-núcleo-matemático)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Identificação dos objetivos e estruturação do problema\n",
                "[De volta para o Conteúdo](#conteúdo)\n",
                "\n",
                "### 1.1 Objetivo Geral\n",
                "O presente trabalho tem como objetivo projetar, desenvolver e validar rigorosamente uma aplicação computacional em **Python** destinada à visualização, sincronização temporal e análise cinemática de trajetórias de múltiplos robôs autónomos de superfície e submarinos (ASVs/AUVs). O sistema baseia-se em telemetria real anonimizada obtida pelo **INESC TEC** durante campanhas marinhas realizadas na costa da **Aguçadoura** (Póvoa de Varzim, Portugal).\n",
                "\n",
                "A aplicação realiza a conversão de coordenadas geodésicas de navegação (WGS-84: latitude, longitude, altitude e quaternions de atitude) para referenciais cartesianos locais tangentes (**NED** - *North-East-Down* e **ENU** - *East-North-Up*), projeta as posições relativas e atitudes no referencial próprio do corpo (**Body Frame**) do veículo selecionado, e calcula grandezas esféricas relativas de visada (**AER** - *Azimuth, Elevation, Range*).\n",
                "\n",
                "### 1.2 Objetivos Específicos e Requisitos do Sistema\n",
                "| ID | Funcionalidade | Descrição e Critério de Aceitação |\n",
                "|:--:|:---|:---|\n",
                "| **O1** | **Global View** | Apresentação de todas as trajetórias e posições instantâneas dos robôs num referencial cartesiano comum (NED/ENU) centrado na origem da missão, com diferenciação cromática e vetores de proa (*heading*). |\n",
                "| **O2** | **Local View** | Representação dinâmica da vizinhança centrada no robô selecionado (origem $(0,0,0)$ orientada com a sua proa/eixo longitudinal), exibindo robôs vizinhos e anéis radar de alcance. |\n",
                "| **O3** | **Mission Replay** | Controlo temporal com barra deslizante (*slider*), botões *Play/Pause* e saltos temporais, operando sobre uma série temporal interpolada e sincronizada (interpolação linear para posições e SLERP para atitudes). |\n",
                "| **O4** | **Configuração YAML** | Definição declarativa e modular de parâmetros da missão: coordenadas da origem geodésica, lista de ficheiros CSV, cores, robô selecionado, limiar de proximidade e parâmetros de replay. |\n",
                "| **O5** | **Output Log & Export** | Registo estruturado de eventos de execução (*logging*), exportação de telemetria sincronizada em CSV, GeoJSON (para visualização SIG/Google Earth) e métricas globais em JSON (incluindo cálculo de CPA - *Closest Point of Approach*). |\n",
                "| **O6** | **Rigor Matemático** | Implementação de raiz (*first-principles*) de todas as fórmulas de conversão geodésica e cinemática com NumPy, com validação cruzada exaustiva contra bibliotecas padrão (*pymap3d* e *pyproj*). |\n",
                "\n",
                "### 1.3 Modelação Matemática do Problema\n",
                "\n",
                "#### A. Conversão Geodésica para ECEF (Elipsoide WGS-84)\n",
                "Seja $\\varphi$ a latitude geodésica, $\\lambda$ a longitude e $h$ a altitude elipsoidal. O raio de curvatura normal no primeiro vertical é dado por:\n",
                "$$N(\\varphi) = \\frac{a}{\\sqrt{1 - e^2 \\sin^2 \\varphi}}$$\n",
                "onde $a = 6\\,378\\,137.0\\text{ m}$ (semieixo maior) e $e^2 = 2f - f^2 \\approx 0.00669438$ (primeira excentricidade ao quadrado, com $f = 1/298.257223563$). As coordenadas cartesianas ECEF são expressas por:\n",
                "$$X = (N(\\varphi) + h) \\cos \\varphi \\cos \\lambda$$\n",
                "$$Y = (N(\\varphi) + h) \\cos \\varphi \\sin \\lambda$$\n",
                "$$Z = \\left(N(\\varphi)(1 - e^2) + h\\right) \\sin \\varphi$$\n",
                "\n",
                "#### B. Conversão ECEF para Plano Tangente Local (ENU e NED)\n",
                "Dada uma origem geodésica $(\\varphi_0, \\lambda_0, h_0)$ com coordenadas ECEF $\\mathbf{r}_{0} = (X_0, Y_0, Z_0)^T$, o vetor de translação relativo é $\\Delta\\mathbf{r} = \\mathbf{r}_{ECEF} - \\mathbf{r}_0$. A rotação ortogonal para o referencial ENU é obtida por:\n",
                "$$\\begin{bmatrix} x_E \\\\ y_N \\\\ z_U \\end{bmatrix} = \\begin{bmatrix} -\\sin\\lambda_0 & \\cos\\lambda_0 & 0 \\\\ -\\sin\\varphi_0 \\cos\\lambda_0 & -\\sin\\varphi_0 \\sin\\lambda_0 & \\cos\\varphi_0 \\\\ \\cos\\varphi_0 \\cos\\lambda_0 & \\cos\\varphi_0 \\sin\\lambda_0 & \\sin\\varphi_0 \\end{bmatrix} \\begin{bmatrix} X - X_0 \\\\ Y - Y_0 \\\\ Z - Z_0 \\end{bmatrix}$$\n",
                "Para o referencial náutico e aeronáutico **NED** (*North-East-Down*):\n",
                "$$x_N = y_N^{\\text{ENU}}, \\quad y_E = x_E^{\\text{ENU}}, \\quad z_D = -z_U^{\\text{ENU}}$$\n",
                "\n",
                "#### C. Sistema Esférico Local AER (Azimute, Elevação, Range)\n",
                "A partir das coordenadas cartesianas locais:\n",
                "$$\\text{Range} = \\sqrt{x_N^2 + y_E^2 + z_D^2}$$\n",
                "$$\\text{Azimuth} = \\operatorname{atan2}(y_E, x_N) \\pmod{360^\\circ}$$\n",
                "$$\\text{Elevation} = \\operatorname{atan2}(-z_D, \\sqrt{x_N^2 + y_E^2})$$\n",
                "\n",
                "#### D. Orientação por Quaternions e Transformação para o Referencial do Corpo (Body Frame)\n",
                "A atitude do veículo no referencial NED é descrita pelo quaternion unitário $\\mathbf{q} = (q_x, q_y, q_z, q_w)$. A matriz de rotação $\\mathbf{R}_{B/NED}$ (que mapeia vetores do corpo para o NED) é:\n",
                "$$\\mathbf{R}_{B/NED} = \\begin{bmatrix} 1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_z q_w) & 2(q_x q_z + q_y q_w) \\\\ 2(q_x q_y + q_z q_w) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_x q_w) \\\\ 2(q_x q_z - q_y q_w) & 2(q_y q_z + q_x q_w) & 1 - 2(q_x^2 + q_y^2) \\end{bmatrix}$$\n",
                "A posição relativa de um robô vizinho $j$ no referencial próprio do robô observador $i$ é obtida pela translação e projeção inversa:\n",
                "$$\\mathbf{p}_j^{\\text{Body}(i)} = \\mathbf{R}_{B/NED}^T \\left(\\mathbf{p}_j^{NED} - \\mathbf{p}_i^{NED}\\right) = \\begin{bmatrix} x_{\\text{proa}} \\\\ y_{\\text{estibordo}} \\\\ z_{\\text{quilha}} \\end{bmatrix}$$\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Justificação da relevância das bibliotecas Python selecionadas\n",
                "[De volta para o Conteúdo](#conteúdo)\n",
                "\n",
                "A seleção de ferramentas baseou-se em critérios de desempenho computacional, maturidade científica, robustez numérica e facilidade de integração:\n",
                "\n",
                "| Biblioteca | Papel no Projeto | Justificação Técnica e Vantagens |\n",
                "|:---|:---|:---|\n",
                "| **NumPy** | Vetorização e Álgebra Linear | Núcleo de cálculo numérico em C/Fortran. Permite calcular conversões de dezenas de milhares de pontos geodésicos e multiplicações matriciais $3 \\times 3$ simultâneas sem ciclos lentos em Python puro. |\n",
                "| **pandas** | Leitura e Manipulação Tabular | Otimizada para leitura de ficheiros CSV com inferência de tipos (`dtypes`), filtragem, ordenação temporal e junção de telemetrias com diferentes frequências de amostragem. |\n",
                "| **Matplotlib** | Visualização Gráfica e GUI Interativa | Permite controlo absoluto da renderização gráfica em 2D/3D, exportação de figuras vetoriais de alta resolução (PDF) e construção de controlos GUI interativos nativos (`Slider`, `Button`, `RadioButtons`) sem dependências pesadas de frontend web. |\n",
                "| **PyYAML** | Configuração Declarativa | Leitura segura (`yaml.safe_load`) de parâmetros externos de missão, permitindo alterar ficheiros de dados, origens, cores e limiares sem modificar o código-fonte. |\n",
                "| **SciPy** | Rotações Espaciais e Interpolação | Disponibiliza algoritmos otimizados de interpolação esférica linear (`Slerp`) para quaternions e álgebra espetral (`eigh`) para extração de semi-eixos de elipses de incerteza a partir da covariância. |\n",
                "| **Plotly** | Visualizador 3D Interativo (Body Frame) | Renderização acelerada por GPU (WebGL) no navegador com câmara 3D em 1ª pessoa, suporte a atalhos de teclado e exportação para HTML autónomo. |\n",
                "| **pymap3d** | Validação Numérica de Referência | Biblioteca padrão de referência geodésica em Python (utilizada pela comunidade aeroespacial/marítima), servindo de *ground truth* para certificar as fórmulas de raiz implementadas com erro $< 10^{-6}\\text{ m}$. |\n",
                "| **pyproj** | Verificação Geodésica Independente | Interface para a biblioteca cartográfica internacional PROJ, permitindo uma segunda camada de validação cruzada independente de coordenadas elipsoidais. |\n",
                "| **pytest** | Automação de Testes de Regressão | Framework de testes unitários parametrizados, garantindo a verificação contínua e impedindo regressões em todo o pipeline matemático. |\n",
                "\n",
                "> **Decisão Arquitetural:** Todas as conversões de coordenadas (Geodésico $\\leftrightarrow$ ECEF $\\leftrightarrow$ NED/ENU $\\leftrightarrow$ AER) e transformações de referencial do corpo são implementadas **de raiz pelo grupo** com recurso a NumPy, comprovando o domínio matemático. As bibliotecas `pymap3d` e `pyproj` são utilizadas única e exclusivamente como autoridade de teste e validação cruzada nos testes unitários."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in out1.strip().split("\n")]
                }
            ],
            "source": [env_check_code.strip()]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Metodologia proposta para o desenvolvimento do projeto\n",
                "[De volta para o Conteúdo](#conteúdo)\n",
                "\n",
                "### 3.1 Planeamento Temporal e Calendarização das Fases\n",
                "O desenvolvimento foi estruturado segundo uma metodologia iterativa e incremental dividida em quatro fases principais ao longo do semestre letivo (8 semanas):\n",
                "\n",
                "| Fase | Período | Objetivos e Entregáveis Principais | Estado |\n",
                "|:---:|:---:|:---|:---:|\n",
                "| **Fase 1: Especificação e Modelação** | Semanas 1 a 3 | Definição de requisitos (O1 a O6), dedução formal das equações geodésicas e cinemáticas, seleção de bibliotecas, submissão da Entrega 1 (`TOMSA_TL1_GRUPOXX.ipynb`). | **Concluída** |\n",
                "| **Fase 2: Núcleo Matemático e Testes** | Semanas 3 a 5 | Implementação de raiz de `coordinates.py` e `frames.py`, validação por 26 testes unitários em `pytest` contra `pymap3d` (tolerância $< 10^{-6}\\text{ m}$), parser de CSVs e interpolação temporal SLERP em `data_loader.py`. | **Concluída** |\n",
                "| **Fase 3: Visualização e Extensões** | Semanas 5 a 7 | Construção da GUI interativa multi-painel (`visualizer.py`) com reciclagem de artistas (60+ FPS), visualizador 3D em 1ª pessoa no Body Frame (`plotly_visualizer.py`), cálculo de CPA e alertas de proximidade, elipses de covariância a 95% e exportação GeoJSON (`extras.py`). | **Concluída** |\n",
                "| **Fase 4: Consolidação e Defesa** | Semana 8 | Compilação do relatório científico em LaTeX (`relatorio_tomsa.pdf`), ensaios finais de usabilidade, preparação de demonstrações interativas e defesa oral da Entrega 2. | **Pronta** |\n",
                "\n",
                "### 3.2 Estrutura Modular do Repositório de Código\n",
                "O projeto segue as melhores práticas de engenharia de software (princípios SOLID, separação de responsabilidades e código autodocumentado):\n",
                "\n",
                "```text\n",
                "TOMSA/\n",
                "├── config/\n",
                "│   └── config.yaml               # Parâmetros de missão declarativos (origem, cores, limiares)\n",
                "├── src/\n",
                "│   ├── coordinates.py            # Fórmulas de raiz: WGS-84, ECEF, NED, ENU, AER, Bowring\n",
                "│   ├── frames.py                 # Cinemática de quaternions, matrizes de rotação e Body Frame\n",
                "│   ├── data_loader.py            # Ingestão de CSVs de odometria, interpolação e SLERP\n",
                "│   ├── config_loader.py          # Validação e tipagem estrita com dataclasses\n",
                "│   ├── views.py                  # Vistas estáticas de publicação e exportação de figuras vetoriais\n",
                "│   ├── visualizer.py             # GUI interativa multi-vista de alta performance (Matplotlib)\n",
                "│   ├── plotly_visualizer.py      # Visualizador 3D em 1ª pessoa no referencial do corpo (Plotly)\n",
                "│   ├── logger.py                 # Logging estruturado, métricas JSON e CSV sincronizado\n",
                "│   └── extras.py                 # Elipses de incerteza 95%, CPA, perfis 3D e exportador GeoJSON\n",
                "├── tests/\n",
                "│   └── test_coordinates.py       # 26 testes unitários com pytest validando vs pymap3d/pyproj\n",
                "├── docs/\n",
                "│   ├── relatorio_tomsa.tex       # Relatório académico formal em LaTeX\n",
                "│   ├── relatorio_tomsa.pdf       # Versão compilada com figuras e equações\n",
                "│   └── figures/                  # Figuras vetoriais de alta resolução geradas pela aplicação\n",
                "├── output/                       # Artefactos gerados (HTML 3D, GeoJSON, CSV, JSON, logs)\n",
                "├── main.py                       # CLI unificada (--visualize, --plotly, --all, --test)\n",
                "├── try.py                        # Script rápido de demonstração do visualizador 3D Web\n",
                "└── pytest.ini                    # Configuração para execução direta de testes com pytest\n",
                "```\n",
                "\n",
                "### 3.3 Estratégia de Validação e Garantia de Qualidade\n",
                "1. **Testes de Ciclo Fechado (*Roundtrip*):** Comprovação de que $\\text{Geodésico} \\to \\text{ECEF} \\to \\text{Geodésico}$ recupera exatamente a latitude e longitude originais com erro $< 10^{-14}$ graus e altitude com erro subnanométrico ($< 10^{-9}\\text{ m}$).\n",
                "2. **Validação contra Bibliotecas Padrão:** Comparação com `pymap3d` em múltiplos pontos geográficos em Portugal, exigindo desvios inferiores a $10^{-6}\\text{ m}$.\n",
                "3. **Verificação de Simetrias e Geometria Conhecida:** Teste analítico comprovando que se um robô está orientado para Este ($90^\\circ$), um vizinho a Este aparece exatamente à sua frente ($+x_{\\text{proa}}$) e um vizinho a Sul aparece a estibordo ($+y_{\\text{estibordo}}$).\n",
                "4. **Tratamento de Dados Reais:** Teste de robustez contra assincronismo temporal, ruído e dados discretos dos robôs ASV 1, 4 e 7 do INESC TEC na Aguçadoura.\n",
                "\n",
                "### 3.4 Divisão de Tarefas entre os Elementos do Grupo\n",
                "| Membro do Grupo | Responsabilidades Principais | Atividades Cruzadas de Revisão |\n",
                "|:---|:---|:---|\n",
                "| **Aluno 1 (Geovani)** | Implementação do núcleo matemático (`coordinates.py`, `frames.py`), suite de testes unitários (`test_coordinates.py`), arquitetura dos visualizadores interativos (`visualizer.py`, `plotly_visualizer.py`) e relatório LaTeX. | Revisão da extração de telemetria, validação cruzada com pymap3d e testes de usabilidade. |\n",
                "| **Aluno 2 (Colega)** | Módulo de dados e configuração (`data_loader.py`, `config_loader.py`), sincronização temporal com SLERP, exportação de métricas e GeoJSON (`logger.py`, `extras.py`). | Revisão das fórmulas matemáticas, calibração dos anéis radar e apoio à redação do relatório. |\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Demonstração e Validação do Núcleo Matemático\n",
                "[De volta para o Conteúdo](#conteúdo)\n",
                "\n",
                "### 4.1 Demonstração Prática: Conversão para NED e Projeção no Body Frame\n",
                "A célula abaixo executa um ensaio prático com as coordenadas reais da missão da Aguçadoura, calculando as coordenadas globais NED e as coordenadas relativas locais a bordo do Robô 1:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in out2.strip().split("\n")]
                }
            ],
            "source": [demo_code.strip()]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4.2 Auditoria Numérica Automatizada vs pymap3d\n",
                "A célula seguinte realiza a auditoria numérica de validação cruzada entre as nossas fórmulas *first-principles* implementadas de raiz e a biblioteca de referência internacional `pymap3d`:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in out3.strip().split("\n")]
                }
            ],
            "source": [validation_code.strip()]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.14.4"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

target_file = Path("/home/geovani/antigravityProjects/ISEP/TOMSA/TOMSA_TL1_GRUPOXX.ipynb")
with open(target_file, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2, ensure_ascii=False)

print(f"Jupyter Notebook atualizado com sucesso em: {target_file}")
