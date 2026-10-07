# TOMSA 2026 — Execution Guide & Command Reference
## Do Caminho ao Código Fonte (From Path to Source)

Este documento fornece o guia passo a passo completo, desde a navegação na árvore de diretórios até à ativação do ambiente e execução de todos os módulos e scripts do projeto **TOMSA 2026** (*Multi-Robot Trajectory Viewer & Mission Analyzer*).

---

## ⚡ Guia Rápido (TL;DR)

### Método 1: Execução Direta (Recomendado — 100% à Prova de Falhas)
Não depende do comando `source`, nem de aliases do sistema (`python` vs `python3`), funcionando em qualquer terminal Linux (bash, sh, zsh):

```bash
# 1. Navegar até ao caminho do projeto (Path)
cd /home/geovani/antigravityProjects/ISEP/TOMSA

# 2. Iniciar o Dashboard Web Interativo (Plotly com 4 Vistas e CPA)
PYTHONPATH=. .venv/bin/python3 multi_robot_system/main.py --web

# OU executar o pipeline com a GUI em Matplotlib:
PYTHONPATH=. .venv/bin/python3 main.py --visualize

# OU executar todos os testes unitários:
PYTHONPATH=. .venv/bin/pytest -v
```

---

### Método 2: Ativação Prévia do Ambiente Virtual
Se preferir ativar a sessão do terminal antes de executar os comandos:

```bash
# 1. Navegar até ao caminho do projeto
cd /home/geovani/antigravityProjects/ISEP/TOMSA

# 2. Ativar o ambiente virtual (utilize '.' se 'source' não existir na sua shell):
. .venv/bin/activate
# ou (em bash puro):
source .venv/bin/activate

# 3. Exportar a raiz do projeto
export PYTHONPATH=.

# 4. Executar os scripts (use python3 se 'python' não estiver mapeado no sistema):
python3 multi_robot_system/main.py --web
```
> 💡 **Nota sobre o erro `source: command not found`:** Em algumas distribuições Linux (Ubuntu/Debian) quando o terminal roda sob `/bin/sh` ou `dash`, o comando `source` não existe como comando embutido. Nesses casos, usa-se `. .venv/bin/activate` (ponto seguido de espaço) ou a **Execução Direta** do Método 1 acima.


## 🧭 Passo a Passo Detalhado: Do Caminho ao Código Fonte

### Passo 1: Caminho do Projeto e Navegação (`Path`)

Abra o seu terminal Linux / bash e navegue até à pasta onde o repositório reside:

```bash
# Navegar até à diretoria raiz do projeto
cd /home/geovani/antigravityProjects/ISEP/TOMSA

# Confirmar que se encontra no caminho correto
pwd
# Saída esperada: /home/geovani/antigravityProjects/ISEP/TOMSA

# Listar os ficheiros principais do projeto
ls -la
```

---

### Passo 2: Configuração e Ativação do Ambiente Virtual (`Source`)

O projeto utiliza um ambiente virtual Python para isolar as dependências numéricas e geoespaciais.

#### 2.1 Criar o ambiente virtual (caso ainda não exista)
```bash
python3 -m venv .venv
```

#### 2.2 Ativar o ambiente virtual com `source`
```bash
source .venv/bin/activate
```
> **Nota:** Após a execução, a linha de comandos do terminal exibirá o prefixo `(.venv)`.  
> Para desativar o ambiente quando terminar os trabalhos, basta digitar `deactivate`.

#### 2.3 Atualizar o gestor de pacotes e instalar dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Passo 3: Configurar o `PYTHONPATH`

Para garantir que o interpretador Python localiza todos os módulos internos (`src` e `multi_robot_system`) independentemente de onde o script for chamado:

```bash
export PYTHONPATH=.
```

---

## 🖥️ Comandos de Execução do Código Fonte (`Source Code`)

O repositório disponibiliza dois subsistemas complementares:
1. **Pipeline Principal (`main.py`)**: Interface clássica com GUI interativa Matplotlib, visualizador 3D Plotly em primeira pessoa, cálculo de CPA e exportação de figuras vetoriais para o relatório LaTeX.
2. **Sistema Modular Subaquático (`multi_robot_system/main.py`)**: Servidor Web multithreaded com Dashboard em Plotly (4 vistas táticas integradas, horizonte artificial, radar AER, filtro Kalman RTS, exportador KML 3D e GeoJSON).

---

### 1. Sistema Principal (`main.py`)

Localizado na raiz do projeto ([main.py](file:///home/geovani/antigravityProjects/ISEP/TOMSA/main.py)):

| Objetivo | Comando | Descrição |
|:---|:---|:---|
| **Validação Completa** | `python main.py --all` | Executa testes unitários, carrega os 3 robôs reais, calcula CPA, exporta CSV sincronizado, GeoJSON e figuras. |
| **GUI Interativa (Matplotlib)** | `python main.py --visualize` | Abre o reprodutor com Slider temporal, Play/Pausa, velocidades (10x, 25x, 50x, 100x), vista Global 2D, 3D e Radar Local. |
| **Visualizador 3D (Plotly Web)** | `python main.py --plotly` | Abre o visualizador 3D em primeira pessoa no navegador com suporte a atalhos de teclado. |
| **Teste Rápido 3D** | `python try.py` | Atalho direto para iniciar o visualizador 3D centrado no Body Frame. |
| **Exportar Figuras de Relatório** | `python main.py --export-figures` | Gera e grava todas as figuras vetoriais de alta resolução em `docs/figures/`. |
| **Exportar Relatórios e Métricas** | `python main.py --export-reports` | Gera `output/mission_metrics.json`, `output/synchronized_trajectories.csv` e `output/mission_trajectories.geojson`. |
| **Executar Testes Unitários** | `python main.py --test` | Executa a suite de testes de rigor matemático via pytest. |
| **Menu de Ajuda** | `python main.py --help` | Lista todas as opções e argumentos aceites pela CLI. |

---

### 2. Sistema Modular Subaquático (`multi_robot_system/main.py`)

Localizado em ([multi_robot_system/main.py](file:///home/geovani/antigravityProjects/ISEP/TOMSA/multi_robot_system/main.py)):

| Objetivo | Comando | Descrição |
|:---|:---|:---|
| **Dashboard Web Interativo** | `python multi_robot_system/main.py --web` | Inicia o servidor local multithreaded em `http://127.0.0.1:8050` e abre o navegador automaticamente. |
| **Pacote HTML Autónomo Offline** | `python multi_robot_system/main.py --bundle` | Gera `multi_robot_system/output/mission_dashboard_bundle.html` (abre em qualquer computador sem servidor nem Python). |
| **Benchmark de Interpoladores** | `python multi_robot_system/main.py --benchmark` | Compara Linear vs Spline Cúbica vs Kalman RTS e gera tabela comparativa em LaTeX. |
| **Exportação Geoespacial (SIG/KML)** | `python multi_robot_system/main.py --export` | Exporta `mission_trajectories_3d.kml` (Google Earth com gx:Track) e `mission_trajectories.geojson` (QGIS). |
| **Execução Integral do Subsistema** | `python multi_robot_system/main.py --all` | Executa testes, benchmark, exportações e inicia o servidor web. |
| **Testes Unitários Subaquáticos** | `python multi_robot_system/main.py --test` | Executa os 24 testes unitários com pytest. |
| **Menu de Ajuda** | `python multi_robot_system/main.py --help` | Exibe a lista de opções e argumentos da CLI. |

---

## 🧪 Execução dos Testes Unitários de Rigor Matemático (`pytest`)

O projeto dispõe de **50 testes unitários** automatizados com 100% de taxa de aprovação:

```bash
# Executar TODOS os testes unitários do projeto (50 testes)
pytest -v

# Executar apenas os testes de geodésia, ECEF, NED, ENU e AER (26 testes)
pytest tests/test_coordinates.py -v

# Executar apenas os testes do sistema subaquático (Kalman, RTS, SO(3), CPA) (24 testes)
pytest multi_robot_system/tests/ -v
```

---

## 📓 Execução do Caderno Jupyter (`Notebook`)

Para inspecionar, executar ou apresentar o caderno interativo da Entrega 1:

```bash
# Iniciar o servidor Jupyter
jupyter lab
# ou
jupyter notebook TOMSA_TL1_GRUPOXX.ipynb
```
O caderno contém todas as demonstrações matemáticas passo a passo com células interativas e gráficos embutidos.

---

## 📄 Compilação do Relatório Académico em LaTeX

O relatório científico completo (14 páginas, normas IEEE/ISEP) encontra-se em `docs/relatorio_tomsa.tex`.

```bash
# Compilar diretamente com o compilador Tectonic (sem necessidade de instalar TeX Live completo)
/home/geovani/.local/bin/tectonic docs/relatorio_tomsa.tex

# O PDF atualizado é gerado diretamente em:
# docs/relatorio_tomsa.pdf
```

---

## 🎮 Guia de Atalhos e Controlos Interativos

### No Dashboard Web (`multi_robot_system/main.py --web`)
* **`1`**: Seleciona o **Robot 1** (AUV Alpha) como observador ativo.
* **`2`**: Seleciona o **Robot 4** (AUV Bravo) como observador ativo.
* **`3`**: Seleciona o **Robot 7** (AUV Charlie) como observador ativo.
* **`V`**: Alterna ciclicamente entre as 4 vistas:
  1. *Global 3D*: Visão externa com batimetria e trilhas coloridas.
  2. *POV (1ª Pessoa)*: Câmara a partir da proa do robô ativo.
  3. *Cockpit de Controlo*: Horizonte artificial, bússola, profundidade, velocímetro e radar AER.
  4. *Distâncias CPA 2D*: Gráfico temporal de proximidade com limiar crítico (< 10 m).
* **`Espaço`**: Play / Pausa na animação temporal.
* **`R`**: Reinicia a reprodução das trajetórias para o início da missão ($t = 0\text{ s}$).
* **Sliders de Velocidade**: Seleciona velocidades de reprodução desde `0.25x` até `⚡ Ultra` (~2 segundos para toda a missão).

### Na Interface Matplotlib (`python main.py --visualize`)
* **Slider Temporal inferior**: Arraste para inspecionar qualquer instante da missão de 40 minutos.
* **Botões de Reprodução**: `Play`, `Pause`, `Reset`.
* **Botões de Velocidade**: `10x`, `25x`, `50x`, `100x`.
* **RadioButtons de Seleção**: Alterne o referencial local entre Robot 1, 4 ou 7.

---

## 📁 Onde Encontrar os Ficheiros Gerados (`Output Artifacts`)

Após a execução dos comandos de exportação, os resultados são guardados nas seguintes pastas:

```text
output/
├── mission_execution.log           # Registo de eventos com timestamps
├── mission_metrics.json            # Estatísticas, distâncias mínimas e CPA
├── synchronized_trajectories.csv   # Série temporal unificada a dt = 1.0s
├── mission_trajectories.geojson    # Trajetórias prontas para QGIS / SIG
├── mission_trajectories_3d.kml     # Trajetórias 3D com gx:Track para Google Earth
└── mission_body_view_3d.html       # Visualizador Web 3D em HTML autónomo

multi_robot_system/output/
├── mission_dashboard_bundle.html   # Dashboard completo com 4 vistas 100% offline
├── interpolator_latex_table.tex    # Tabela comparativa gerada para o relatório LaTeX
├── synchronized_trajectories.csv   # Dados sincronizados com suavização Kalman/RTS
├── mission_trajectories.geojson    # GeoJSON do subsistema unificado
└── mission_trajectories_3d.kml     # KML 3D com marcas de tempo

docs/figures/
├── mission_trajectories_2d.pdf     # Vista Global 2D em alta resolução
├── mission_trajectories_3d.pdf     # Vista Global 3D em alta resolução
├── local_body_view.pdf             # Vista Local e radar AER em alta resolução
└── distance_matrix.pdf             # Matriz de distâncias e análise de CPA
```

---

## 🔧 Resolução de Problemas Comuns (Troubleshooting)

| Sintoma | Causa Provável | Solução |
|:---|:---|:---|
| `source: command not found` | A shell atual (ex.: `/bin/sh` ou `dash`) não possui o comando `source`. | Use `. .venv/bin/activate` (ponto seguido de espaço), abra uma subshell bash (`bash`), ou use o método de **Execução Direta**: `PYTHONPATH=. .venv/bin/python3 ...` |
| `Command 'python' not found` | O ambiente virtual não foi ativado e o Ubuntu não possui o alias legado `python`. | Utilize `python3` (com venv ativado) ou invoque diretamente o executável do venv: `.venv/bin/python3`. |
| `ModuleNotFoundError: No module named 'src'` ou `'multi_robot_system'` | A variável `PYTHONPATH` não está configurada para a raiz do projeto. | Execute `export PYTHONPATH=.` no terminal ou prefixe o comando: `PYTHONPATH=. .venv/bin/python3 ...` |
| `Address already in use: 8050` | Outra instância do servidor web já está a correr no porto 8050. | Termine o processo anterior com `fuser -k 8050/tcp` ou altere a porta no ficheiro de configuração YAML. |
| `FileNotFoundError: config/config.yaml` | O comando foi executado fora da pasta raiz do projeto. | Execute `cd /home/geovani/antigravityProjects/ISEP/TOMSA` antes de iniciar. |

