# TOMSA 2026 - Multi-Robot Trajectory Viewer & Analyzer

**Instituto Superior de Engenharia do Porto (ISEP)**  
**Unidade Curricular:** Tópicos de Matemática para Sistemas Autónomos (TOMSA)  
**Campanha Experimental:** Missão Marinha INESC TEC na Aguçadoura (Robôs ASV 1, 4 e 7)  
**Autor:** Geovani Debastiani (Nº 1211756)

---

## 📌 Visão Geral do Projeto

Este repositório contém a implementação completa, rigorosamente testada e documentada, do sistema de visualização, análise cinemática e sincronização temporal de múltiplos robôs autónomos de superfície e submarinos (ASVs/AUVs).

O sistema processa ficheiros reais de odometria do ROS (`robot1_s1.csv`, `robot4_s1.csv` e `robot7_s1.csv`), convertendo as coordenadas geodésicas (WGS-84) e atitudes (quaternions) para planos tangentes locais (**NED** e **ENU**), para o referencial próprio de cada robô (**Body Frame**) e para grandezas esféricas relativas (**AER**).

---

## 📐 A Matemática Descomplicada

### 1. Modelo da Terra e Geodésico WGS-84
A Terra é modelada como um elipsoide de revolução achatado nos pólos:
* Semieixo maior: $a = 6\,378\,137.0\text{ m}$
* Achatamento: $f = 1 / 298.257223563$
* Primeira excentricidade ao quadrado: $e^2 = 2f - f^2 \approx 0.00669438$
* **Raio Normal de Curvatura:**
  $$N(\varphi) = \frac{a}{\sqrt{1 - e^2 \sin^2 \varphi}}$$

### 2. Conversão Geodésico $\to$ ECEF (Earth-Centered, Earth-Fixed)
Translação tridimensional para a origem no centro de massa da Terra:
$$X = (N(\varphi) + h) \cos\varphi \cos\lambda$$
$$Y = (N(\varphi) + h) \cos\varphi \sin\lambda$$
$$Z = (N(\varphi)(1 - e^2) + h) \sin\varphi$$

### 3. Conversão ECEF $\to$ Plano Tangente Local (NED e ENU)
Com origem na referência inicial da Aguçadoura $(\varphi_0, \lambda_0, h_0)$, com $\Delta\mathbf{r} = \mathbf{r}_{ECEF} - \mathbf{r}_0$:
$$\begin{bmatrix} x_E \\ y_N \\ z_U \end{bmatrix} = \begin{bmatrix} -\sin\lambda_0 & \cos\lambda_0 & 0 \\ -\sin\varphi_0 \cos\lambda_0 & -\sin\varphi_0 \sin\lambda_0 & \cos\varphi_0 \\ \cos\varphi_0 \cos\lambda_0 & \cos\varphi_0 \sin\lambda_0 & \sin\varphi_0 \end{bmatrix} \Delta\mathbf{r}$$
No referencial naval/aeronáutico **NED** (*North-East-Down*):
$$x_N = y_N^{\text{ENU}}, \quad y_E = x_E^{\text{ENU}}, \quad z_D = -z_U^{\text{ENU}}$$

### 4. Coordenadas Esféricas AER (Azimute, Elevação, Range)
$$\text{Range} = \sqrt{x_N^2 + y_E^2 + z_D^2}, \quad \text{Azimuth} = \operatorname{atan2}(y_E, x_N) \pmod{360^\circ}, \quad \text{Elevation} = \operatorname{atan2}(-z_D, \sqrt{x_N^2 + y_E^2})$$

### 5. Atitude por Quaternions e Referencial do Corpo (Body Frame)
A atitude do robô é descrita pelo quaternion unitário $\mathbf{q} = (q_x, q_y, q_z, q_w)$, cuja Matriz de Cossenos Diretores $\mathbf{R}_{B/NED}$ projeta vetores do corpo no NED. A projeção de um robô vizinho $j$ no referencial do robô $i$ é:
$$\mathbf{p}_j^{\text{Body}(i)} = \mathbf{R}_{B/NED}^T (\mathbf{p}_j^{NED} - \mathbf{p}_i^{NED}) = \begin{bmatrix} x_{\text{proa}} \\ y_{\text{estibordo}} \\ z_{\text{quilha}} \end{bmatrix}$$
* $x_{\text{proa}} > 0$: Alvo à frente
* $y_{\text{estibordo}} > 0$: Alvo à direita (estibordo)
* Marcação Relativa (*Relative Bearing*): $\beta = \operatorname{atan2}(y_{\text{estibordo}}, x_{\text{proa}})$

---

## 📦 Bibliotecas Utilizadas e Justificação Técnica

| Biblioteca | Papel no Projeto | Justificação |
|:---|:---|:---|
| **NumPy** | Vetorização e Álgebra Linear | Cálculo matricial ultrarrápido em C/Fortran para dezenas de milhares de pontos sem ciclos Python lentos. |
| **pandas** | Leitura e Manipulação Tabular | Ingestão robusta dos ficheiros CSV de odometria do ROS e ordenação por timestamp. |
| **Matplotlib** | Visualização e GUI Interativa | Renderização de alta fidelidade das Vistas Global e Local, além de controlos GUI (`Slider`, `Button`, `RadioButtons`) sem servidores web pesados. |
| **PyYAML** | Configuração Declarativa | Separação estrita entre parâmetros de missão (`config.yaml`) e código-fonte executável. |
| **SciPy** | Interpolação SLERP e Covariância | Interpolação esférica contínua de atitude (`Slerp`) e decomposição espetral (`eigh`) de matrizes de incerteza. |
| **pymap3d** / **pyproj** | Autoridade de Validação | Utilizadas exclusivamente como *ground truth* nos testes unitários para certificar a precisão das nossas fórmulas de raiz ($< 10^{-9}\text{ m}$). |
| **pytest** | Automação de Testes | Execução sistemática de asserções numéricas para impedir regressões. |

---

## 🗂️ Estrutura do Repositório

```text
TOMSA/
├── config/
│   └── config.yaml               # Configuração declarativa da missão (origem, cores, robôs, limiares)
├── data/                         # Ficheiros CSV reais de odometria do ROS (INESC TEC Aguçadoura)
│   ├── robot1_s1.csv             # Telemetria ASV Alpha (Robot 1)
│   ├── robot4_s1.csv             # Telemetria ASV Bravo (Robot 4)
│   └── robot7_s1.csv             # Telemetria ASV Charlie (Robot 7)
├── src/
│   ├── __init__.py
│   ├── coordinates.py            # Fórmulas de raiz (WGS-84, ECEF, NED, ENU, AER, Bowring)
│   ├── frames.py                 # Quaternions, matrizes de rotação e projeção no Body Frame
│   ├── data_loader.py            # Leitura de CSVs, sincronização temporal e interpolação SLERP
│   ├── config_loader.py          # Validação e carregamento de configurações com Dataclasses
│   ├── views.py                  # Renderização gráfica de Global View, Local View e AER
│   ├── visualizer.py             # GUI interativa com Replay, Slider temporal e reciclagem de artistas
│   ├── plotly_visualizer.py      # Visualizador 3D em 1ª pessoa no Body Frame (WebGL / Plotly)
│   ├── logger.py                 # Registo de eventos, métricas de missão e exportação CSV
│   └── extras.py                 # Elipses de covariância 95%, CPA, 3D e exportador GeoJSON
├── multi_robot_system/           # Subsistema unificado subaquático (Plotly Dashboard & Kalman RTS)
│   ├── config/                   # Configurações do subsistema web
│   ├── core/                     # Geodésia, SO(3), filtros Kalman/RTS e CPA analítico
│   ├── visualization/            # Vistas 3D, POV 1ª pessoa, Cockpit e servidor Web
│   ├── io_export/                # Exportadores KML 3D, GeoJSON e gerador de tabelas LaTeX
│   ├── tests/                    # 24 testes unitários automatizados
│   └── main.py                   # Ponto de entrada CLI do subsistema subaquático
├── tests/
│   ├── __init__.py
│   └── test_coordinates.py       # 26 testes unitários com pytest validando vs pymap3d
├── docs/                         # Documentação completa e relatórios
│   ├── README.md                 # Índice geral da documentação
│   ├── relatorio_tomsa.tex       # Relatório acadêmico em LaTeX (14 páginas, IEEE)
│   ├── relatorio_tomsa.pdf       # PDF compilado do relatório acadêmico
│   ├── figures/                  # Figuras vetoriais de alta resolução geradas pelo sistema
│   ├── enunciado/                # Enunciado oficial ISEP e formato das logs INESC TEC
│   ├── entrega1/                 # Relatório de fundamentação e bibliotecas da Entrega 1
│   └── manuais/                  # Manual de utilizador, arquitetura e fundamentação matemática
├── base_sources/                 # Ficheiros fonte e protótipos usados como base do projeto
│   ├── README.md                 # Descrição detalhada dos ficheiros e protótipos arquivados
│   ├── code/                     # Scripts preliminares de demonstração (missions.py, test_uwmath.py)
│   ├── vista_primeira_pessoa/    # Protótipo inicial em C++ e Python
│   ├── trajectories/             # Visualizador estático e arquivo ZIP original
│   ├── scratch/                  # Scripts exploratórios de notebook e scripts JS
│   ├── draft_notebooks/          # Rascunhos preliminares de cadernos Jupyter
│   └── initial_notes/            # Brainstorming e modelos de requisitos
├── output/
│   ├── mission_execution.log     # Registo detalhado da execução
│   ├── mission_metrics.json      # Métricas e distâncias mínimas (CPA)
│   ├── synchronized_trajectories.csv # Série temporal sincronizada a 1.0s
│   ├── mission_trajectories.geojson  # Trajetórias prontas para Google Earth / QGIS
│   ├── mission_trajectories_3d.kml   # Trajetórias 3D com gx:Track
│   └── mission_body_view_3d.html # Visualizador 3D interativo autónomo em HTML
├── TOMSA_TL1_GRUPOXX.ipynb       # Caderno Jupyter oficial para a Entrega 1
├── main.py                       # Interface de linha de comandos (CLI principal)
├── try.py                        # Atalho de teste rápido do visualizador 3D Web
├── pytest.ini                    # Configuração para execução direta de testes
├── requirements.txt              # Dependências do projeto (NumPy, SciPy, Matplotlib, Plotly, etc.)
├── .gitignore                    # Ficheiro de exclusões Git (ignora .venv, caches, etc.)
├── README_COMMANDS.md            # Guia exaustivo de comandos de execução (From Path to Source)
└── README.md                     # Documentação geral do projeto
```

---

## 🚀 Como Executar o Projeto: Do Caminho ao Código Fonte (From Path to Source)

Para um guia exaustivo com todos os parâmetros detalhados, consulte também [README_COMMANDS.md](file:///home/geovani/antigravityProjects/ISEP/TOMSA/README_COMMANDS.md).

### 1. Navegação e Preparação do Caminho (`Path`)
Abra o terminal bash e entre na diretoria do projeto:
```bash
# Navegar até ao caminho do projeto
cd /home/geovani/antigravityProjects/ISEP/TOMSA

# Confirmar o diretório de trabalho
pwd
```

### 2. Ativação do Ambiente Virtual (`Source`) ou Execução Direta

#### Opção A: Execução Direta (Recomendado — 100% à prova de falhas)
Executa diretamente o interpretador do ambiente virtual sem depender de ativação nem de comandos específicos da shell:
```bash
# Iniciar o Dashboard Web diretamente:
PYTHONPATH=. .venv/bin/python3 multi_robot_system/main.py --web

# Iniciar a interface gráfica Matplotlib:
PYTHONPATH=. .venv/bin/python3 main.py --visualize

# Executar todos os testes:
PYTHONPATH=. .venv/bin/pytest -v
```

#### Opção B: Ativação Tradicional do Ambiente Virtual
```bash
# Ativar o ambiente virtual (se 'source' der erro na sua shell, use o comando '.'):
. .venv/bin/activate
# ou (em bash puro):
source .venv/bin/activate

# Instalar ou atualizar dependências:
pip install -r requirements.txt

# Exportar a raiz do projeto para o PYTHONPATH:
export PYTHONPATH=.
```
*(Se o ambiente não estiver ativado, o Ubuntu não reconhece o comando `python`, devendo utilizar `python3` ou a Opção A).*

---

### 3. Execução do Pipeline Principal (`main.py`)

Com o ambiente ativado (`source .venv/bin/activate`):

#### A. Validação Completa, Testes e Geração de Ficheiros
```bash
python main.py --all
```
*Executa em cadeia: 26 testes unitários, carregamento de odometria real, métricas CPA, exportação CSV, GeoJSON, HTML 3D e figuras em `docs/figures/`.*

#### B. Interface Gráfica Interativa (Mission Replay em Matplotlib)
```bash
python main.py --visualize
```
*Interface desktop com controlo de velocidade (10x a 100x), Slider temporal de 40 minutos e vistas Global 2D, Espacial 3D e Radar Local sincronizadas.*

#### C. Visualizador 3D em 1ª Pessoa no Navegador (Plotly)
```bash
python main.py --plotly
# ou diretamente:
python try.py
```
*Perspetiva imersiva centrada no Body Frame do veículo com atalhos de teclado `1`, `2` e `3` para comutação instantânea de robô.*

#### D. Exportação Seletiva de Relatórios e Figuras
```bash
# Apenas exportar figuras vetoriais de alta resolução para docs/figures/:
python main.py --export-figures

# Apenas exportar relatórios de métricas JSON, CSV e GeoJSON:
python main.py --export-reports
```

---

### 4. Execução do Sistema Modular Subaquático (`multi_robot_system/main.py`)

O subsistema consolidado em [multi_robot_system/](file:///home/geovani/antigravityProjects/ISEP/TOMSA/multi_robot_system) oferece visualização tática avançada e filtros de Kalman com suavizador RTS:

#### A. Iniciar o Dashboard Web Interativo (Plotly com 5 Vistas)
```bash
python multi_robot_system/main.py --web
```
*Abre automaticamente `http://127.0.0.1:8050` com 5 vistas: Global 3D, 3ª Pessoa com radar, Cockpit de navegação, gráfico de distâncias e separador «5. Encontros e relatório». Tudo o que mudou, por ordem, está em [docs/REGISTO_DE_ALTERACOES.md](docs/REGISTO_DE_ALTERACOES.md).*

#### B. Gerar o Pacote HTML Autónomo Offline
```bash
python multi_robot_system/main.py --bundle
```
*Gera `multi_robot_system/output/mission_dashboard_bundle.html` (executa offline em qualquer navegador sem necessidade de servidor).*

#### C. Benchmark Comparativo de Interpoladores
```bash
python multi_robot_system/main.py --benchmark
```
*Compara Linear vs Spline Cúbica vs Kalman RTS e gera a tabela LaTeX em `multi_robot_system/output/interpolator_latex_table.tex`.*

#### D. Exportação Geoespacial Completa (KML 3D & GeoJSON)
```bash
python multi_robot_system/main.py --export
```
*Gera trajetórias 3D com marcas de tempo para Google Earth (`.kml`) e camadas vetoriais para QGIS (`.geojson`).*

#### E. Execução Integral do Subsistema Subaquático
```bash
python multi_robot_system/main.py --all
```

---

### 5. Execução dos Testes Unitários de Rigor Matemático (`pytest`)

O repositório dispõe de **50 testes unitários automatizados** com validação cruzada contra `pymap3d` ($< 10^{-9}\text{ m}$):

```bash
# Executar todos os 50 testes do projeto
pytest -v

# Executar os testes de conversão geodésica e referenciais (26 testes)
pytest tests/test_coordinates.py -v

# Executar os testes do sistema subaquático (Kalman RTS, SO(3), CPA) (24 testes)
pytest multi_robot_system/tests/ -v
```

---

### 6. Execução do Caderno Jupyter e Relatório LaTeX

```bash
# Abrir o caderno interativo oficial da Entrega 1:
jupyter notebook TOMSA_TL1_GRUPOXX.ipynb

# Compilar o relatório científico (14 páginas) para PDF:
/home/geovani/.local/bin/tectonic docs/relatorio_tomsa.tex
```

---

## 🌟 Avaliação de Viabilidade: "Algo Extra"

No âmbito das funcionalidades extras avaliadas para o projeto:
1. **Elipses de Incerteza (Covariância a 95%):** *(Viável e Implementada)* Extraída a covariância $6 \times 6$ das mensagens ROS, projetada no plano horizontal e decomposta em autovalores para traçar elipses de dispersão estatística ($\sigma \approx 4.1\text{ mm}$, indicando GNSS-RTK de qualidade geodésica).
2. **Monitor de Proximidade e Cálculo de CPA:** *(Viável e Implementada)* O sistema detetou um evento real de aproximação crítica entre o **Robot 1** e o **Robot 4** a **27.66 metros** aos $1450\text{ s}$ da missão, ativando com sucesso o alerta de colisão para o limiar de $30\text{ m}$.
3. **Trajetórias Tridimensionais 3D:** *(Viável e Implementada)* Representação 3D das rotas marinhas em `docs/figures/mission_trajectories_3d.pdf`.
4. **Exportação GeoJSON / KML:** *(Viável e Implementada)* Ficheiro `output/mission_trajectories.geojson` pronto para sobreposição em mapas de satélite no Google Earth ou QGIS.
5. **Visualizador 3D em 1ª Pessoa com Plotly:** *(Viável e Implementada)* Sistema imersivo em `src/plotly_visualizer.py` com projeção contínua no referencial do corpo e comutação instantânea de câmara por atalhos de teclado.
