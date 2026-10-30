# TOMSA 2026 — Visualizador e Analisador de Trajetórias Multi-Robô

Unidade curricular **Tópicos de Matemática para Sistemas Autónomos (TOMSA)** — Mestrado em Engenharia Eletrotécnica e de Computadores, ISEP, 2026.

**Autores:** Geovani Lopes (1211756), Luis Lei (1120542), Hugo Amado (1261726).
**Dados:** logs reais de três veículos autónomos subaquáticos (AUV) do INESC TEC, campanha da Aguçadoura. Uso restrito à unidade curricular.

---

## Índice

1. [Apresentação do projeto](#1-apresentação-do-projeto)
2. [Requisitos do sistema](#2-requisitos-do-sistema)
3. [Instalação passo a passo (Ubuntu)](#3-instalação-passo-a-passo-ubuntu)
4. [Execução do programa](#4-execução-do-programa)
5. [Utilização do dashboard](#5-utilização-do-dashboard)
6. [Organização do projeto](#6-organização-do-projeto)
7. [Dados de entrada](#7-dados-de-entrada)
8. [Resolução de problemas](#8-resolução-de-problemas)
9. [Estrutura técnica do código](#9-estrutura-técnica-do-código)
10. [Validação deste README](#10-validação-deste-readme)

---

## 1. Apresentação do projeto

### Objetivo

O programa lê os registos de odometria (formato ROS `nav_msgs/Odometry` em CSV) de três robôs, **Robot 1**, **Robot 4** e **Robot 7**, e permite **rever a missão** num dashboard no navegador. Faz, por esta ordem:

1. **Leitura e limpeza** dos CSV (ordenação por tempo, remoção de linhas inválidas, tratamento de instantes repetidos).
2. **Conversão de coordenadas**: latitude, longitude e altura (WGS-84) para ECEF e depois para o referencial local **NED** (Norte, Este, Down) em metros.
3. **Orientação**: quaterniões para matrizes de rotação e ângulos de Euler (roll, pitch, yaw), convenção ZYX.
4. **Estimação**: filtro de Kalman de velocidade constante com suavizador de Rauch-Tung-Striebel (RTS), que dá posições suavizadas e velocidades estimadas.
5. **Análise entre robôs**: distâncias 3D, ponto de maior aproximação (CPA), tempo até ao CPA, encontros e avisos de proximidade e de colisão.
6. **Visualização** interativa (Plotly e JavaScript) e **exportação** para GeoJSON, KML e CSV.

### O dashboard

É uma página web servida localmente (`http://127.0.0.1:8050`) com **6 separadores** (vistas), uma barra de tempo, reprodução com várias velocidades e avisos. Serve para ver os três robôs em 3D, ver o mar pelos olhos de cada um, ler os instrumentos de bordo, acompanhar as distâncias entre robôs e consultar o relatório de encontros.

### O que o programa **não** faz

- **Não controla robôs nem evita colisões.** Deteta, prevê (CPA em linha reta) e avisa. A "paragem" dos robôs numa colisão é só visual, sobre os dados gravados (ver [secção 5.6](#56-avisos-de-proximidade-e-colisão)).
- O filtro de Kalman é **offline** (usa o registo todo, passado e futuro). Não é uma estimativa em tempo real.

### Duas bases de código no repositório

| Código | Onde | Estado |
|---|---|---|
| **Principal (usar este)** | `multi_robot_system/` | Dashboard, Kalman/RTS, CPA, relatório. É o descrito neste README. |
| Legado (Entrega 1) | `main.py` na raiz e pasta `src/` | Visualizador em Matplotlib. **O `main.py` da raiz não arranca** neste estado (ver [secção 8](#8-resolução-de-problemas)). Os testes do legado correm. |

---

## 2. Requisitos do sistema

| Item | Requisito |
|---|---|
| **Sistema operativo** | Linux. Testado em **Ubuntu 24.04** (ver [secção 10](#10-validação-deste-readme)). Windows e macOS não foram testados. |
| **Python** | **3.11, 3.12 e 3.13 testados.** Não há versão mínima declarada no repositório. O código não usa nada específico de versões recentes que tenhamos encontrado, mas só as três acima foram executadas. |
| **Bibliotecas Python** | `numpy`, `pandas`, `scipy`, `plotly`, `pyyaml`, `matplotlib`, `pytest`, `pymap3d`, `pyproj` (lista em `requirements.txt`) |
| **Dependências do sistema** | `git` (para clonar), `python3`, `python3-pip`, `python3-venv`. Mais nenhuma. |
| **Internet** | **Sim, para abrir o dashboard.** A página carrega o `plotly.js` a partir de `https://cdn.plot.ly`. Sem Internet a página abre, mas as figuras 3D não aparecem. Também é preciso Internet para `pip install` e `git clone`. O ficheiro gerado por `--bundle` **também** usa esse endereço (não é offline a 100%). |
| **Jupyter** | **Não é necessário** para correr o programa. Só é preciso para abrir o relatório `TOMSA_TL2_GRUPOXX.ipynb` (`pip install jupyter`, ver [secção 3.7](#37-opcional-abrir-o-relatório-em-jupyter)). Não está no `requirements.txt`. |
| **Dashboard** | Navegador moderno com JavaScript e WebGL (Chrome, Chromium ou Firefox). Foi validado em Chromium. Ecrã com pelo menos 1280 pixéis de largura. |
| **Espaço e memória** | Os 3 CSV ocupam cerca de 45 MB. O arranque processa 30 mil amostras em poucos segundos. |

**Versões usadas nos testes** (instaladas automaticamente pelo `pip` a 30 de outubro de 2026): numpy 2.4.6 e 2.5.4, pandas 3.1.0, scipy 1.17.1 e 1.18.1, plotly 7.1.0, PyYAML 6.0.3, matplotlib 3.11.2, pytest 9.2.0, pymap3d 3.2.0, pyproj 3.8.0. O `requirements.txt` só fixa **mínimos** (por exemplo `numpy>=1.24.0`), por isso o `pip` instala sempre as versões mais recentes compatíveis.

---

## 3. Instalação passo a passo (Ubuntu)

Todos os comandos são para um terminal. Não é preciso editar nenhum caminho.

### 3.1 Instalar Git, Python, pip e venv

```bash
sudo apt update
sudo apt install -y git python3 python3-pip python3-venv
git --version
python3 --version
```

O `python3 --version` deve mostrar 3.11 ou superior. O Ubuntu 24.04 traz 3.12. O Ubuntu 22.04 traz 3.10, que **não testámos**.

### 3.2 Clonar o repositório

```bash
git clone https://github.com/Juninhoo/TOMSA.git
cd TOMSA
```

A partir daqui, **todos os comandos correm dentro da pasta `TOMSA`** (a raiz do repositório, onde estão `requirements.txt` e `multi_robot_system/`).

Se recebeu o projeto num ficheiro ZIP, descompacte-o e entre na pasta:

```bash
unzip TOMSA.zip
cd TOMSA
```

> Atenção: `multi_robot_system/data/*.csv` são **ligações simbólicas** para `data/*.csv`. Em Linux e macOS funcionam. Num ZIP criado em Windows podem ficar como ficheiros de texto vazios; nesse caso, apague `multi_robot_system/data` e copie `data` para esse lugar (`cp -r data multi_robot_system/data`).

### 3.3 Criar o ambiente virtual

```bash
python3 -m venv .venv
```

### 3.4 Ativar o ambiente virtual

```bash
source .venv/bin/activate
```

O terminal passa a mostrar `(.venv)` no início da linha. Tem de repetir este comando em cada terminal novo.

### 3.5 Instalar as bibliotecas

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.6 Variável de ambiente

O projeto usa importações absolutas (`from multi_robot_system...`), por isso o Python precisa de saber que a raiz do repositório faz parte do caminho de módulos:

```bash
export PYTHONPATH=.
```

Vale só para o terminal atual e só funciona com o terminal **na raiz do repositório**. Para o tornar permanente na sessão do ambiente virtual pode acrescentar essa linha ao fim de `.venv/bin/activate`.

### 3.7 (Opcional) Abrir o relatório em Jupyter

```bash
pip install jupyter
jupyter notebook TOMSA_TL2_GRUPOXX.ipynb
```

O notebook deve ser aberto com o terminal na raiz do repositório. Procura sozinho a pasta `multi_robot_system` subindo a partir da pasta atual.

### 3.8 Confirmar a instalação

```bash
python -m pytest -q
python -m pytest multi_robot_system/tests -q
```

Resultado esperado: `26 passed` e `32 passed`. A primeira linha corre os testes do código legado (`tests/`, escolhido por `pytest.ini`), a segunda os do código principal.

---

## 4. Execução do programa

Com o ambiente ativo, `PYTHONPATH=.` definido e o terminal na raiz do repositório:

```bash
export PYTHONPATH=.
python3 multi_robot_system/main.py --web
```

Estes são exatamente os comandos que confirmámos no código (`multi_robot_system/main.py`). O programa carrega os três CSV (cerca de 3 segundos), mostra o resumo no terminal e fica à espera:

```
[SERVIDOR DASHBOARD WEB] Ativo em: http://127.0.0.1:8050
```

Abre o navegador sozinho (função `webbrowser.open`). Se estiver num servidor sem ambiente gráfico, abra **http://127.0.0.1:8050** manualmente. Para terminar, carregue em `Ctrl+C` no terminal.

### Todos os modos de execução

| Comando | O que faz |
|---|---|
| `python3 multi_robot_system/main.py --web` | Carrega a missão e abre o **dashboard** em `http://127.0.0.1:8050`. |
| `python3 multi_robot_system/main.py --test` | Corre os testes (pytest) e **depois** carrega a missão (não abre o dashboard). |
| `python3 multi_robot_system/main.py --export` | Cria `multi_robot_system/output/synchronized_trajectories.csv`, `mission_trajectories.geojson` (QGIS) e `mission_trajectories_3d.kml` (Google Earth). |
| `python3 multi_robot_system/main.py --benchmark` | Compara interpolação linear, spline cúbica e Kalman+RTS. Escreve `output/interpolator_comparison.csv` e `multi_robot_system/output/interpolator_latex_table.tex`. |
| `python3 multi_robot_system/main.py --bundle` | Escreve `multi_robot_system/output/mission_dashboard_bundle.html` (a página do dashboard num só ficheiro, **que ainda precisa de Internet** para o plotly.js). |
| `python3 multi_robot_system/main.py --all` | Faz tudo: testes, benchmark, exportação, bundle e dashboard. **É também o que acontece se não passar argumentos.** |
| `python3 multi_robot_system/main.py --config OUTRO.yaml` | Usa outro ficheiro de configuração (por omissão `multi_robot_system/config/mission_config.yaml`). |
| `python3 multi_robot_system/main.py --dados A.csv B.csv` | Usa os CSV indicados (ou uma pasta com CSV) no lugar dos da configuração (ver [secção 7](#7-dados-de-entrada)). |

Podem combinar-se, por exemplo `--export --web`. As pastas `output` e `multi_robot_system/output` já existem no repositório; se as apagar, os modos `--export`, `--benchmark` e `--bundle` falham (ver [secção 8](#8-resolução-de-problemas)).

### Abrir já num instante, vista e robô

O dashboard aceita parâmetros no endereço:

```
http://127.0.0.1:8050/?t=3249&view=cockpit&robot=2&speed=5&play=1
```

| Parâmetro | Significado |
|---|---|
| `t` | segundos desde o início da missão |
| `view` | `global`, `pov` (3ª pessoa), `fp` (1ª pessoa), `cockpit`, `prox` (distâncias) ou `report` |
| `robot` | número do botão do robô (1, 2 ou 3 para Robot 1, Robot 4 e Robot 7) |
| `speed` | velocidade de reprodução (0.25, 1, 5, 10, 20 ou 50) |
| `play=1` | começa a reproduzir |

### Código legado (Entrega 1)

- `python3 main.py` (raiz) **falha** neste estado: `NameError: name 'Optional' is not defined` em `src/logger.py` (falta `from typing import Optional`). Não o corrigimos para não mexer em código funcional sem necessidade.
- `python3 try.py` (raiz) funciona e escreve `output/mission_body_view_3d.html`.

---

## 5. Utilização do dashboard

### 5.1 Cabeçalho e barra de controlo

| Elemento | Função |
|---|---|
| **Botões de robô** (um por robô) | Escolhem o robô ativo. A Vista Global passa a segui-lo. |
| **Play / Pausa** | Inicia e pára a reprodução. Ao chegar ao fim volta ao início. |
| **Reiniciar (R)** | Volta ao instante inicial da missão. |
| **Seguir (F)** | Liga e desliga o seguimento do robô ativo pela câmara da Vista Global. |
| **Vista geral (G)** | Volta à vista de cima com os 3 robôs. |
| **Previsão 60 s (P)** | Mostra, para cada robô, a reta que percorreria em 60 s se mantivesse a velocidade estimada (pára à superfície). |
| **Robots parados (C)** | Só aparece se existir colisão. Alterna entre robôs parados e dados reais. |
| **Barra de tempo** | Arrastar para saltar no tempo. Os blocos amarelos e vermelhos por baixo marcam os períodos de proximidade (clique salta para 10 segundos antes). |
| **Velocidade** | 0.25x, 1x, 5x, 10x, 20x, 50x e "Ultra" (a missão toda em cerca de 2 segundos). |

### 5.2 Atalhos de teclado

Confirmados no código (`web_server.py`, ouvinte `keydown`):

| Tecla | Ação |
|---|---|
| `1`, `2`, `3`, … | Seleciona o robô de ordem 1, 2, 3, … |
| `V` | Passa à vista seguinte: Global, 3ª pessoa, 1ª pessoa, Cockpit, Distâncias, Relatório, e volta ao início |
| `Espaço` | Play e pausa |
| `R` | Reinicia |
| `F` | Seguir o robô |
| `G` | Vista geral |
| `P` | Previsão de 60 s ligada ou desligada |
| `C` | Robôs parados ou dados reais (só com colisão) |
| `H` | Painel de bordo da 1ª pessoa ligado ou desligado |
| `+` e `-` | Zoom da câmara na 3ª pessoa |

As teclas não funcionam enquanto o cursor estiver dentro de uma caixa de texto.

### 5.3 Os 6 separadores

| N.º | Separador | O que mostra |
|---|---|---|
| 1 | **Vista Global 3D** | Mar, fundo, rasto e posição de cada robô com o casco orientado por roll, pitch e yaw, setas dos eixos do corpo, previsão de 60 s, aviso de proximidade. Rato: rodar, zoom e deslocar (Plotly). |
| 2 | **Vista 3ª Pessoa** | Câmara atrás e acima do robô ativo; **radar** com a proa para cima (alcance 50, 100, 250, 500 ou 1000 m, botões − e +) e mini-cockpit. Rato: arrastar roda a câmara, roda do rato faz zoom, duplo clique repõe. |
| 3 | **Cockpit / Controlo** | Painéis do robô ativo: atitude (roll, pitch, yaw), **bússola com rumo e rota** e deriva, velocidade medida e estimada, profundidade, e tabela de **vizinhos** (azimute, elevação, distância, aproximação, tempo até ao CPA e CPA). |
| 4 | **Distâncias CPA 2D** | Distância 3D entre cada par de robôs ao longo do tempo, faixas de aviso e de perigo, cursor do tempo atual, escalas 0 a 400 m, 0 a 60 m e logarítmica. |
| 5 | **Encontros e relatório** | Janelas de operação de cada robô, resumo por par (distância mínima), lista de encontros (distância abaixo de 60 m), saltos de posição detetados. Exporta `.md` e `.csv`. |
| 6 | **Vista 1ª Pessoa** | Câmara no nariz do robô ativo, com painel de bordo: fita de rumo, profundidade, arfagem (pitch), rolamento (roll), velocidades, taxa de viragem e vizinho mais próximo. Botão "Ver pelo vizinho mais próximo" e `H` para esconder o painel. |

### 5.4 Indicadores de posição, velocidade e orientação

- **Posição NED** (Norte, Este, Profundidade em metros), no cabeçalho do Cockpit e nas etiquetas.
- **Orientação:** roll, pitch e yaw em graus; **rumo** = yaw entre 0 e 360 graus (Norte = 0, Este = 90).
- **Velocidade estimada** (Kalman/RTS, projetada na proa) e **velocidade medida** (campo `twist.twist.linear.x` do CSV), em metros por segundo.
- **Taxa de viragem** em graus por segundo.
- **Rota** (direção real do movimento, só quando a velocidade horizontal é superior a 0.3 m por segundo) e **deriva** (rota menos rumo).

### 5.5 Reproduzir e navegar no tempo

Os robôs não estão todos ativos ao mesmo tempo: Robot 7 de 0 a 133,5 minutos, Robot 4 de 20,0 a 69,9 minutos, Robot 1 de 30,0 a 114,8 minutos (tempo desde o primeiro registo). Fora da sua janela, um robô aparece esbatido e marcado como "inativo". Só existe **janela comum aos três** durante cerca de 39,9 minutos (2396,3 segundos), entre os 30,0 e os 69,9 minutos.

### 5.6 Avisos de proximidade e colisão

Limiares (ficheiro `multi_robot_system/config/mission_config.yaml`, secção `spatial`), sempre com a **distância 3D**:

| Distância entre dois robôs | Nível | Reação do dashboard |
|---|---|---|
| abaixo de **60 m** | Encontro | Só no separador 5 (lista de encontros). |
| abaixo de **30 m** (`proximity_warning_m`) | **Atenção** | Faixa **amarela** "ATENÇÃO: ROBOTS PRÓXIMOS" nas vistas 1, 2 e 6, bloco amarelo na barra de tempo. |
| abaixo de **10 m** (`collision_threshold_m`) | **Perigo de colisão** | Faixa **vermelha** "PERIGO DE COLISÃO", bloco vermelho na barra de tempo. |
| abaixo de **2 m** (`contact_distance_m`) | **Colisão** | Faixa "COLISÃO": o Play **pára** nesse instante e os dois robôs ficam **parados** (apenas visualização). Tecla `C` mostra os dados reais. |

**Detalhe importante:** nos dados reais fornecidos a menor distância entre robôs é de **27,7 m** (Robot 1 e Robot 4, aos 54 min 9,3 s). Há apenas avisos amarelos; nunca perigo nem colisão.

O dashboard **deteta** (distância atual), **prevê** (CPA e TCPA com movimento em linha reta, horizonte de 60 s, no Cockpit; linha de previsão de 60 s no 3D) e **avisa**. **Não impede** colisões.

---

## 6. Organização do projeto

```
TOMSA/
├── README.md                    Este ficheiro
├── requirements.txt             Dependências (mínimos)
├── pytest.ini                   Configura o pytest (pythonpath = ., testpaths = tests)
├── TOMSA_TL2_GRUPOXX.ipynb      Relatório da Entrega 2 (Jupyter)
├── TOMSA_TL1_GRUPOXX.ipynb      Relatório da Entrega 1
├── data/                        CSV originais dos 3 robôs (robot1_s1, robot4_s1, robot7_s1)
├── output/                      Saídas do código legado e do benchmark (CSV)
├── docs/                        Enunciado, manuais, relatórios e figuras
│
├── multi_robot_system/          CÓDIGO PRINCIPAL
│   ├── main.py                  Ponto de entrada (linha de comandos)
│   ├── config/mission_config.yaml   Configuração da missão
│   ├── data/                    Ligações simbólicas para ../data
│   ├── core/
│   │   ├── data/
│   │   │   ├── odom_loader.py       Leitura do CSV, limpeza, NED, orientação, Kalman, saltos
│   │   │   ├── trajectory.py        RobotTrajectory e MissionDataset (SLERP, estado no instante t)
│   │   │   └── mission_report.py    Encontros, resumo por par, saltos
│   │   └── math/
│   │       ├── geodesy.py           WGS-84, ECEF, NED, ENU, AER
│   │       ├── rotations.py         Quaterniões, matrizes de rotação, Euler ZYX, SLERP
│   │       ├── kalman_rts.py        Filtro de Kalman e suavizador RTS, velocidade e viragem
│   │       ├── proximity.py         Distância relativa, AER, CPA e TCPA
│   │       └── interpolators.py     Comparação linear, spline cúbica e Kalman+RTS
│   ├── visualization/
│   │   ├── web_server.py            Dashboard: HTML, JavaScript, servidor HTTP
│   │   ├── plotly_views.py          Figuras Plotly (3D global, 3ª pessoa, distâncias, casco)
│   │   └── cockpit_view.py          Cockpit em Plotly (importado, não usado pelo dashboard)
│   ├── io_export/
│   │   ├── geojson_exporter.py      Exportação GeoJSON
│   │   ├── kml_exporter.py          Exportação KML 3D
│   │   └── benchmark_exporter.py    Benchmark dos interpoladores, tabela LaTeX
│   ├── tests/                   32 testes (pytest)
│   └── output/                  Ficheiros gerados (exportações, bundle)
│
├── src/                         Código legado (Matplotlib), Entrega 1
├── main.py                      Ponto de entrada legado (ver secção 4)
├── tests/                       26 testes do código legado
└── base_sources/                Rascunhos e material de trabalho
```

### Quem faz o quê

| Responsabilidade | Módulo |
|---|---|
| **Leitura dos dados** | `core/data/odom_loader.py` (`load_single_robot_csv`), `main.py` (`load_mission`, `robots_from_files`) |
| **Processamento matemático** | `core/math/geodesy.py`, `core/math/rotations.py` |
| **Estimação e filtros** | `core/math/kalman_rts.py` (`TrajectoryKalmanRTS`) |
| **Simulação (estado no instante t)** | `core/data/trajectory.py` (`MissionDataset.get_state_at_time`); no dashboard, a interpolação é refeita em JavaScript |
| **Deteção de proximidade e risco de colisão** | `core/math/proximity.py` (CPA, testado), `core/data/mission_report.py` (encontros), `web_server.py` (avisos e colisão em JavaScript) |
| **Visualização** | `visualization/plotly_views.py` |
| **Dashboard** | `visualization/web_server.py` |
| **Exportações** | `io_export/` |

---

## 7. Dados de entrada

### Formato

Um CSV por robô, com cabeçalho, no formato `nav_msgs/Odometry` do ROS, como fornecido pelo INESC TEC (`docs/enunciado/TOMSA2026_TL_readme.pdf`). São **obrigatórias** estas 8 colunas:

| Coluna | Significado | Unidade |
|---|---|---|
| `Time` | instante do registo (segundos desde 1970, POSIX) | s |
| `pose.pose.position.x` | **latitude** | graus decimais (Norte positivo) |
| `pose.pose.position.y` | **longitude** | graus decimais (Este positivo) |
| `pose.pose.position.z` | **altura** acima do elipsoide WGS-84 | m |
| `pose.pose.orientation.x`, `.y`, `.z`, `.w` | **quaternião** de atitude (ordem x, y, z, w) | sem unidades |

São **opcionais** (se faltarem, o programa continua com valores por omissão):

| Coluna | Uso | Se faltar |
|---|---|---|
| `pose.covariance` | 36 valores (matriz 6x6 por linhas, entre parênteses). A diagonal de posição define o ruído de medição do Kalman. | Ruído de 0,8 m |
| `twist.twist.linear.x`, `.y`, `.z` | Velocidade medida no referencial do corpo, em metros por segundo (só para mostrar) | Zeros |

As restantes colunas (`header.seq`, `header.stamp.*`, `header.frame_id`, `child_frame_id`, `twist.twist.angular.*`, `twist.covariance`) existem nos ficheiros fornecidos mas **não são usadas**.

### Sistemas de coordenadas e unidades

| Referencial | Eixos | Unidades |
|---|---|---|
| **Entrada** | WGS-84: latitude, longitude, altura elipsoidal | graus, graus, m |
| **Global do programa (NED)** | x = Norte, y = Este, z = **Down** (profundidade positiva para baixo) | m |
| **Corpo do robô** | x = proa, y = estibordo (direita), z = quilha (baixo) | m |

A origem do NED é, por omissão (`origin.mode: data_mean`), a média das latitudes e longitudes de todas as amostras e a média das medianas das 50 primeiras alturas de cada robô (superfície do mar). Com `mode: fixed` usa-se a "Surface Reference" do enunciado (latitude 41.45201666666667, longitude -8.81322222222222, altura 55.44791126176966 m).

Ângulos: graus nas saídas, quaterniões em estado bruto. Tempo: segundos. Rumo: 0 a 360 graus, Norte = 0, Este = 90.

### Substituir os dados

**Opção A — sem editar nada**, pela linha de comandos:

```bash
export PYTHONPATH=.
python3 multi_robot_system/main.py --web --dados pasta_com_csv
python3 multi_robot_system/main.py --web --dados robot1.csv robot4.csv robot7.csv
```

O nome do robô sai do ficheiro: `robot7_s2.csv` dá "Robot 7". Sem número no nome usa-se o nome do ficheiro. A origem passa a ser a média dos novos dados. Se os robôs nunca estiveram ativos ao mesmo tempo, o programa avisa que não há janela comum e as distâncias entre eles não se aplicam.

**Opção B — editar a configuração** `multi_robot_system/config/mission_config.yaml`, secção `robots` (um bloco `id`, `file`, `color`, `label` por robô). O caminho em `file` pode ser relativo à raiz do repositório ou à pasta `multi_robot_system`.

**Tratamento automático de CSV imperfeitos** (testado): linhas com valores em falta nas colunas obrigatórias são ignoradas (com aviso), registos com o mesmo instante são afastados 1 milissegundo (com aviso), os dados são ordenados por tempo e o ficheiro original nunca é alterado. Um CSV sem `pose.covariance` ou sem `twist` é aceite.

### Parâmetros da configuração que contam

| Chave | Valor | Efeito |
|---|---|---|
| `spatial.collision_threshold_m` | 10.0 | Perigo de colisão |
| `spatial.proximity_warning_m` | 30.0 | Aviso de proximidade |
| `spatial.contact_distance_m` | 2.0 | Colisão (pára a reprodução) |
| `spatial.radar_range_m` | 100.0 | Alcance inicial do radar |
| `filter.use_rts_smoother` | true | Liga o Kalman e o RTS |
| `filter.process_noise_q` | 0.05 | Ruído de processo do Kalman |
| `output.web_host`, `output.web_port` | 127.0.0.1, 8050 | Endereço do dashboard |
| `frame`, `replay.*`, `camera.*`, `spatial.neighbor_radius_m`, `spatial.seabed_margin_m`, `filter.default_meas_std_m`, `output.benchmark_file` | (vários) | Constam do ficheiro mas **nenhum código os lê**: alterá-los não tem efeito (verificado por pesquisa no código; alguns valores existem como argumentos por omissão das funções, com o mesmo número). |

---

## 8. Resolução de problemas

Todas as mensagens abaixo foram **reproduzidas** numa instalação limpa.

| Problema | Mensagem típica | Solução |
|---|---|---|
| **Falta o PYTHONPATH** | `ModuleNotFoundError: No module named 'multi_robot_system'` | Estar na raiz do repositório e fazer `export PYTHONPATH=.`. Em alternativa, numa só linha: `PYTHONPATH=. python3 multi_robot_system/main.py --web`. |
| **Terminal na pasta errada** | `FileNotFoundError: ... 'multi_robot_system/config/mission_config.yaml'` | Fazer `cd` para a pasta `TOMSA` (a que contém `requirements.txt`). Os caminhos são relativos a ela. |
| **Biblioteca em falta** | `ModuleNotFoundError: No module named 'yaml'` (ou `numpy`, `pandas`, `plotly`, `scipy`) | Ativar o ambiente (`source .venv/bin/activate`) e correr `pip install -r requirements.txt`. Confirmar que o `python3` é o do ambiente: `which python3` deve apontar para `.venv`. |
| **Porta 8050 ocupada** | `OSError: [Errno 98] Address already in use` (aparece só depois de carregar os dados) | Ver quem a usa com `ss -ltnp \| grep 8050` (comando standard do Ubuntu, não testado) e terminá-lo, ou mudar `output.web_port` em `mission_config.yaml` (por exemplo 8051) e abrir `http://127.0.0.1:8051`. O programa não escolhe outra porta sozinho. |
| **O navegador não abre** | (nada acontece) | Abrir manualmente `http://127.0.0.1:8050`. Em servidores sem ambiente gráfico (SSH) usar um túnel SSH (`ssh -L 8050:127.0.0.1:8050 utilizador@servidor`, sugestão **não testada**). |
| **Página abre mas as figuras 3D ficam vazias** | Consola do navegador: falha a carregar `cdn.plot.ly` | Falta de Internet ou bloqueio da rede. O `plotly.js` é carregado da Internet; ligue-se à rede e recarregue. Não há versão offline neste momento. |
| **Dependências incompatíveis** | Erros de `pip` ou de importação depois de instalar | Criar um ambiente novo (`rm -rf .venv && python3 -m venv .venv`) e reinstalar. Testámos com Python 3.11, 3.12 e 3.13. |
| **CSV não encontrado** | `FileNotFoundError: ... 'multi_robot_system/nome.csv'` | Verificar o caminho. Mesmo que passe um caminho que não existe, a mensagem mostra-o com o prefixo `multi_robot_system/`, porque o programa tenta também esse prefixo. |
| **Dados em `multi_robot_system/data` vazios ou em falta** | `FileNotFoundError` ou `EmptyDataError` | As ligações simbólicas foram perdidas (ZIP do Windows). Repor: `rm -rf multi_robot_system/data && cp -r data multi_robot_system/data`. |
| **CSV sem uma coluna obrigatória** | `ValueError: Usecols do not match columns, columns expected but not found: [...]` | O nome da coluna em falta aparece na mensagem. Comparar com a tabela da [secção 7](#7-dados-de-entrada). |
| **`--export`, `--benchmark` ou `--bundle` falham** | `OSError: Cannot save file into a non-existent directory: 'multi_robot_system/output'` ou `FileNotFoundError: ... interpolator_latex_table.tex` | As pastas de saída não existem. Criá-las: `mkdir -p multi_robot_system/output output`. |
| **`python3 main.py` (raiz) falha** | `NameError: name 'Optional' is not defined` | Defeito do código legado (falta `from typing import Optional` em `src/logger.py`). Usar `multi_robot_system/main.py`. |
| **Os atalhos de teclado não respondem** | — | Clicar primeiro na página (foco) e não estar com o cursor numa caixa de texto. |

---

## 9. Estrutura técnica do código

### Fluxo de dados

```
CSV (lat, lon, alt, quaternião, covariância, twist)
  │  odom_loader.load_single_robot_csv
  ▼
limpeza → geodesy (WGS-84 → ECEF → NED) → rotations (quaternião → R, Euler ZYX)
  │
  ▼
kalman_rts.TrajectoryKalmanRTS  → posições suavizadas + velocidades NED
  │
  ▼
trajectory.RobotTrajectory (um por robô)  →  MissionDataset (janelas, SLERP, estado em t)
  │
  ├──► mission_report.compute_mission_report  (encontros, mínimos, saltos)
  ├──► plotly_views  (figuras 3D e de distâncias)
  ├──► io_export  (GeoJSON, KML, CSV, benchmark)
  ▼
web_server.MissionWebDashboard.build_dashboard_html
  → "linha de tempo" em JSON (grelha de 2 s) + JavaScript no navegador
     (interpolação linear, vistas, rumo, distâncias, CPA, avisos, colisão)
```

### Decisões que convém conhecer

- **Python calcula, o navegador desenha.** O Python entrega ao navegador uma grelha temporal fixa de 2 segundos com as séries de cada robô. O JavaScript só interpola linearmente entre amostras e desenha.
- **Convenção de rotações:** `R = Rz(yaw) · Ry(pitch) · Rx(roll)` (ZYX, aeroespacial), matriz do corpo para NED. Quaterniões na ordem `[x, y, z, w]` (a mesma do ROS).
- **Distância entre robôs:** euclidiana **3D** em NED.
- **CPA:** o cálculo em Python (`proximity.py`) está testado mas o dashboard refaz a mesma fórmula em JavaScript.

### Exemplos mínimos de utilização

Carregar a missão e obter o estado de todos os robôs num instante:

```python
from multi_robot_system.main import load_mission

dataset, cfg = load_mission("multi_robot_system/config/mission_config.yaml")
t = dataset.common_start + 600.0                  # 10 minutos dentro da janela comum
estado = dataset.get_state_at_time(t)             # dicionário por robô
print(estado["Robot 1"]["pos_ned"], estado["Robot 1"]["euler_deg"])
```

Distância mínima entre robôs e encontros:

```python
from multi_robot_system.core.data.mission_report import compute_mission_report

rel = compute_mission_report(dataset)
for par in rel["pairs"]:
    print(par["a"], par["b"], round(par["d_min"], 1), "m")
```

Ponto de maior aproximação entre dois robôs:

```python
import numpy as np
from multi_robot_system.core.math.proximity import compute_analytic_cpa

t_cpa, d_cpa = compute_analytic_cpa(
    np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]),     # posição e velocidade do robô 1
    np.array([100.0, 0.0, 0.0]), np.array([-3.0, 0.0, 0.0]),  # posição e velocidade do robô 2
)
print(t_cpa, d_cpa)    # 20.0 s, 0.0 m
```

Todos estes exemplos correm com `PYTHONPATH=.` e o terminal na raiz do repositório.

### Testes

```bash
python -m pytest multi_robot_system/tests -q    # 32 testes do código principal
python -m pytest -q                              # 26 testes do código legado
```

Cobrem: rotações (ortonormalidade, Euler, SLERP), geodesia (contra `pymap3d`), Kalman/RTS (redução de ruído e velocidades), CPA (colisão frontal, passagem lateral, afastamento), relatório da missão, CSV imperfeitos e a presença da deteção de colisão no dashboard. **Não há testes automáticos do JavaScript** em execução no navegador.

Mais pormenores: `docs/manuais/MANUAL_USUARIO.md` e o relatório `TOMSA_TL2_GRUPOXX.ipynb` (fundamentação matemática de cada módulo).

---

## 10. Validação deste README

Os passos foram executados numa instalação limpa, a partir de uma cópia do código **sem** `.venv`, `__pycache__` nem `.pytest_cache`.

### O que foi testado

| Passo | Ambiente | Resultado |
|---|---|---|
| `apt install git python3 python3-pip python3-venv` | Ubuntu 24.04.5 | Simulado (`apt-get -s`): pacotes existentes, comando válido. Não instalado de raiz. |
| `git clone https://github.com/Juninhoo/TOMSA.git` | Ubuntu 24.04.5 | Funciona (repositório público). |
| `python3 -m venv .venv` + `pip install -r requirements.txt` | Python 3.12.3 e 3.11.17 | Instalação concluída sem erros. |
| `python -m pytest -q` e `python -m pytest multi_robot_system/tests -q` | Python 3.11, 3.12 e 3.13 | `26 passed` e `32 passed` em todos. |
| `python3 multi_robot_system/main.py --web` | Python 3.12.3 | Servidor ativo; pedido HTTP a `http://127.0.0.1:8050/` devolve código 200 com 3,2 MB de HTML. |
| `--export`, `--benchmark`, `--bundle`, `--test` | Python 3.12.3 | Concluídos; ficheiros criados. |
| `--dados` com CSV inexistente e com coluna em falta | Python 3.12.3 | Erros iguais aos da secção 8. |
| Erros da secção 8 (sem `PYTHONPATH`, pasta errada, sem bibliotecas, porta ocupada) | Python 3.12.3 | Mensagens reproduzidas. |
| Dashboard no navegador (vistas, atalhos, avisos) | Chromium headless (Playwright), durante o desenvolvimento | Funcional. |

### O que **não** foi testado

- Instalação em **Ubuntu 22.04** (Python 3.10 de origem) ou noutras distribuições.
- `sudo apt install ...` numa máquina sem esses pacotes (apenas simulação).
- Abertura automática do navegador (`webbrowser.open`) num ambiente gráfico real. O ambiente de teste não tem ecrã.
- **Windows e macOS**, incluindo as ligações simbólicas de `multi_robot_system/data` num ZIP feito em Windows.
- Dashboard em **Firefox, Safari e Edge**.
- O arranque **sem Internet** (confirmado apenas pela leitura do código: o `plotly.js` vem da CDN).
- A versão do GitHub no momento da leitura: o teste de `git clone` confirma o endereço, mas as instruções descrevem o estado **final** do código. Só são válidas depois de a versão final estar enviada para o repositório.
- Python 3.10 e anteriores; versões de bibliotecas diferentes das listadas na secção 2.
