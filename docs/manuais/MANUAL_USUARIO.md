# Manual de Utilizador e Guia de Operação - TOMSA 2026
## Sistema Multi-Robô Subaquático: Visualizador, Sincronização e Análise Cinemática

Este manual documenta a operação do sistema unificado na pasta `multi_robot_system/`, consolidando as funcionalidades das ferramentas anteriores num ambiente coeso, multithreaded e centrado em Plotly.

---

## 1. Visão Geral da Arquitetura e Decisões de Engenharia

O sistema foi estruturado segundo uma **Arquitetura Desacoplada (Clean Architecture)**:

```
multi_robot_system/
├── config/
│   └── mission_config.yaml         # Parametrização integral da missão (origem, cores, limiares)
├── data/
│   ├── robot1_s1.csv               # Telemetria real INESC TEC Robot 1
│   ├── robot4_s1.csv               # Telemetria real INESC TEC Robot 4
│   └── robot7_s1.csv               # Telemetria real INESC TEC Robot 7
├── core/
│   ├── math/                       # Núcleo matemático puro (NumPy/SciPy sem UI)
│   │   ├── geodesy.py              # WGS-84, ECEF, NED, ENU, AER (validado vs pymap3d)
│   │   ├── rotations.py            # Quaterniões unitários em SO(3), Euler ZYX, SLERP
│   │   ├── kalman_rts.py           # Filtro Kalman CWNA e Suavizador RTS ótimo
│   │   ├── proximity.py            # Distâncias relativas, AER no corpo e CPA analítico (< 10m)
│   │   └── interpolators.py        # Comparador Linear vs Spline Cúbica vs Kalman/RTS
│   └── data/
│       ├── odom_loader.py          # Leitor seguro de CSV (ordenação estrita por tempo)
│       └── trajectory.py           # Dataclass RobotTrajectory e MissionDataset sincronizado
├── visualization/                  # Camada de apresentação e interface web
│   ├── plotly_views.py             # Motor 3D da Vista Global e da Vista 3ª Pessoa
│   ├── cockpit_view.py             # 3ª Vista: Painel de Instrumentos / Sala de Controlo
│   └── web_server.py               # Servidor HTTP multithreaded com dashboard integrado
├── io_export/
│   ├── geojson_exporter.py         # Exportador para SIG (QGIS)
│   ├── kml_exporter.py             # Exportador para Google Earth 3D (gx:Track)
│   └── benchmark_exporter.py       # Gerador de tabelas LaTeX e figuras de erro
├── tests/                          # 24 testes unitários automatizados (pytest)
├── output/                         # Resultados gerados (CSV, GeoJSON, KML, HTML bundle)
├── main.py                         # Ponto de entrada CLI
└── requirements.txt                # Dependências do projeto
```

---

## 2. Como Executar o Sistema

Utilize o ambiente virtual Python `.venv` já configurado na raiz do projeto.

### 2.1 Iniciar o Dashboard Web Interativo (Plotly)
```bash
PYTHONPATH=. .venv/bin/python3 multi_robot_system/main.py --web
```
O servidor abre automaticamente no navegador o endereço:
`http://127.0.0.1:8050`

### 2.2 Gerar o Pacote HTML Autónomo Offline
Gera um ficheiro `.html` único (auto-contido) que abre diretamente em qualquer computador sem necessidade de servidor:
```bash
PYTHONPATH=. .venv/bin/python3 multi_robot_system/main.py --bundle
```
Ficheiro criado: `multi_robot_system/output/mission_dashboard_bundle.html`

### 2.3 Executar Validação Completa (Testes + Benchmark + Exportações)
```bash
PYTHONPATH=. .venv/bin/python3 multi_robot_system/main.py --all
```

---

## 3. As Seis Vistas do Sistema

O utilizador pode alternar entre as 5 vistas através dos botões no cabeçalho ou premindo a tecla **`V`**:

### Vista 1: Global 3D (Visão Panorâmica)
- **Cenário**: Representa todos os submarinos num referencial local comum NED com profundidade real.
- **Batimetria**: Fundo do mar em grelha cinzenta e superfície da água como plano azul translúcido ($D = 0\,\text{m}$).
- **Rasto Progressivo**: A trajetória inicia sem traço e vai sendo desenhada gradualmente à medida que o tempo avança.
- **Submarino 3D**: cada veículo é uma malha sólida (casco, barbatanas e leme) orientada em 3D conforme yaw, pitch e roll, na cor do robot.

### Vista 2: Vista 3.ª Pessoa
- **Perspetiva**: câmara atrás e acima do robot selecionado (Robot 1, 4 ou 7), que se vê de fora. Rato: arrastar roda à volta, roda do rato ou teclas + e - mudam a distância, duplo clique repõe.
- **Radar e mini-cockpit**: radar circular com os outros robots (alcance regulável) e quatro instrumentos (velocidade, rotação, profundidade e inclinação) do robot selecionado.
- **Previsão de 60 s (tecla P, vistas Global e 3ª Pessoa)**: linha tracejada com a posição de cada robot daqui a 60 s se mantiver a velocidade.

### Vista 3: Cockpit / Sala de Controlo (Painel de Instrumentos)
- **Atitude 3D e eixos do corpo**: modelo do robot com os eixos x (proa), y (estibordo) e z (quilha), os eixos N, E, D, o rolamento (roll), a arfagem (pitch) e a guinada (yaw) em destaque e o quaternião.
- **Rumo e rota**: bússola com o rumo (heading) e a rota (course), esta só quando a velocidade é superior a 0,3 m/s.
- **Velocidade**: medida (twist do CSV) contra estimada (filtro RTS).
- **Profundidade**: ao longo do tempo, com os saltos de posição assinalados.
- **Vizinhos no referencial do corpo**: azimute, elevação, distância, aproximação, tempo até ao ponto de maior aproximação (TCPA) e distância nesse ponto (CPA, horizonte de 60 s).
- **Alerta de proximidade**: atenção abaixo de 30 m e perigo abaixo de 10 m (limiares em `mission_config.yaml`).

### Vista 4: Distâncias CPA 2D (Evolução Temporal)
- **Gráfico Temporal**: Curvas contínuas de distância de separação entre cada par de robôs ao longo do tempo de missão.
- **Limiar Crítico**: Linha tracejada vermelha no valor limite de **10 metros** e cursor vertical azul sincronizado com o tempo atual da simulação.
- **Faixas e mínimos**: faixa de aviso (abaixo de 30 m), mínimo de cada par assinalado e barras que mostram quando cada robot tem dados.
- **Suavizado, bruto ou ambos**: botões à esquerda para ver a posição filtrada (RTS), a do CSV sem filtro, ou as duas.

### Vista 5: Encontros e relatório da missão
- **Resumo**: menor distância da missão, número de encontros (períodos seguidos abaixo de 60 m), alertas e saltos de posição.
- **Tabelas**: janelas de dados por robot, resumo por par, encontros (com nível e aviso se coincidem com um salto de posição) e saltos no registo bruto.
- **Navegação**: o botão de cada linha salta para 10 s antes e abre a Vista Global.
- **Exportação**: relatório em Markdown (.md) e encontros em CSV (.csv).

---

### Vista 6: 1ª Pessoa (dentro do submarino)
- **Câmara**: no nariz do robô selecionado, orientada pela proa, arfagem e rolamento; o próprio casco não se vê.
- **Painel de bordo (tecla H para esconder)**: fita de rumo com a direção do vizinho mais próximo, profundidade, arfagem, rolamento, velocidades, taxa de viragem e caixa do vizinho mais próximo (distância, azimute, elevação, aproximação e alerta).
- **Ver pelo vizinho mais próximo**: botão que passa a câmara para o robô mais próximo do atual.

## 4. Teclas de Atalho Rápidas

| Tecla | Função |
|---|---|
| **`1`** | Seleciona o **Robot 1** (AUV Alpha) como submarino ativo |
| **`2`** | Seleciona o **Robot 4** (AUV Bravo) como submarino ativo |
| **`3`** | Seleciona o **Robot 7** (AUV Charlie) como submarino ativo |
| **`V`** | Alterna ciclicamente entre as 5 Vistas: Global 3D $\to$ 3ª Pessoa $\to$ Cockpit $\to$ CPA 2D $\to$ Encontros e relatório |
| **`F`** / **`G`** | Segue o robot selecionado / volta à vista geral |
| **`P`** | Liga ou desliga a previsão de 60 s |
| **`H`** | Mostra ou esconde o painel de bordo da Vista 1ª Pessoa |
| **`R`** | **Reinicia a reprodução das trajetórias** (reset imediato ao instante inicial) |
| **Espaço** | **Play / Pausa** da animação |

### Níveis de Velocidade de Replay
- **0.25x**: Modo câmara lenta para análise detalhada de cruzamentos.
- **1x**: Velocidade em tempo real.
- **5x**: Avanço rápido 5x.
- **10x / 20x / 50x**: Avanço rápido 10x, 20x e 50x.
- **⚡ Ultra**: Reprodução ultrarrápida da missão completa em cerca de **2 segundos**.

---

## 5. Como Abrir e Visualizar no QGIS (GeoJSON)

O ficheiro `output/mission_trajectories.geojson` é um ficheiro geoespacial padrão com coordenadas WGS-84 (EPSG:4326).

1. Abra o **QGIS**.
2. No menu superior, clique em **Camada** (*Layer*) $\to$ **Adicionar Camada** (*Add Layer*) $\to$ **Adicionar Camada Vetorial** (*Add Vector Layer*).
3. Em *Conjunto de Dados Vetoriais*, selecione o ficheiro:
   `multi_robot_system/output/mission_trajectories.geojson`
4. Clique em **Adicionar**.
5. As trajetórias dos submarinos aparecerão na costa da **Aguçadoura** (Póvoa de Varzim).
6. **Dica de Visualização**:
   - Abra as *Propriedades da Camada* $\to$ *Simbologia*.
   - Escolha o modo de cor baseado no campo `stroke` para ver cada robô com a sua respetiva cor.
   - Adicione uma camada base de satélite (ex.: *Google Satellite* ou *OpenStreetMap* via painel *XYZ Tiles*) para ver a costa marítima real!

---

## 6. Como Abrir e Visualizar no Google Earth (KML 3D)

O ficheiro `output/mission_trajectories_3d.kml` utiliza a extensão padrão OGC `gx:Track` com marcas de tempo e altitude real.

### Google Earth Pro (Desktop):
1. Abra o **Google Earth Pro**.
2. Clique em **Ficheiro** $\to$ **Abrir** e selecione `multi_robot_system/output/mission_trajectories_3d.kml`.
3. A câmara voará automaticamente para a costa norte de Portugal (Aguçadoura).
4. No canto superior esquerdo surgirá a **barra de tempo do Google Earth**:
   - Pode arrastar a barra de tempo ou clicar no botão *Play* do Google Earth para ver os submarinos a percorrer as trajetórias tridimensionais no oceano!

### Google Earth Web:
1. Aceda a [earth.google.com](https://earth.google.com).
2. Vá ao menu lateral de **Projetos**.
3. Clique em **Novo Projeto** $\to$ **Importar ficheiro KML do computador**.
4. Selecione `multi_robot_system/output/mission_trajectories_3d.kml`.
