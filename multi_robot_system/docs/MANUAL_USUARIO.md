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
│   ├── plotly_views.py             # Motor 3D da Vista Global e da Vista POV (1ª Pessoa)
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

## 3. As Quatro Vistas do Sistema

O utilizador pode alternar entre as 4 vistas através dos botões no cabeçalho ou premindo a tecla **`V`**:

### Vista 1: Global 3D (Visão Panorâmica)
- **Cenário**: Representa todos os submarinos num referencial local comum NED com profundidade real.
- **Batimetria**: Fundo do mar em grelha cinzenta e superfície da água como plano azul translúcido ($D = 0\,\text{m}$).
- **Rasto Progressivo**: A trajetória inicia sem traço e vai sendo desenhada gradualmente à medida que o tempo avança.
- **Modelos 3D Wireframe**: Cada veículo é representado por um modelo 3D com nariz, asas e leme dorsal orientado em 3D conforme yaw, pitch e roll.

### Vista 2: POV (1.ª Pessoa / Câmara Submarina)
- **Perspetiva**: Câmara virtual posicionada no nariz do submarino ativo (Robot 1, 4 ou 7), orientada pela sua proa e pitch.
- **Cena Tridimensional**: Visualização relativas dos outros veículos no mar com o casco do próprio veículo oculto para visibilidade total.

### Vista 3: Cockpit / Sala de Controlo (Painel de Instrumentos)
- **Horizonte Artificial e Bússola**: Ângulos de Roll, Pitch e Heading (0° a 360° com rosa-dos-ventos).
- **Profundímetro Digital**: Mostrador de cota vertical com linha de fundo.
- **Velocímetro de Avanço**: Velocidade linear orientada na direção do movimento longitudinal.
- **Taxa de Viragem (Yaw Rate)**: Indicador de velocidade angular em graus por segundo durante as curvas.
- **Radar Tático AER**: Ecrã polar que projeta a posição relativa de todos os submarinos circundantes.
- **Painel Anunciador de Segurança CPA**: Alerta a vermelho quando a distância de aproximação for inferior a **10 metros**.

### Vista 4: Distâncias CPA 2D (Evolução Temporal)
- **Gráfico Temporal**: Curvas contínuas de distância de separação entre cada par de robôs ao longo do tempo de missão.
- **Limiar Crítico**: Linha tracejada vermelha no valor limite de **10 metros** e cursor vertical azul sincronizado com o tempo atual da simulação.

---

## 4. Teclas de Atalho Rápidas

| Tecla | Função |
|---|---|
| **`1`** | Seleciona o **Robot 1** (AUV Alpha) como submarino ativo |
| **`2`** | Seleciona o **Robot 4** (AUV Bravo) como submarino ativo |
| **`3`** | Seleciona o **Robot 7** (AUV Charlie) como submarino ativo |
| **`V`** | Alterna ciclicamente entre as 4 Vistas: Global 3D $\to$ POV $\to$ Cockpit $\to$ CPA 2D |
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
