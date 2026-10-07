# TÓPICOS DE MATEMÁTICA PARA SISTEMAS AUTÓNOMOS
## TRABALHO LETIVO - ENTREGA 1

> **Nota 1:** Todos elementos do grupo são responsáveis por esta entrega  
> **Nota 2:** Todos elementos do grupo tomaram conhecimento das regras estabelecidas para o funcionamento da u.c. no [Moodle do ISEP](https://moodle.isep.ipp.pt/)

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

---

## 1. Identificação dos objetivos e estruturação do problema
[De volta para "Conteúdo"](#conteúdo)

### 1.1 Objetivo Geral
O presente projeto tem como objetivo projetar, desenvolver e validar rigorosamente uma aplicação computacional avançada em **Python** para processamento, sincronização temporal, análise cinemática e visualização tridimensional de múltiplos veículos autónomos submarinos e de superfície (AUVs/ASVs). O sistema utiliza telemetria real de odometria obtida em ambiente marinho pelo **INESC TEC** na costa da **Aguçadoura** (Póvoa de Varzim, Portugal), envolvendo três robôs autónomos (`Robot 1`, `Robot 4` e `Robot 7`).

A aplicação realiza a conversão de coordenadas geodésicas de navegação (WGS-84: latitude, longitude, altitude elipsoidal e quaterniões de atitude em $SO(3)$) para referenciais cartesianos locais tangentes à Terra (**NED** - *North-East-Down* e **ENU** - *East-North-Up*), calcula a projeção dinâmica no referencial do corpo (**Body Frame**) do veículo ativo e extrai grandezas relativas esféricas de visada (**AER** - *Azimuth, Elevation, Range*).

### 1.2 Objetivos Específicos e Estruturação do Problema
O problema foi estruturado num conjunto de módulos funcionais desacoplados com critérios claros de aceitação:

| Módulo / Objetivo | Descrição da Funcionalidade | Critério de Aceitação / Rigor Matemático |
|:---|:---|:---|
| **1. Processamento e Sincronização** | Ingestão de ficheiros CSV de odometria do ROS com amostragem assíncrona, eliminação de duplicados e ordenação por *timestamp*. | Determinação da janela comum de interseção temporal ($T_{\text{overlap}} \approx 2396.3\text{ s}$) e reamostragem uniforme a $\Delta t = 1.0\text{ s}$. |
| **2. Conversão Geodésica First-Principles** | Conversão de coordenadas elipsoidais WGS-84 $(\varphi, \lambda, h)$ para ECEF $(X, Y, Z)$ e plano tangente local NED/ENU $(x_N, y_E, z_D)$. | Fórmulas implementadas de raiz com NumPy, validadas contra `pymap3d` com erro absoluto $< 10^{-6}\text{ m}$. |
| **3. Suavização e Atenuação de Ruído** | Aplicação de Filtro de Kalman de Velocidade Constante e Suavizador RTS (*Rauch-Tung-Striebel*) de duas passagens. | Redução drástica da variância do ruído de odometria e estimativa contínua das velocidades longitudinais e taxas de viragem (*yaw rate*). |
| **4. Visualização Multi-Vista Plotly** | Dashboard tridimensional interativo no navegador com 4 Vistas: **Global 3D** (com batimetria e rasto progressivo), **POV 1ª Pessoa** (câmara no nariz do robô), **Cockpit / Sala de Controlo** (instrumentos analógicos) e **Distâncias CPA 2D**. | Comutação instantânea de vistas, suporte a teclas de atalho (<kbd>1</kbd>, <kbd>2</kbd>, <kbd>3</kbd>, <kbd>V</kbd>, <kbd>R</kbd>, <kbd>Espaço</kbd>) e múltiplos níveis de velocidade até `⚡ Ultra` (~2s). |
| **5. Análise de Segurança e CPA** | Monitorização da separação espacial inter-robôs e cálculo analítico em forma fechada do Ponto de Maior Aproximação (*Closest Point of Approach*). | Alerta automático de colisão com destaque visual a vermelho quando a distância cai abaixo do limiar crítico de **10 metros**. |
| **6. Exportação Geoespacial & Bundle Offline** | Exportação de dados para SIG em **GeoJSON** (QGIS), **KML 3D** (*gx:Track* para Google Earth) e bundle **HTML autónomo 100% offline**. | Ficheiros auto-contidos de alta fidelidade prontos para análise por equipas de campo sem dependência de internet. |

### 1.3 Modelação Matemática das Conversões
1. **Modelo da Terra WGS-84 & Raio Curvatura Normal**:
   $$N(\varphi) = \frac{a}{\sqrt{1 - e^2 \sin^2 \varphi}}$$
   onde $a = 6\,378\,137.0\text{ m}$ e $e^2 = 2f - f^2 \approx 0.00669438$.

2. **Conversão Geodésico $\to$ ECEF**:
   $$X = (N(\varphi) + h) \cos \varphi \cos \lambda, \quad Y = (N(\varphi) + h) \cos \varphi \sin \lambda, \quad Z = \left(N(\varphi)(1 - e^2) + h\right) \sin \varphi$$

3. **Matriz de Rotação ECEF $\to$ ENU / NED**:
   Dada a origem de referência $(\varphi_0, \lambda_0, h_0)$:
   $$\begin{bmatrix} x_E \\ y_N \\ z_U \end{bmatrix} = \begin{bmatrix} -\sin\lambda_0 & \cos\lambda_0 & 0 \\ -\sin\varphi_0 \cos\lambda_0 & -\sin\varphi_0 \sin\lambda_0 & \cos\varphi_0 \\ \cos\varphi_0 \cos\lambda_0 & \cos\varphi_0 \sin\lambda_0 & \sin\varphi_0 \end{bmatrix} \begin{bmatrix} X - X_0 \\ Y - Y_0 \\ Z - Z_0 \end{bmatrix}$$
   Para NED: $x_N = y_N^{\text{ENU}}$, $y_E = x_E^{\text{ENU}}$, $z_D = -z_U^{\text{ENU}}$.

4. **Transformação para o Referencial do Corpo (Body Frame)**:
   Seja $\mathbf{q} = (q_x, q_y, q_z, q_w)$ o quaternion unitário de atitude do robô observador $i$. A posição de um robô vizinho $j$ no Body Frame de $i$ é:
   $$\mathbf{p}_j^{\text{Body}(i)} = \mathbf{R}_{B/NED}^T \left(\mathbf{p}_j^{NED} - \mathbf{p}_i^{NED}\right) = \begin{bmatrix} x_{\text{proa}} \\ y_{\text{estibordo}} \\ z_{\text{quilha}} \end{bmatrix}$$

---

## 2. Justificação da relevância das bibliotecas Python selecionadas
[De volta para "Conteúdo"](#conteúdo)

A escolha das bibliotecas Python foi pautada por critérios estritos de eficiência computacional, maturidade da comunidade científica, desacoplamento e suporte a pipelines numéricos vetorizados:

| Biblioteca | Papel no Sistema | Justificação Técnica e Vantagens Competitivas |
|:---|:---|:---|
| **NumPy** | Cálculo Matricial e Álgebra Linear | Permite a execução vetorizada em linguagem C de todas as transformações tridimensionais, rotações de quaternions e filtros de Kalman para dezenas de milhares de amostras sem ciclos lentos em Python. |
| **SciPy** | Suavização e Interpolação SLERP | Fornece algoritmos numéricos estabilizados para resolução de equações diferenciais, álgebra linear avançada e interpolação esférica linear (`Slerp`) de quaternions de atitude no grupo $SO(3)$. |
| **Pandas** | Ingestão e Estruturação Tabular | Utilizado no carregamento rápido dos ficheiros CSV de odometria do ROS, ordenação estrita por marcas temporais, limpeza de duplicados e indexação temporal. |
| **Plotly** | Engine de Visualização 3D e Dashboard Web | Biblioteca principal de apresentação gráfica. Permite renderização 3D com aceleração WebGL, rotação interativa de câmara, criação de mostradores analógicos no Cockpit e geração de ficheiros HTML 100% autónomos offline. |
| **PyYAML** | Configuração Declarativa Externa | Garante a separação total entre os dados/parâmetros da missão (`mission_config.yaml`) e o código-fonte executável, facilitando alterações de origem geodésica ou limiares de segurança sem alterar o código. |
| **pytest** | Suite de Testes Unitários Automatizados | Garante a integridade matemática do sistema através de 24 testes unitários automatizados, assegurando tolerâncias numéricas estritas ($< 10^{-6}\text{ m}$) e impedindo regressões em edições futuras. |
| **pymap3d** / **pyproj** | Autoridade Externa de Validação Numérica | Bibliotecas padrão de geodesia utilizadas **exclusivamente como autoridade de teste** nos ensaios automatizados de validação cruzada das nossas fórmulas de raiz. |

> **Decisão de Arquitetura:** Todas as equações e transformações geométricas foram implementadas de raiz (*first-principles*) pela equipa utilizando NumPy, garantindo o domínio concetual e matemático total do problema.

---

## 3. Metodologia proposta para o desenvolvimento do projeto
[De volta para "Conteúdo"](#conteúdo)

### 3.1 Abordagem Metodológica e Fases de Desenvolvimento
O projeto foi conduzido segundo uma metodologia **Iterativa e Incremental (Agile/Clean Architecture)**, estruturada nas seguintes etapas calendarizadas:

```
                  ┌──────────────────────────────────────────────────┐
                  │ 1. Especificação & Modelação Matemática          │
                  │    - Dedução das equações WGS-84, ECEF, NED, AER │
                  │    - Estruturação do repositório modular         │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │ 2. Núcleo Matemático & Suavização RTS            │
                  │    - Implementação de geodesy.py e rotations.py  │
                  │    - Filtro de Kalman & Suavizador RTS ótimo     │
                  │    - Validação por 24 testes unitários com pytest│
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │ 3. Motor de Renderização 3D & Cockpit            │
                  │    - Construção das 4 Vistas em Plotly           │
                  │    - Câmara POV em 1ª Pessoa orientável          │
                  │    - Instrumentos analógicos e Radar 360°        │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │ 4. Controlo Temporal & Exportação SIG            │
                  │    - Replay a 60 FPS com velocidade 0.25x a 50x  │
                  │    - Bundle HTML offline, GeoJSON e KML 3D       │
                  │    - Documentação e backup completo no sistema   │
                  └──────────────────────────────────────────────────┘
```

### 3.2 Arquitetura de Software Desacoplada (Clean Architecture)
A estrutura de pastas do projeto `multi_robot_system/` reflete o desacoplamento rigoroso entre modelo matemático, camada de dados, apresentação gráfica e exportação:

```text
multi_robot_system/
├── config/
│   └── mission_config.yaml         # Configuração declarativa da missão (origem, robôs, limiares)
├── data/
│   ├── robot1_s1.csv               # Odometria real do Robot 1 (INESC TEC Aguçadoura)
│   ├── robot4_s1.csv               # Odometria real do Robot 4 (INESC TEC Aguçadoura)
│   └── robot7_s1.csv               # Odometria real do Robot 7 (INESC TEC Aguçadoura)
├── core/                           # Núcleo matemático puro sem dependências de UI
│   ├── data/
│   │   ├── odom_loader.py          # Parser e ordenação temporal de ficheiros CSV
│   │   └── trajectory.py           # Classes RobotTrajectory e MissionDataset
│   └── math/
│       ├── geodesy.py              # Conversões WGS-84, ECEF, NED, ENU, AER (first-principles)
│       ├── rotations.py            # Quaterniões em SO(3), Euler ZYX e SLERP
│       ├── kalman_rts.py           # Filtro Kalman CWNA e Suavizador RTS ótimo
│       ├── proximity.py            # Análise de separação espacial e cálculo de CPA (< 10m)
│       └── interpolators.py        # Comparativo entre interpolação linear, spline e RTS
├── visualization/                  # Camada de apresentação gráfica e servidor web
│   ├── plotly_views.py             # Renderizador 3D para Vista Global e POV (1ª Pessoa)
│   ├── cockpit_view.py             # Mostradores analógicos, horizonte artificial e radar
│   └── web_server.py               # Servidor HTTP multithreaded com dashboard interativo
├── io_export/                      # Módulo de exportação de dados e relatórios
│   ├── geojson_exporter.py         # Exportador SIG GeoJSON (QGIS)
│   ├── kml_exporter.py             # Exportador KML 3D com gx:Track (Google Earth)
│   └── benchmark_exporter.py       # Gerador de tabelas LaTeX e análises de desempenho
├── tests/                          # 24 testes unitários automatizados (pytest)
├── output/                         # Resultados gerados (HTML bundle, KML, GeoJSON, CSV)
└── main.py                         # Ponto de entrada CLI unificado
```

### 3.3 Divisão de Responsabilidades na Equipa
O trabalho foi distribuído de forma equilibrada e colaborativa entre os elementos do grupo:

| Membro da Equipa | Responsabilidades Principais | Atividades de Validação Cruzada |
|:---|:---|:---|
| **Geovani Lopes** (Nº 1131518) | Arquitetura geral do sistema, implementação do núcleo de geodesia (`geodesy.py`), quaternions (`rotations.py`) e desenvolvimento das vistas 3D em Plotly (`plotly_views.py`, `web_server.py`). | Testes unitários com Pytest, validação de tolerâncias geodésicas contra `pymap3d` e relatórios. |
| **Luis Lei** (Nº 1120542) | Desenvolvimento do módulo de ingestão de odometria (`odom_loader.py`), filtro de Kalman e suavizador RTS (`kalman_rts.py`), e sincronização temporal. | Auditoria de ruído de medição, calibração do modelo de covariância e validação dos dados dos CSVs. |
| **Hugo Amado** (Nº 1261726) | Implementação do Cockpit de instrumentos analógicos (`cockpit_view.py`), cálculo analítico de CPA (`proximity.py`) e exportadores GeoJSON/KML 3D (`io_export`). | Testes de usabilidade dos atalhos de teclado, verificação de mapas no QGIS/Google Earth e documentação. |

---
