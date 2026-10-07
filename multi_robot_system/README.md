# TOMSA 2026 - Multi-Robot Subsea Trajectory Viewer & Mission Analyzer

Sistema unificado e profissional em Python para processamento, sincronização, análise geométrica e visualização tridimensional de múltiplos veículos autónomos submarinos (AUVs/ASVs) com telemetria real do **INESC TEC (Aguçadoura)**.

## Funcionalidades Chave
1. **Frontend Principal em Plotly**: Dashboard interativo no navegador com Dark Mode tático subaquático.
2. **4 Vistas Integradas**:
   - **Vista Global 3D**: Visão panorâmica com batimetria (fundo do mar), plano da água e rasto progressivo color-coded por robô.
   - **Vista POV (1ª Pessoa)**: Câmara virtual no bico do submarino ativo com orientação 3D em tempo real.
   - **Vista Cockpit / Sala de Controlo**: Painel instrumental avançado com horizonte artificial, bússola, profundímetro, velocímetro longitudinal, taxa de viragem e radar tático AER.
   - **Vista Distâncias CPA 2D**: Gráfico da evolução temporal da distância de separação com linha limite de segurança (< 10 metros).
3. **Filtro de Kalman & Suavizador RTS**: Reconstrução estocástica ótima do estado cinemático e velocidades a partir de dados reais de odometria ROS, eliminando o ruído de medição.
4. **Análise de Proximidade & Segurança CPA**: Cálculo analítico do Ponto de Maior Aproximação (*Closest Point of Approach*) com alerta visual imediato a vermelho para distâncias inferiores a **10 metros**.
5. **Controlo Temporal Avançado**: Slider de tempo, Play/Pausa e 7 níveis de velocidade (`0.25x` Lento, `1x` Normal, `5x`, `10x`, `20x`, `50x` e `⚡ Ultra` que reproduz toda a missão em ~2 segundos).
6. **Exportação Geoespacial & Bundle HTML**: Gerador de bundle **HTML 100% offline** (`mission_dashboard_bundle.html`), **GeoJSON** para QGIS e **KML 3D** com marcas de tempo (*gx:Track*) para Google Earth.
7. **Rigor Matemático**: 24 testes unitários automatizados cobrindo WGS-84, quaterniões em $SO(3)$, SLERP, atenuação de ruído RTS e soluções de CPA.

## Como Executar

```bash
# Iniciar o dashboard web e abrir o navegador:
PYTHONPATH=. .venv/bin/python3 multi_robot_system/main.py --web

# Gerar o bundle HTML autónomo offline (abre em qualquer computador sem servidor):
PYTHONPATH=. .venv/bin/python3 multi_robot_system/main.py --bundle

# Executar a suite completa de 24 testes unitários (pytest):
PYTHONPATH=. .venv/bin/python3 multi_robot_system/main.py --test

# Exportar GeoJSON, KML 3D e CSV sincronizado:
PYTHONPATH=. .venv/bin/python3 multi_robot_system/main.py --export

# Executar tudo em sequência:
PYTHONPATH=. .venv/bin/python3 multi_robot_system/main.py --all
```

## Teclas de Atalho Rápidas
- **1, 2, 3**: Alterna o submarino ativo (`Robot 1`, `Robot 4` ou `Robot 7`).
- **V**: Alterna ciclicamente entre as 4 Vistas (`Global 3D` $\to$ `POV` $\to$ `Cockpit` $\to$ `Distâncias CPA 2D`).
- **R**: **Reinicia a reprodução das trajetórias** (reset imediato ao início).
- **Espaço**: Play / Pausa.

Consulte o manual detalhado em: [docs/MANUAL_USUARIO.md](docs/MANUAL_USUARIO.md).
