# Vista em 1ª pessoa (POV de cada drone) — Plotly

Visualizador 3D onde a câmara é o ponto de vista de cada drone. Carregar **1, 2, 3…** muda de drone.
Python = aplicação e visualização. C++ (OOP) = toda a matemática (módulo `uwmath`).

## Instalar e correr (comandos numa só linha)
    pip install plotly numpy pybind11 setuptools
    python setup.py build_ext -b python
    python python/test_uwmath.py
    python python/viewer_pov.py --csv robot1_s1.csv robot4_s1.csv robot7_s1.csv

Sem `--csv` corre 3 missões de demonstração. Aceita CSV ROS Odometry (x=latitude, y=longitude, z=altura elipsoidal, quaternion corpo→NED) ou `t,x,y,z` em NED.
O programa abre um servidor local (http://127.0.0.1:8765) e o browser; Ctrl+C para sair.

Opções: `--radius` (alcance em m), `--fov 70` (HUD), `--dt 2` (passo do slider, s), `--surface-h` (altura da superfície do mar), `--seabed` (grelha do fundo), `--port`, `--no-open`.
Em Windows é preciso um compilador C++ (MSVC Build Tools); em Linux/macOS basta g++/clang.

## Teclas
| Tecla | Ação |
|---|---|
| 1–9 | vista do drone N |
| Espaço | play / pausa |
| ← → | passo de tempo |
| R | repor a câmara a olhar em frente |
| + / − | velocidade do replay |

O rato continua a rodar/zoom a câmara; **R** (ou mudar de drone) volta a olhar para a frente.

## Estrutura
- `cpp/include/uwmath.hpp` — Vec3, Mat3, rotações (ZYX, Rodrigues), Pose, Camera (pinhole, matriz K 3×3), Trajectory
- `cpp/src/bindings.cpp` — pybind11 (inclui versões em lote para arrays numpy)
- `python/viewer_pov.py` — servidor local + página Plotly (cada vista é calculada a pedido no C++)
- `python/missions.py` — 3 missões de demonstração + leitor de CSV
- `python/test_uwmath.py` — testes (R^T R = I, ida-e-volta, exemplo da secção 8 do PDF, etc.)
- `docs/fundamentos_matematicos.pdf` (+ `.tex`) — dedução passo a passo de toda a matemática

## Como funciona (resumo)
Para o drone k no instante t, todos os pontos do mundo (rasto dos outros drones, modelo dos drones,
grelha do fundo e da superfície) passam para o referencial da câmara: `p_c = R_wc^T (p_w − t_wc)`.
Mapeia-se para os eixos do Plotly (X direita, Y frente, Z cima) e a câmara do Plotly fica na origem a olhar para +Y.
O HUD mostra, para cada outro drone, distância, azimute/elevação e pixel via `Camera.project` (C++).

## Limitações conhecidas
- O FOV da renderização é o fixo do Plotly; o `--fov` só afeta o teste "dentro do campo" do HUD.
- Com CSV de odometria usa-se o quaternion medido; só com posições, a orientação sai da velocidade (roll = 0).
- robot7: o yaw medido aponta ~180° ao contrário do sentido de movimento (convenção do frame "WGS84" ou marcha-atrás); usado tal como está.
- Os CSV dizem `world_ned` mas as posições são lat/lon/altura; convertidas para NED (origem = centroide, superfície = h estimada).
