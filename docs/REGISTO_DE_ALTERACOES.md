# Registo de alterações ao visualizador (TOMSA 2026)

Tudo o que foi mudado no visualizador multi-robot desde que se começou a pedir melhorias, por ordem. Cada ponto diz o que mudou, em que ficheiros e onde se vê no ecrã. Os dados (CSV) nunca foram alterados.

Como ver tudo: `PYTHONPATH=. .venv/bin/python3 multi_robot_system/main.py --web` e abrir `http://127.0.0.1:8050`.
Como testar: `PYTHONPATH=. .venv/bin/python3 -m pytest multi_robot_system/tests -q` (26 testes).

## Fase 1: Vista Global 3D e Vista 3ª Pessoa

1. **Eixos do corpo x, y, z.** Cada robot tem três setas coladas ao corpo, que rodam com roll, pitch e yaw: x vermelho (proa), y verde (estibordo), z azul (quilha). As setas são as colunas da matriz de rotação, a mesma que se usa para passar vetores do referencial NED para o do corpo. Ficheiros: `plotly_views.py` (`compute_body_axes`, traços das setas) e `web_server.py` (`bodyAxesNED`, `updateScene`).
2. **Câmara que segue o robot.** Botão «Seguir (F)»: a Vista Global fica centrada no robot selecionado e mantém o afastamento e o ângulo que o utilizador escolher com o rato. Botão «Vista geral (G)» volta à vista de cima com os três robots. Ao escolher um robot (botões ou teclas 1, 2, 3) a câmara passa a segui-lo. Ficheiro: `web_server.py` (`followCamera`, `setFollow`, `overviewCamera`).
3. **Aviso de proximidade.** Faixa amarela «ATENÇÃO: ROBOTS PRÓXIMOS» quando dois robots ativos estão a menos de 30 m e faixa vermelha a piscar «PERIGO DE COLISÃO» a menos de 10 m, nas vistas Global e 3ª Pessoa. Limiares em `spatial` no `mission_config.yaml` (`collision_threshold_m` e `proximity_warning_m`). Ficheiro: `web_server.py` (`updateProximityBanner`).
4. **Marcas na barra do tempo.** Pequenos blocos amarelos e vermelhos por baixo da barra do tempo nos períodos em que dois robots estiveram abaixo dos limiares; um clique salta para 10 s antes. (Esta alteração foi aplicada na cópia local e só entra no repositório quando for feita a commit.)

## Fase 2: Cockpit novo (separador 3)

5. **Cockpit refeito de raiz.** O Cockpit antigo era uma figura Plotly com instrumentos de avião (horizonte artificial, velocímetro, profundímetro) que não tinham sentido para um AUV. O novo é um painel desenhado no browser (SVG) e mostra o que interessa para navegar:
   - atitude 3D do robot com os eixos x, y, z, os eixos N, E, D, rolamento, arfagem e guinada em destaque e o quaternião;
   - rumo (heading) numa bússola e rota (course) só quando a velocidade é superior a 0,3 m/s, porque abaixo disso a rota é ruído;
   - velocidade medida (twist do CSV) contra velocidade estimada (filtro RTS);
   - profundidade ao longo do tempo, com os saltos de posição assinalados;
   - vizinhos no referencial do corpo: azimute, elevação, distância, aproximação, tempo até ao ponto de maior aproximação (TCPA) e distância nesse ponto (CPA, horizonte de 60 s, como em `proximity.py`);
   - alerta de proximidade com os mesmos limiares.
   Aviso «Robot X sem dados neste instante» com botão para ir ao início dos dados desse robot. Ficheiros: `web_server.py` (funções `ck...`), `odom_loader.py` e `trajectory.py` (ver ponto 6).
6. **Dados novos no carregador.** `odom_loader.py` passou a extrair a velocidade medida no corpo (`twist.twist.linear`), a incerteza horizontal declarada (da covariância) e os **saltos de posição** do registo bruto (mudanças que implicariam mais de 2,5 m/s). Saltos encontrados: Robot 1 aos 6715,6 s (17,6 m) e 6724,1 s (5,5 m); Robot 4 aos 4141,6 s (3,2 m); Robot 7 aos 4006,4 s (27,7 m, a 30,8 m de profundidade) e 6894,3 s (44,6 m). `RobotTrajectory` ganhou os campos `vel_body_meas`, `sigma_h_decl`, `jump_events`.
7. **Nomes completos.** Todas as siglas nos instrumentos passaram a nome completo (Profundidade, Arfagem (pitch), Rolamento (roll), Rumo, e assim por diante).

## Fase 3: Gráfico de distâncias (separador 4)

8. **Gráfico refeito.** Uma cor por par, nome do par no fim da linha, mínimo de cada par assinalado, faixas de aviso (abaixo de 30 m) e de perigo (abaixo de 10 m), barras que mostram quando cada robot tem dados, botões de escala (0 a 400 m, ampliação 0 a 60 m, logarítmica). A distância é 3D. Foi verificado contra um cálculo independente em ECEF (diferença de 1 a 12 cm). Mínimos: 27,7 m (Robot 1 e Robot 4, aos 3250 s), 175,0 m (Robot 1 e Robot 7), 206,7 m (Robot 4 e Robot 7). O gráfico parecia errado por três razões: nas partes em que só o Robot 7 tem dados só há uma linha, as cores dos pares eram parecidas e a escala ia até 400 m. Ficheiro: `plotly_views.py` (`create_proximity_timeline_figure`).

## Fase 4: Lote final

9. **Origem oficial como opção.** No `mission_config.yaml`, `origin.mode` aceita `data_mean` (origem calculada dos dados, a que estava em uso), `surface_reference` ou `fixed` (as duas últimas usam a origem oficial do enunciado). O ecrã mostra no rodapé a origem em uso. A escolha muda a profundidade em cerca de 1 m e desloca as posições NED cerca de 33 m a Norte e 57 m a Este; as **distâncias entre robots não mudam**. Ficheiros: `main.py`, `web_server.py`, `mission_config.yaml`.
10. **Separador 5, «Encontros e relatório».** Menor distância da missão, janelas de dados por robot, resumo por par (distância mínima, quando, tempo abaixo de 60, 30 e 10 m), lista de encontros (períodos seguidos abaixo de 60 m, com nível e aviso se coincidem com um salto de posição) e lista de saltos. Cada linha tem um botão que salta para 10 s antes e abre a Vista Global. Botões para descarregar o relatório em Markdown e os encontros em CSV. A linha do encontro atual fica realçada. Ficheiros novos: `core/data/mission_report.py` e `tests/test_mission_report.py` (2 testes). A tecla V passa por 5 vistas.
11. **Posição bruta contra suavizada.** O gráfico de distâncias tem botões «Suavizado (RTS)», «Bruto (CSV)» e «Ambos». Para isso `RobotTrajectory` guarda `pos_ned_raw`. Ficheiros: `trajectory.py`, `main.py`, `plotly_views.py`.
12. **Previsão de 60 s.** Linha tracejada na cor de cada robot, com a posição que teria daqui a 60 s se mantivesse a velocidade estimada (termina à superfície se fosse sair da água). Botão «Previsão 60 s» ou tecla P. Vistas Global e 3ª Pessoa. Ficheiros: `plotly_views.py` (traços 2 + 7N + i) e `web_server.py`.
13. **Painel de bordo de submarino (retirado na fusão).** Tinha sido feito um painel por cima da câmara de 1ª pessoa, mas os colegas substituíram essa vista pela Vista 3ª Pessoa (radar e mini-cockpit), que ficou. O código do painel foi retirado e o resto (separador 5, previsão, submarino 3D, atitude no Cockpit) mantém-se.
13b. **Vista 1ª Pessoa reposta como separador 6.** Na fusão com o trabalho dos colegas a 1ª pessoa tinha sido retirada porque a Vista 3ª Pessoa deles ocupa o separador 2. Voltou como separador próprio, com o painel de bordo (fita de rumo, profundidade, arfagem, rolamento, velocidades e vizinho mais próximo). Ficheiro: `web_server.py` (`setFirstPersonCamera`, `updateFirstPersonHud`).
14. **Limiares do gráfico e rodapé.** O gráfico de distâncias e o rodapé passaram a usar os limiares do `mission_config.yaml` em vez de 10 m e 30 m escritos no código.

## O que se descobriu nos dados (e não é um erro do programa)

- A origem do programa (`data_mean`) não é a Surface Reference do enunciado; ver ponto 9.
- O Robot 7 tem `frame_id` «WGS84» e os outros «world_ned»; a posição vem sempre em latitude, longitude e altura.
- A coluna `Time` e o `header.stamp` diferem entre 80 e 220 ms.
- O Robot 7 está quase parado (velocidade mediana 0,07 m/s) e a velocidade medida é negativa em 37 % das amostras.
- A incerteza declarada na covariância é cerca de 10 cm, mas há saltos de 3 a 45 m; a covariância não os prevê.
- O rumo concorda com a rota quando o robot anda (diferença mediana de +0,8° no Robot 1 e −1,4° no Robot 4), o que confirma eixos e quaterniões.
- Janelas de dados (desde o início da missão): Robot 7 de 0 a 133 min 33 s; Robot 4 de 20 min 0 s a 69 min 55 s; Robot 1 de 29 min 59 s a 114 min 49 s.

## Notas

- `cockpit_view.py` (o Cockpit antigo em Plotly) ficou no repositório sem ser usado.
- Os scripts de aplicação (`aplicar_*.py`) correm uma única vez e não tocam em git.
