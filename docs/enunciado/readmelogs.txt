TOMSA 2026
Disclaimer: Os logs disponibilizados para o trabalho da unidade curricular (u.c.) TOMSA de 2026 são retirados,
filtrados e anonimizados de missões realizadas na Aguçadoura, no âmbito de atividades realizadas pelo INESC
TEC. A usa utilização é restrita à u.c. de TOMSA. Qualquer outro uso não é autorizado.
Esturtura das LOG FILES - formato csv
Cada log file contempla:
Informação temporal
Time: instante associado ao registo no ficheiro.
header.seq: número sequencial da mensagem ROS.
header.stamp.secs e header.stamp.nsecs: timestamp original da mensagem ROS, dividido em segundos e
nanossegundos.
Referenciais
header.frame_id: referencial relativamente ao qual a pose é expressa (por exemplo, map, odom ou world).
child_frame_id: referencial do objeto ou veículo cuja pose está a ser descrita (tipicamente base_link).
Pose
Posição: (pose.pose.position.x, pose.pose.position.y) coordenadas (latitude, longitude) em graus e
(pose.pose.position.z) altitude em metros.
Orientação: Componentes do quaternion que representa a orientação tridimensional do veículo.
Incerteza da pose: Matriz de covariância associada à pose, armazenada como um vetor de 36 elementos. Esta
matriz quantifica a incerteza da estimativa de posição e orientação.
Velocidades
Velocidade linear: Componentes da velocidade linear, normalmente expressas em m/s.
Velocidade angular: Componentes da velocidade angular, normalmente expressas em rad/s.
Incerteza das velocidades: Matriz de covariância da estimativa das velocidades lineares e angulares, igualmente
armazenada como um vetor de 36 elementos.Surface Reference
Referência inicial utilizada no âmbito da missão
latitude: 41.45201666666667 # (Decimal degrees)
longitude: -8.81322222222222 # (Decimal degrees)
altitude: 55.44791126176966 # (Meters above sea level at the surface)
