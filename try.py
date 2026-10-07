#!/usr/bin/env python3
"""Script de Teste e Demonstração do Visualizador 3D em 1ª Pessoa (Plotly).

Executa o visualizador 3D imersivo centrado no referencial próprio (Body Frame) de cada robô:
- Teclas '1', '2', '3' para comutar instantaneamente a perspetiva de observação
- Slider temporal de Replay
- Exporta e abre 'output/mission_body_view_3d.html'
"""

import sys
from pathlib import Path

# Garantir que a raiz do projeto está no sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.plotly_visualizer import launch_interactive_3d

if __name__ == "__main__":
    print("A iniciar Visualizador 3D Interativo (Plotly)...")
    print("Pressione '1', '2' ou '3' na janela do navegador para mudar o robô observador.")
    launch_interactive_3d(config_path="config/config.yaml", open_browser=True)
