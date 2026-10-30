"""Teste da deteção de colisão no dashboard."""
from multi_robot_system.main import load_mission
from multi_robot_system.visualization.web_server import MissionWebDashboard


def test_colisao_no_painel():
    """Verifica que a página recebe a distância de contacto (2 m) e inclui a deteção de colisão e o botão 'robots parados'."""
    ds, cfg = load_mission("multi_robot_system/config/mission_config.yaml")
    html = MissionWebDashboard(ds, cfg).build_dashboard_html()
    assert '"contact_distance_m": 2.0' in html
    assert "function detectCollision" in html and "btn-freeze" in html
