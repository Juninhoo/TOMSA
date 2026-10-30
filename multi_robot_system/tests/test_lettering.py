"""Testes do texto INESC TEC desenhado no casco do submarino."""
import math

from multi_robot_system.visualization.plotly_views import build_submarine_lettering, _sub_radius


def test_lettering_tem_pontos_e_separadores():
    """Verifica que o texto tem pontos e separadores entre traços."""
    t = build_submarine_lettering()
    assert len(t["x"]) == len(t["y"]) == len(t["z"]) > 50
    assert None in t["x"]


def test_lettering_fica_na_superficie_do_casco():
    """Verifica que todos os pontos ficam colados ao casco (raio 1,15 m + 2 cm) e que há texto dos dois lados."""
    t = build_submarine_lettering()
    for x, y, z in zip(t["x"], t["y"], t["z"]):
        if x is None:
            continue
        r = math.hypot(y, z)
        assert abs(r - 1.17) < 0.05
    assert any(y > 0 for y in t["y"] if y is not None) and any(y < 0 for y in t["y"] if y is not None)
