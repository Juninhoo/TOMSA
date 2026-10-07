# TOMSA 2026 — Índice da Documentação

Esta pasta centraliza toda a documentação técnica, científica e pedagógica do projeto **TOMSA 2026** (*Multi-Robot Trajectory Viewer & Mission Analyzer*).

---

## 📚 Estrutura da Documentação

```text
docs/
├── relatorio_tomsa.tex           # Código-fonte LaTeX do Relatório Científico Oficial (14 páginas)
├── relatorio_tomsa.pdf           # PDF compilado do Relatório Científico Oficial
├── figures/                      # Figuras vetoriais de alta resolução utilizadas no relatório
│   ├── coordinate_frames_diagram.pdf
│   ├── global_view_trajectories.pdf
│   ├── local_view_body_frame.pdf
│   ├── aer_relative_profiles.pdf
│   ├── proximity_analysis.pdf
│   └── mission_trajectories_3d.pdf
├── enunciado/                    # Documentos oficiais e especificações da missão ISEP/INESC TEC
│   ├── TOMSA2026_TL_readme.pdf   # Enunciado do Trabalho Letivo
│   └── readmelogs.txt            # Especificação técnica dos ficheiros de odometria ROS
├── entrega1/                     # Documentação de submissão da Entrega 1
│   └── TOMSA_TL1_ENTREGA1.md     # Relatório em Markdown com fundamentação e bibliotecas
└── manuais/                      # Guias operacionais e fundamentações teóricas
    ├── MANUAL_USUARIO.md         # Manual do utilizador do dashboard Web e sistema subaquático
    ├── guia_execucao_passo_a_passo.pdf / .tex
    ├── especificacao_requisitos_arquitetura.pdf / .tex
    └── fundamentacao_matematica_completa.pdf / .tex
```

---

## 🛠️ Como Recompilar os Documentos LaTeX

Para compilar o relatório principal com o compilador **Tectonic**:
```bash
/home/geovani/.local/bin/tectonic docs/relatorio_tomsa.tex
```
O PDF atualizado será gravado em `docs/relatorio_tomsa.pdf`.
