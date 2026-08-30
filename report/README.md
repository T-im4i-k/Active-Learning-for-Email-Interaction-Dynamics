# Refinements to the SAE–Thompson Sampling Framework for Active Email Outreach

LaTeX source for the short paper extending Žid, Alves & Kordík (CIKM '25),
*Active Recommendation for Email Outreach Dynamics*.

## Building

Requires a standard TeX Live install (`pdflatex`, `bibtex`, `natbib`,
`hyperref`, `booktabs`, `microtype`).

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

or simply:

```bash
latexmk -pdf main.tex
```

## Project structure

```
.
├── main.tex                 Root document: preamble, title, section includes, bibliography
├── preamble.tex              Package imports and custom macros
├── references.bib            Bibliography database (original CIKM paper)
├── README.md
└── sections/
    ├── introduction.tex      Section 1
    ├── recap.tex             Section 2 — recap of the original model
    ├── refinements.tex       Section 3 — pulls in the seven subsections below
    ├── refinements/
    │   ├── alpha_scheduling.tex       3.1 Dynamic alpha-scheduling
    │   ├── recency_weighted_phi.tex   3.2 Recency-weighted historic rate
    │   ├── forward_pass_f.tex         3.3 Forward-pass f_j(t)
    │   ├── variance_based_p.tex       3.4 Variance-based p_j
    │   ├── factored_score.tex         3.5 Factored score s_j(t)
    │   ├── tto.tex                    3.6 Time-to-open incorporation
    │   └── deep_autoencoder.tex       3.7 Deep autoencoder
    ├── experiments.tex       Section 4 — methodology, results table, discussion
    └── future_work.tex       Section 5

```

Fill in `\author{}` in `main.tex` with your name/affiliation before submitting
or sharing.
