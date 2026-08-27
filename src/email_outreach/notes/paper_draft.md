# Refinements to the SAE–Bandit Framework for Active Email Recommendation

**Working draft — extends** Žid, C., Alves, R., and Kordík, P. (2025). *Active Recommendation for Email Outreach
Dynamics.* CIKM '25. https://doi.org/10.1145/3746252.3760832

*Tymofii Ivanov, Faculty of Information Technology, CTU Prague*

## Abstract

Žid et al. (2025) propose an active-learning framework for cold-start email campaigns that combines a shallow
autoencoder (SAE), used as a collaborative-filtering signal, with a Thompson Sampling (TS) multi-armed bandit over
recipients. This document reviews that baseline model and proposes a set of refinements to its scoring function,
training signal, and network architecture. Each proposal is validated experimentally on the same recall-based evaluation
protocol as the original paper. We further catalog open directions that were identified but not yet implemented.

## 1. Introduction

Email outreach campaigns face a cold-start problem: a new template has no interaction history, and collecting it is
costly. The baseline model addresses this by treating each recipient as a bandit arm, scoring arms with a SAE-derived
collaborative-filtering signal, and using Thompson Sampling to select successive batches of recipients during an
active-learning phase.

This document has two goals:

1. **Recap** the baseline model precisely enough to serve as a shared reference for all extensions (Section 2).
2. **Propose and evaluate** a series of refinements to the score function, the training objective, and the network
   architecture (Sections 3–5), and separately record ideas that remain untested (Section 4).

Throughout, **proposed models** are fully specified and experimentally evaluated; **future work** items are explicitly
marked as such and kept brief.

## 2. Baseline Model Recap

### 2.1 Problem Setup

Let $X \in \{0,1\}^{n \times m}$ be the interaction matrix, where $n$ is the number of templates, $m$ is the number of
recipients, and $X_{i,j} = 1$ iff recipient $j$ opened template $i$. Given a new template $n+1$, the goal is to predict,
before a deadline $T$, which recipients will open it, using only interactions observed during an active-learning
window $[0, T)$.

### 2.2 Shallow Autoencoder (SAE)

A shallow autoencoder is trained on $X$:

$$\min_{E,D \in \mathbb{R}^{m,d}} \ell\big (X, \sigma (X B_{E,D})\big), \qquad B_{E,D} = ED^\top - \mathrm{diag}\big ([E \odot D]\mathbf{1}\big)$$

where $d$ is the bottleneck size, $\ell$ is element-wise binary cross-entropy, and the diagonal constraint prevents each
recipient from predicting their own entry. We write $f_{SAE} (x) = \sigma (x^\top B_{E,D})$ for the trained forward
pass, mapping a binary open-vector to a vector of opening probabilities.

Define $\Sigma = \sigma (B_{E,D})$. Since $\Sigma_{i,j} = f_{SAE} (e_i)e_j$, entry $\Sigma_{i,j}$ is the probability
recipient $j$ opens a template given that only recipient $i$ already opened it.

> **Convention.** $\Sigma$ is not symmetric ($E \neq D$). We use **row = conditioning recipient (already opened),
column = target recipient (probability being predicted)** — the convention under which $p_j$ and $f_j (t)$ below are
> well-defined. The paper's prose description reads with indices swapped relative to this; its own formulas match the
> convention used here.

### 2.3 Arm Score Function

Each recipient $j$ is scored at active-learning time $t$ as:

$$s_j (t) = \alpha \cdot \phi_j p_j + (1-\alpha) \cdot f_j (t)$$

| Term     | Definition                                  | Meaning                                                               |
|----------|---------------------------------------------|-----------------------------------------------------------------------|
| $\phi_j$ | $\frac{1}{n}\sum_{i=1}^{n} X_{i,j}$         | Historic open rate of recipient $j$                                   |
| $p_j$    | $\frac{1}{m-1}\sum_{i \neq j} \Sigma_{j,i}$ | Influence of $j$ opening on other recipients                          |
| $f_j(t)$ | $\bar x(t)^\top \Sigma_{:,j}$               | Confidence that $j$ opens, given the current opened-state $\bar x(t)$ |

$\alpha \in [0,1]$ trades off the historic/influence term against the current-state term. $s_j (t)$ parameterizes a Beta
distribution used by Thompson Sampling: $\alpha_j (t) = Gs_j (t)$, $\beta_j (t) = G (1-s_j (t))$, with confidence
modifier $G$.

> **Note.** We use the normalized state $\bar x (t) = x (t)/n_t$ (as in the paper's Figure 1), not the raw $x_{n+1} (t)$
> used in the paper's Eq. (4) text. The unnormalized version can push $s_j (t)$ outside $[0,1]$, producing a negative Beta
> parameter.

## 3. Proposed Extensions

Each subsection below is a fully specified, experimentally evaluated modification to the baseline. Results are reported
jointly in Section 5.

### 3.1 Dynamic Alpha Scheduling

The baseline uses a fixed $\alpha$. Since $\phi_j p_j$ reflects exploration (information gain about other recipients)
and $f_j (t)$ reflects exploitation (direct confidence for recipient $j$), $\alpha$ is naturally interpreted as an
exploration/exploitation weight, and a schedule $\alpha (t)$ decreasing from exploration to exploitation over the
active-learning window is a natural extension.

**3.1.1 Sent-mail-based scheduling.** Let $N (t)$ be the number of emails sent by time $t$, $N$ the total,
and $\mu = N (t)/N$. We define two schedules from $l \in [0,1]$ to $r \in [0,1]$:

$$\text{Linear:} \quad \alpha (\mu) = l (1-\mu) + r\mu$$
$$\text{Geometric:} \quad \alpha (\mu) = l^{ (1-\mu)} r^{\mu}$$

**3.1.2 Open-mail-based (confidence) scheduling.** Alternatively, $\alpha$ can track "trust" in $f_j (t)$: as more opens
are observed, $f_j (t)$ becomes a more reliable point-estimate and should be weighted more.
Let $o (t) = x (t)^\top \mathbf{1}$ and $\tilde o = o (t)/m$. We define a hyperbolic decay with dampening
factor $\kappa$ (where $\alpha (\kappa) = \tfrac12$):

$$\alpha (\tilde o) = \frac{\kappa}{\tilde o + \kappa}$$

> **Note.** $\tilde o$ normalizes by the full recipient pool $m$, not by recipients reached so far. Given a ~9% baseline
> open rate, $\tilde o$ stays small through most of active learning, so $\kappa$ must be tuned well below typical eventual
> open rates for this schedule to move meaningfully.

### 3.2 Recency-Weighted Historical Engagement

$\phi_j$ weights every past template equally, ignoring that recipient behavior can drift over time. Assuming templates
are sent at uniform intervals ordered by index, define relative recency $d_i = (n-i)/n$ and weight:

$$\omega_i = 2^{-d_i/h}, \qquad \phi_j = \frac{\omega^\top X_{:,j}}{\sum_{i=1}^n \omega_i}$$

where $h > 0$ is the half-life of the decay. As $h \to \infty$, $\omega_i \to 1$ and $\phi_j$ reduces to the original
uniform average.

### 3.3 Forward-Pass Definition of $f_j (t)$

$f_j (t) = \bar x (t)^\top \Sigma_{:,j}$ is a linear combination of individual per-recipient influences and cannot
capture interactions between simultaneous openers. We propose querying the trained autoencoder directly on the current
state:

$$f_j (t) = \sigma\big (x (t)^\top B_{E,D}\big)e_j = f_{SAE} (x (t))\, e_j$$

using the raw binary $x (t)$ (not $\bar x (t)$). Because $\sigma$ is applied after aggregation rather than before,
scaling the input by $1/n_t$ would shrink the logit and push $f_j (t)$ toward maximum uncertainty as more recipients
open — the opposite of the intended effect — so the un-normalized, in-distribution binary $x (t)$ is used instead.
Since $\sigma$ is nonlinear, this differs from the baseline definition except when exactly one recipient has opened, and
now captures joint (not just individual) influence of all current openers.

### 3.4 Variance-Based Definition of $p_j$

$p_j$ should reward recipients whose opening maximally *reduces uncertainty* in other recipients' $f$-scores — i.e.,
information gain. The baseline definition, $p_j = \frac{1}{m-1}\sum_{i\ne j} \Sigma_{j,i}$, instead rewards recipients
who maximally *increase* other recipients' scores, which is not the same objective. We propose scoring $p_j$ by the
spread (variance) it induces instead of the mean:

$$p_j = \frac{4}{m-1}\sum_{i \ne j} \big (\Sigma_{j,i} - \bar\Sigma_{j,:}\big)^2, \qquad \bar\Sigma_{j,:} = \frac{1}{m-1}\sum_{i \ne j}\Sigma_{j,i}$$

This is the variance of $\Sigma_{j,:}$ (excluding $j$), scaled by 4 so the maximum possible Bernoulli variance (0.25)
maps to 1.

### 3.5 Multiplicative Score Decomposition

The baseline's two terms conflate two independent axes: exploration/exploitation and historic/current data — it is not
clear why exploration should always use historic data. We propose separating these axes explicitly:

$$s_j (t) = \pi_j (t)\, u_j (t)$$

$$\pi_j (t) = \alpha (t)\phi_j + (1-\alpha (t))f_j (t) \quad \text{ (open-probability estimate, historic vs. current)}$$
$$u_j (t) = \beta (t)p_j + (1-\beta (t))\cdot 1 \quad \text{ (utility, direct vs. informational)}$$

$\pi_j (t)$ estimates the probability recipient $j$ opens; $u_j (t)$ weights the direct utility of that open
(constant $1$) against its indirect, information-gain utility ($p_j$). $\alpha (t)$ and $\beta (t)$ can be scheduled
independently using any schedule from Section 3.1, reusing all baseline primitives without modification.

### 3.6 Incorporating Time-to-Open (TTO)

The SAE trains only on the binary $X$, treating a fast open and a slow open identically — despite only fast opens being
actionable within the short active-learning window.

**TTO matrix and decayed label.** Let $\Delta \in (\mathbb{R}_{\ge 0}\cup\{+\infty\})^{n,m}$ record send-to-open time
($+\infty$ for non-opens); $X_{i,j} = \mathbb{1}[\Delta_{i,j} < \infty]$, so $\Delta$ is strictly more expressive
than $X$. Define a decayed label with half-life $h_\delta$:

$$Y_{i,j} = 2^{-\Delta_{i,j}/h_\delta}, \qquad 2^{-\infty} := 0$$

As $h_\delta \to \infty$, $Y \to X$, recovering the binary model. This composes directly with the recency weights of
Section 3.2: $\phi_j = (\omega^\top Y_{:,j})/\sum_i \omega_i$.

**Binary-to-TTO training ($X \to Y$).** The SAE's input stays binary; only the target changes:

$$\min_{E,D} \ell\big (Y, \sigma (X B_{E,D})\big)$$

Because the input space is unchanged, $p_j = \frac{1}{m-1}\sum_{i\ne j}\Sigma^Y_{j,i}$
(with $\Sigma^Y = \sigma (B_{E,D})$) carries over unchanged, and $f_j (t)$ requires no modification at inference. The
interpretation of $\Sigma^Y_{j,i}$ shifts, however, from "probability $i$ opens" to a speed-weighted blend of *whether*
and *how fast* $i$ opens; this shift propagates to $p_j$, $f_j (t)$, and $s_j (t)$.

**TTO cutoff thresholding.** A separate mismatch: SAE training rows reflect *fully converged* open patterns, while
active-learning queries $x (t)$ are necessarily incomplete for $t < T$. We address this by censoring the training input
the same way active learning censors $x (t)$: fix a cutoff $\delta_c \in [0,\infty)$ and define

$$C_{i,j} = \mathbb{1}[\Delta_{i,j} \le \delta_c], \qquad \min_{E,D} \ell\big (X, \sigma (C B_{E,D})\big)$$

$C$ retains only opens fast enough to plausibly be observed during an operational window, training the SAE to recover
the true eventual pattern $X$ from this censored view. As $\delta_c \to \infty$, $C \to X$ and the model reduces to the
original formulation; as $\delta_c \to 0$, all training signal is discarded. $\delta_c$ should be anchored to the batch
interval $T/b$ and the deadline $T$. This construction is the $h_\delta \to 0$ limit of the decayed label $Y$ above: $Y$
softly reweights the *target*, while $C$ hard-masks the *input*.

### 3.7 Nonlinear (Deep) Autoencoder

The baseline reconstruction $\sigma (xB_{E,D})$ uses a single fixed $m \times m$ linear map, limiting the model to
linear recipient–recipient relationships. Both $p_j$ and $f_j (t)$ (Section 3.3) depend only on the ability to
query $f_{SAE}$, not on any explicit property of $B_{E,D}$:

$$p_j = \frac{1}{m-1}\sum_{i \ne j} f_{SAE} (e_j)\,e_i, \qquad f_j (t) = f_{SAE} (x (t))\,e_j$$

so $f_{SAE}$ can be replaced by a deeper, nonlinear network without changing either definition. We propose a two-layer
encoder/decoder with a ReLU nonlinearity:

$$\text{Encoder: Linear} (m,2d) \to \text{ReLU} \to \text{Linear} (2d,d)$$
$$\text{Decoder: Linear} (d,2d) \to \text{ReLU} \to \text{Linear} (2d,m)$$

$$f_{DAE} (x) = (\sigma \circ \text{Decoder} \circ \text{Encoder})(x), \qquad \min_\theta \ell\big (X, f_{DAE} (X;\theta)\big)$$

with $d$ retaining its role as bottleneck size, and optional dropout/layer normalization at the bottleneck to control
overfitting. A deep network has no single weight matrix to zero out a self-prediction shortcut, so we substitute a
denoising objective: randomly mask a subset of input entries to $0$ at each training step and require reconstruction of
the full, unmasked $X$, forcing every prediction to depend on other recipients rather than the recipient's own entry.
Under this substitution, $p_j$ and $f_j (t)$ retain their forward-pass definitions with $f_{SAE}$ replaced by $f_{DAE}$.

## 4. Open Directions for Future Work

The following ideas are motivated by the extensions above but have **not** been implemented or validated. They are
recorded for future exploration only.

- **Combined alpha scheduling.** Make $\alpha (t)$ a function of both $\mu$ (Section 3.1.1) and $\tilde o$ (Section
  3.1.2). Risk: excessive hyperparameter count.
- **$z (t)$ interpolation for TTO.** Replace the binary query state with $z_j (t) = 2^{-\delta_j/h_\delta}$ for
  recipients who opened template $n+1$ by time $t$. Mathematically valid (linear $x^\top B_{E,D}$, $z (t) \in [0,1]^m$),
  but unclear whether early-opener TTO adds signal beyond binary $x (t)$.
- **Hyperbolic decay for $Y$.** $Y_{i,j} = \kappa_\delta/ (\Delta_{i,j}+\kappa_\delta)$, mirroring Section 3.1.2's
  schedule, would give a heavier tail than the exponential decay in Section 3.6 — worth an empirical comparison.
- **TTO-to-TTO model ($Y \to Y$).** Training on $\min_{E,D}\ell (Y,\sigma (YB_{E,D}))$ would expose TTO on the input
  side too, but an instantaneous open is then a boundary point rather than a typical training example, so $p_j$ can no
  longer be queried at $e_j$ and would need re-deriving (e.g. at a population-level mean TTO) — with unclear risk of
  double-counting against $\phi_j$.
- **$G$ re-calibration under TTO.** Any TTO-blended $\Sigma^Y$ shifts the distribution of $s_j (t)$ relative to the
  original $X$-based $\Sigma$; the confidence modifier $G$ (tuned on the baseline) likely needs re-tuning for the
  Binary-to-TTO model and any of the extensions above. Not yet checked empirically.

## 5. Experimental Evaluation

### 5.1 Setup

Each variant is evaluated with a grid search over its hyperparameters, selecting the best model on the validation set.
We report:

$$\text{Recall-AUC} = \int_0^1 \text{Recall} (\tau)\, d\tau$$

together with Recall@5%, @15%, @25%, and @35% of recipients targeted, on the test set (mean ± std over 10 simulations).

### 5.2 Baseline Reproduction

Using the original paper's evaluation quantiles (25/50/75%), our reproduction closely matches the published results,
validating the experimental pipeline:

| Recall@25%    | Recall@50%    | Recall@75%    |
|---------------|---------------|---------------|
| 0.923 ± 0.014 | 0.975 ± 0.008 | 0.989 ± 0.004 |

*(Original paper: 0.923±0.0005, 0.975±0.0002, 0.989±0.0004 — means match; our std is larger due to fewer simulation
runs.)*

### 5.3 Results Summary

All subsequent experiments use our evaluation protocol (Recall@5/15/25/35%, Recall-AUC):

| Model                                | Selected hyperparameters                                                          | Recall@5%   | Recall@15%  | Recall@25%  | Recall@35%  | AUC         |
|--------------------------------------|-----------------------------------------------------------------------------------|-------------|-------------|-------------|-------------|-------------|
| Baseline                             | —                                                                                 | 0.385±0.076 | 0.821±0.028 | 0.923±0.014 | 0.958±0.011 | 0.894±0.012 |
| Linear alpha scheduling (§3.1.1)     | $l=0.1$, $r=0.05$                                                                 | 0.383±0.073 | 0.823±0.023 | 0.927±0.012 | 0.960±0.010 | 0.894±0.011 |
| Geometric alpha scheduling (§3.1.1)  | $l=0.3$, $r=0.05$                                                                 | 0.377±0.050 | 0.823±0.017 | 0.927±0.012 | 0.959±0.010 | 0.894±0.009 |
| Confidence alpha scheduling (§3.1.2) | $\kappa=0.003$                                                                    | 0.341±0.031 | 0.818±0.022 | 0.926±0.014 | 0.960±0.010 | 0.891±0.009 |
| Template weights (§3.2)              | $h=0.3$                                                                           | 0.447±0.041 | 0.837±0.018 | 0.928±0.014 | 0.960±0.011 | 0.903±0.009 |
| Forward-pass $f$ (§3.3)              | —                                                                                 | 0.375±0.049 | 0.841±0.029 | 0.937±0.018 | 0.966±0.010 | 0.901±0.011 |
| Variance-based $p$ (§3.4)            | —                                                                                 | 0.479±0.034 | 0.837±0.017 | 0.926±0.013 | 0.960±0.010 | 0.905±0.009 |
| Alternative $s$ decomposition (§3.5) | $\alpha(t)$: confidence, $\kappa=0.005$; $\beta(t)$: geometric, $l=0.1$, $r=0.05$ | 0.425±0.026 | 0.821±0.017 | 0.922±0.013 | 0.959±0.010 | 0.900±0.008 |
| TTO cutoff thresholding (§3.6)       | $\delta_c=720$                                                                    | 0.390±0.038 | 0.824±0.016 | 0.924±0.012 | 0.960±0.010 | 0.895±0.008 |
| Deep autoencoder (§3.7)              | $d=16$                                                                            | 0.400±0.090 | 0.815±0.085 | 0.936±0.019 | 0.963±0.011 | 0.898±0.019 |

*The Binary-to-TTO training variant (§3.6) was proposed but not yet run through this evaluation protocol.*

### 5.4 Discussion

- **Variance-based $p$** and **template weights** give the largest AUC gains (0.905, 0.903 vs. 0.894 baseline), with
  variance-based $p$ also giving the best early-recall (@5%) improvement.
- **Forward-pass $f$** gives the strongest gain at moderate-to-high recall targets (@15–35%), consistent with its goal
  of capturing joint opener influence.
- **Alpha-scheduling variants** (linear, geometric) roughly match the baseline; **confidence-based scheduling** slightly
  underperforms at low recall, consistent with the noted risk that $\tilde o$ moves too slowly under a low baseline open
  rate.
- **TTO cutoff thresholding** and the **deep autoencoder** are roughly on par with baseline AUC, with the deep
  autoencoder trading higher variance for improved recall at higher targeting fractions.

Result figures (smoothed test-set Recall curves) for each variant:

| Model                         | Figure                                          |
|-------------------------------|-------------------------------------------------|
| Baseline                      | `images/baseline_recalls_0_1.png`               |
| Linear alpha scheduling       | `images/linear_alpha_recalls_0_1.png`           |
| Geometric alpha scheduling    | `images/geometric_alpha_recalls_0_1.png`        |
| Confidence alpha scheduling   | `images/confidence_based_alpha_recalls_0_1.png` |
| Template weights              | `images/exp_template_weight_recalls_0_1.png`    |
| Forward-pass $f$              | `images/forward_pass_f_recalls_0_1.png`         |
| Variance-based $p$            | `images/variance_based_p_recalls_0_1.png`       |
| Alternative $s$ decomposition | `images/alternative_s_recalls_0_1.png`          |
| TTO cutoff thresholding       | `images/tto_cutoff_recalls_0_1.png`             |
| Deep autoencoder              | `images/deep_autoencoder_recalls_0_1.png`       |

## 6. Conclusion

Building on the SAE–Thompson Sampling bandit of Žid et al. (2025), we proposed seven concrete extensions — dynamic alpha
scheduling, recency-weighted engagement, a forward-pass confidence score, a variance-based influence score, a decoupled
multiplicative score, time-to-open-aware training, and a nonlinear autoencoder — each evaluated against the reproduced
baseline. Variance-based influence scoring and recency-weighted engagement give the most consistent gains in this
evaluation. Several further ideas (combined scheduling, TTO-aware query states, TTO-to-TTO training, confidence
recalibration) remain open for future validation.