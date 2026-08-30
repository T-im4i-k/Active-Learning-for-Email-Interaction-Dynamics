# Refinements to the SAE–Thompson Sampling Framework for Active Email Outreach

*Extends:* Žid, Č., Alves, R., and Kordík, P. 2025. *Active Recommendation for Email Outreach Dynamics.* In
*Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)*, Seoul,
Republic of Korea. https://doi.org/10.1145/3746252.3760832

---

## 1. Introduction

Žid et al. (2025) frame cold-start recipient selection for a new email campaign as an iterative multi-armed bandit:
each arm is a recipient, and Thompson Sampling (TS) parameters are derived from a shallow autoencoder (SAE) trained
on historical recipient–template opens, avoiding retraining during the active-learning phase. Several components of
that model — a fixed exploration/exploitation coefficient, an unweighted historic open rate, a linear aggregation for
the current-state confidence term, training on binary labels only, and a shallow architecture — were fixed by the
original authors for simplicity. We revisit each of these choices and propose concrete, mathematically grounded
refinements: dynamic scheduling of the blend coefficient, a recency-weighted historic rate, a nonlinear forward-pass
confidence term, a variance-based influence score, a factored score function, two ways of exposing the autoencoder to
previously unused time-to-open (TTO) signal, and a deep, nonlinear autoencoder. We report experimental results for
each proposal against a reproduced baseline and outline directions for further work.

## 2. Recap of the Original Model

**Setup.** We observe $X \in \{0,1\}^{n \times m}$, where $n$ is the number of templates, $m$ the number of
recipients, and $X_{i,j}=1$ iff recipient $j$ opened template $i$. For a new template $n{+}1$, sent over $b$ batches
within a deadline $T$, the model selects at each batch a small subset of recipients so as to maximize eventual
openers reached early.

**Shallow autoencoder.** A SAE with encoder/decoder $E, D \in \mathbb{R}^{m\times d}$ is trained as

$$\min_{E,D} \ell\big(X,\ \sigma(XB_{E,D})\big), \qquad B_{E,D} = ED^\top - \mathrm{diag}\big([E\odot D]\mathbf{1}_m\big),$$

with $\ell$ the element-wise binary cross-entropy and the diagonal constraint ruling out trivial self-prediction. We
write $f_{SAE}(x) = \sigma(x^\top B_{E,D})$ for the trained forward pass and $\Sigma = \sigma(B_{E,D})$; entry
$\Sigma_{i,j}$ is the probability of recipient $j$ opening a template given that only recipient $i$ has opened it —
a measure of $i$'s influence on $j$.

**Arm score.** The bandit score for recipient $j$ at time $t$ is

$$s_j(t) = \alpha\,\phi_j p_j + (1-\alpha) f_j(t), \tag{1}$$

$$\phi_j = \frac1n\sum_{i=1}^n X_{i,j}, \qquad p_j = \frac{1}{m-1}\sum_{i\ne j}\Sigma_{j,i}, \qquad f_j(t) = \bar x(t)^\top \Sigma_{:j}, \tag{2}$$

where $\phi_j$ is recipient $j$'s historic open rate, $p_j$ is a general influence score, $f_j(t)$ is a confidence
score specific to template $n{+}1$ given the observed (normalized) state $\bar x(t)$, and $\alpha \in [0,1]$ trades
off the two. TS parameters are then $\alpha_j(t) = Gs_j(t)$, $\beta_j(t) = G(1-s_j(t))$ for a confidence modifier $G$.

Two orthogonal readings of Eq. (1) motivate our refinements: $\phi_j p_j$ vs. $f_j(t)$ can be read as
*exploration vs. exploitation*, or equivalently as *historic vs. current* data.

## 3. Proposed Refinements

### 3.1 Dynamic $\alpha$-Scheduling

A fixed $\alpha$ ignores that the value of exploration should shrink as the campaign progresses. We schedule
$\alpha(t)$ against two natural progress variables: $\mu = N(t)/N$, the fraction of the sending budget already used,
and $\tilde o = o(t)/m$, the fraction of recipients who have already opened, with $o(t) = x(t)^\top\mathbf 1$.

*Send-volume-based (function of $\mu$).* A **linear** schedule from $l$ to $r$,

$$\alpha(\mu) = l(1-\mu) + r\mu, \tag{3}$$

and a **geometric** (log-linear) schedule,

$$\alpha(\mu) = l^{1-\mu}\, r^{\mu}. \tag{4}$$

*Open-rate-based (function of $\tilde o$).* Since $f_j(t)$ is itself a point estimate that becomes more reliable as
more opens are observed, we also consider a **confidence** schedule, hyperbolic in $\tilde o$:

$$\alpha(\tilde o) = \frac{\kappa}{\tilde o + \kappa}, \tag{5}$$

with dampening factor $\kappa$ (so $\alpha(\kappa)=\tfrac12$). All three reduce to the constant baseline at their
degenerate limits ($l=r$, or $\kappa\to\infty$). A schedule jointly conditioned on both $\mu$ and $\tilde o$ is
conceptually natural but is left to future work (Section 5) to avoid an unconstrained hyperparameter count.

### 3.2 Recency-Weighted Historic Rate $\phi_j$

The original $\phi_j$ weights every past template equally, ignoring drift in recipient behavior over time. Assuming
templates $X_{1:},\dots,X_{n:}$ were sent at uniform intervals, we assign template $i$ an exponential weight

$$\omega_i = 2^{-d_i/h}, \qquad d_i = \frac{n-i}{n}, \tag{6}$$

with half-life $h>0$, and redefine

$$\phi_j = \frac{\omega^\top X_{:j}}{\sum_i \omega_i}. \tag{7}$$

Normalizing by $\sum_i\omega_i$ keeps $\phi_j\in[0,1]$. As $h\to\infty$ every $\omega_i\to1$, recovering the original
unweighted average as a special case.

### 3.3 Forward-Pass Definition of $f_j(t)$

$f_j(t) = \bar x(t)^\top\Sigma_{:j}$ is a linear combination of individual influences $\Sigma_{i,j}$ and therefore
captures only the *average* effect of openers, never their joint interaction. We instead pass the full current state
through the trained network:

$$f_j(t) = \sigma\big(x(t)^\top B_{E,D}\big)e_j = f_{SAE}(x(t))\,e_j. \tag{8}$$

Because $\sigma$ is nonlinear, $\sigma(x(t)^\top B_{E,D}) \ne x(t)^\top\sigma(B_{E,D})$ in general (equality holds
only for one-hot $x(t)$), so Eq. (8) is a genuinely different, jointly-conditioned probability estimate rather than a
relabeling of Eq. (2).

### 3.4 Variance-Based Definition of $p_j$

$p_j$ is meant to reward recipients whose opening is *informative* about others. The original definition,
$p_j = \frac{1}{m-1}\sum_{i\ne j}\Sigma_{j,i}$, rewards recipients that raise the *average level* of other
recipients' $f$-scores — which is not the same as reducing uncertainty about them. We instead reward maximal
*spread* induced across other recipients' scores:

$$p_j = \frac{4}{m-1}\sum_{i\ne j}\big(\Sigma_{j,i} - \bar\Sigma_{j:}\big)^2, \qquad \bar\Sigma_{j:} = \frac{1}{m-1}\sum_{i\ne j}\Sigma_{j,i}. \tag{9}$$

This is the variance of $\Sigma_{j,:}$ (excluding $j$), scaled by $4$ so that the maximum possible variance of a
Bernoulli variable (0.25) maps to 1, keeping $p_j \in [0,1]$.

### 3.5 Factored Score $s_j(t) = \pi_j(t)\,u_j(t)$

The linear combination in Eq. (1) conflates the two orthogonal axes noted in Section 2 — it is not clear, for
instance, why exploration should draw solely on historic data while exploitation draws solely on current data. We
instead factor the score into a *probability-of-opening* term and a *utility-of-opening* term, each independently
schedulable:

$$\pi_j(t) = \alpha(t)\phi_j + (1-\alpha(t))f_j(t), \qquad u_j(t) = \beta(t)p_j + (1-\beta(t))\cdot 1, \tag{10}$$

$$s_j(t) = \pi_j(t)\,u_j(t) = \Big(\alpha(t)\phi_j + [1-\alpha(t)]f_j(t)\Big)\Big(\beta(t)p_j + [1-\beta(t)]\Big). \tag{11}$$

Here $\pi_j(t)$ blends historic and current opening-probability estimates, while $u_j(t)$ blends the direct value of
an open (the constant $1$) with its indirect information value $p_j$. This factorization reuses every primitive
introduced above unchanged, so refinements from Sections 3.1–3.4 can be dropped into $\pi_j(t)$ and $u_j(t)$ directly
— we use a confidence schedule (Eq. 5) for $\alpha(t)$ and a send-volume schedule (Eq. 3/4) for $\beta(t)$.

### 3.6 Incorporating Time-to-Open (TTO)

The SAE is trained purely on binary opens, discarding *how fast* a recipient opened — signal that matters given the
short operational window ($T/b$, $T$). Let $\Delta_{i,j}\in\mathbb{R}_{\ge0}\cup\{+\infty\}$ be the observed
time-to-open (with $\Delta_{i,j}=+\infty$ for non-openers, so $X_{i,j}=\mathbb{1}[\Delta_{i,j}<\infty]$), and define a
decayed open label

$$Y_{i,j} = 2^{-\Delta_{i,j}/h_\delta}, \qquad 2^{-\infty} := 0, \tag{12}$$

with TTO half-life $h_\delta$; as $h_\delta\to\infty$, $Y_{i,j}\to X_{i,j}$ for every opener.

**Binary-to-TTO autoencoder.** We keep the SAE's binary input but replace the reconstruction target:

$$\min_{E,D}\ \ell\big(Y,\ \sigma(XB_{E,D})\big). \tag{13}$$

Since the input space is unchanged, the forward-pass definitions of $p_j$ and $f_j(t)$ (Eqs. 8–9) carry over
mechanically; only their interpretation shifts — $\Sigma_{j,i}$ now blends *whether* $i$ opens with *how fast*.

**TTO cutoff.** Training rows of $X$ reflect the *eventual* open pattern, while active-learning queries $x(t)$ are
necessarily censored at $t \lt T$. We instead censor the training input itself, fixing a cutoff $\delta_c$ and defining
$C_{i,j} = \mathbb{1}[\Delta_{i,j}\le\delta_c]$, then training

$$\min_{E,D}\ \ell\big(X,\ \sigma(CB_{E,D})\big). \tag{14}$$

$C$ retains only opens fast enough to plausibly be observed within an active operational window and treats slower
opens as non-opens during training — matching the censoring recipients are actually subject to during active
learning. As $\delta_c\to\infty$, $C\to X$, recovering the original model.

### 3.7 Deep Autoencoder

$B_{E,D}$ in Eq. (2)/(2) is a single fixed $m\times m$ matrix, so the SAE can only capture linear
recipient–recipient relationships. Since $p_j$ and $f_j(t)$ (Section 3.3–3.4) already depend only on the ability to
query $f_{SAE}$, not on any explicit property of $B_{E,D}$, we can swap in a deeper, nonlinear network without
changing either definition. We use a two-layer encoder/decoder with a ReLU nonlinearity,

$$\text{Encoder: Linear}(m,2d)\to\text{ReLU}\to\text{Linear}(2d,d), \qquad \text{Decoder: Linear}(d,2d)\to\text{ReLU}\to\text{Linear}(2d,m), \tag{15}$$

giving $f_{DAE}(x) = (\sigma\circ\text{Decoder}\circ\text{Encoder})(x)$, optionally with dropout and bottleneck-layer
normalization. A deep network has no single weight matrix whose diagonal can be constrained to zero; we substitute a
denoising objective, randomly masking input entries and requiring reconstruction of the full unmasked $X$, forcing
every prediction to depend on other recipients rather than a self-shortcut. Training becomes
$\min_\theta \ell(X, f_{DAE}(X;\theta))$, and $p_j$, $f_j(t)$ retain their Section 3.3–3.4 forms with $f_{SAE}$
replaced by $f_{DAE}$.

## 4. Experiments

### 4.1 Methodology

For each variant we grid-search its hyperparameters and select the configuration maximizing validation-set
Recall-AUC, $\mathrm{AUC} = \int_0^1 \mathrm{Recall}(\tau)\,d\tau$, where $\mathrm{Recall}(\tau)$ is recall at
sending fraction $\tau$. We report test-set Recall@5/15/25/35% and Recall-AUC, mean $\pm$ std over repeated
simulations. Before testing modifications we reproduced the original model on our pipeline; at the operating points
reported by Žid et al. (25/50/75%) our reproduction matches the published point estimates exactly
($0.923,\ 0.975,\ 0.989$), with larger variance attributable to fewer simulation runs.

### 4.2 Results

| Variant | Selected hyperparameters | Recall@5% | Recall@15% | Recall@25% | Recall@35% | AUC |
|---|---|---|---|---|---|---|
| Baseline | — | 0.385 ± 0.076 | 0.821 ± 0.028 | 0.923 ± 0.014 | 0.958 ± 0.011 | 0.894 ± 0.012 |
| Linear $\alpha$-schedule | $l=0.1,\ r=0.05$ | 0.383 ± 0.073 | 0.823 ± 0.023 | 0.927 ± 0.012 | 0.960 ± 0.010 | 0.894 ± 0.011 |
| Geometric $\alpha$-schedule | $l=0.3,\ r=0.05$ | 0.377 ± 0.050 | 0.823 ± 0.017 | 0.927 ± 0.012 | 0.959 ± 0.010 | 0.894 ± 0.009 |
| Confidence $\alpha$-schedule | $\kappa=0.003$ | 0.341 ± 0.031 | 0.818 ± 0.022 | 0.926 ± 0.014 | 0.960 ± 0.010 | 0.891 ± 0.009 |
| Recency-weighted $\phi_j$ | $h=0.3$ | 0.447 ± 0.041 | 0.837 ± 0.018 | 0.928 ± 0.014 | 0.960 ± 0.011 | 0.903 ± 0.009 |
| Forward-pass $f_j(t)$ | — | 0.375 ± 0.049 | 0.841 ± 0.029 | 0.937 ± 0.018 | 0.966 ± 0.010 | 0.901 ± 0.011 |
| Factored $s_j(t)$ | $\alpha(t)$: conf., $\kappa=0.005$; $\beta(t)$: geom., $l=0.1,\ r=0.05$ | 0.425 ± 0.026 | 0.821 ± 0.017 | 0.922 ± 0.013 | 0.959 ± 0.010 | 0.900 ± 0.008 |
| Variance-based $p_j$ | — | 0.479 ± 0.034 | 0.837 ± 0.017 | 0.926 ± 0.013 | 0.960 ± 0.010 | 0.905 ± 0.009 |
| Binary-to-TTO autoencoder | $h_\delta=11520$ | 0.331 ± 0.051 | 0.719 ± 0.040 | 0.896 ± 0.015 | 0.955 ± 0.008 | 0.877 ± 0.011 |
| TTO cutoff | $\delta_c=720$ | 0.390 ± 0.038 | 0.824 ± 0.016 | 0.924 ± 0.012 | 0.960 ± 0.010 | 0.895 ± 0.008 |
| Deep autoencoder | $d=16$ | 0.404 ± 0.086 | 0.829 ± 0.052 | 0.931 ± 0.019 | 0.963 ± 0.011 | 0.900 ± 0.017 |

### 4.3 Discussion

Differences from baseline are largest at low sending fractions and shrink as $\tau$ grows, which is expected: baseline
recall already exceeds 0.92 by $\tau=25\%$, leaving little headroom. The **variance-based $p_j$** (0.479 at 5%) and
**recency-weighted $\phi_j$** (0.447) give the largest low-fraction gains, followed by the **factored $s_j(t)$**
(0.425); all three also improve AUC (0.905, 0.903, 0.900 vs. 0.894 baseline). The **forward-pass $f_j(t)$** shows no
gain at 5% but a consistent uplift from 15% onward (0.841/0.937/0.966), suggesting nonlinear aggregation of openers
only pays off once several recipients have opened. The three **$\alpha$-scheduling** variants are essentially neutral
at low $\tau$, and the confidence schedule is *worse* than baseline (0.341 at 5%) — its effective range is likely too
narrow given the dataset's low ($9.1\%$) empirical open rate, and $\kappa$ warrants re-tuning. The
**Binary-to-TTO autoencoder** underperforms across every quantile, most severely at 15% (0.719 vs. 0.821): discounting
slow opens in the reconstruction target appears to degrade collaborative-filtering signal quality more than the
speed information adds, at least at the selected $h_\delta$; this may also reflect the confidence modifier $G$
(tuned for the original $\Sigma$) being miscalibrated for a TTO-blended $\Sigma$ of different scale. **TTO cutoff**
censoring gives a small, consistent uplift across all quantiles at negligible interpretive cost — training-time
censoring that matches the active-learning observability window helps modestly without changing what $\Sigma$
represents. The **deep autoencoder** improves Recall@5% (0.404) and the higher quantiles modestly, but with
markedly higher variance throughout (e.g. ±0.086 at 5%, ±0.052 at 15% — 2–3$\times$ most other variants), indicating
the added capacity is not yet well-regularized at this dataset size. All results test one modification at a time;
combinations are untested.

## 5. Future Work

- **Combined $\alpha$-scheduling.** A schedule jointly conditioned on send-progress $\mu$ and open-progress
  $\tilde o$ rather than either alone (Section 3.1); the main obstacle is the added hyperparameter count.
- **Plug-and-play composition under the factored score.** Systematically combine the primitive-level refinements —
  recency-weighted $\phi_j$, forward-pass $f_j(t)$, variance-based $p_j$, TTO- or deep-autoencoder-derived $\Sigma$ —
  inside $\pi_j(t)$ and $u_j(t)$ (Eq. 10) rather than evaluating each in isolation, and search jointly rather than
  ablating one factor at a time, since the gains in Section 4.3 are not guaranteed to be additive.
- **$z(t)$ interpolation for TTO-informed inference.** Rather than querying $f_j(t)$ on the binary $x(t)$, define a
  partially TTO-informed state $z_i(t) = 2^{-\delta_i/h_\delta}$ for recipients who have already opened template
  $n{+}1$ (and $0$ otherwise), and evaluate $f_j(t) = f_{SAE}(z(t))e_j$. Since $x^\top B_{E,D}$ is linear and
  $z(t)\in[0,1]^m$, this is a valid extension of a model trained on binary rows; whether early-openers' TTO adds
  signal beyond the binary state is untested.
- **Hyperbolic decay for $Y$.** $Y_{i,j} = \kappa_\delta/(\Delta_{i,j}+\kappa_\delta)$ gives a heavier tail than the
  exponential label of Eq. (12) and merits a direct comparison.
- **TTO-to-TTO autoencoder.** Replacing the SAE input with $Y$ as well, $\min_{E,D}\ell(Y,\sigma(YB_{E,D}))$, exposes
  TTO on both sides but changes the role of $e_j$ — an instantaneous open becomes a boundary point rather than a
  typical input — so $p_j$ can no longer be queried at $e_j$ and would need re-deriving, e.g. via a recipient- or
  population-level mean-TTO query, with unclear risk of double-counting against $\phi_j$.
- **$G$ re-calibration.** Any TTO-blended $\Sigma$ shifts the scale of $s_j(t)$ relative to the original
  $X$-based $\Sigma$; the confidence modifier $G$ likely needs re-tuning under any TTO variant, and may explain part
  of the Binary-to-TTO underperformance observed in Section 4.3.
- **Asymmetric deep encoder/decoder.** The two-layer network of Section 3.7 uses matched encoder/decoder depth and
  width; decoupling them (e.g. a wider encoder with a narrower, more constrained decoder, or vice versa) may better
  balance representational capacity against the denoising-based regularization, and warrants exploration alongside a
  wider sweep over masking rate, dropout, and bottleneck width $d$ to address the variance observed in Section 4.3.
- **Time-informed $p_j$.** The influence score $p_j$ (Section 3.4) currently reflects only whether opening is
  informative; extending it to reward recipients whose *speed* of opening is informative about others' opening
  speed — not merely their opening probability — is a natural companion to the TTO constructions of Section 3.6 but
  has not been formulated or tested here.

## References

Žid, Č., Alves, R., and Kordík, P. 2025. Active Recommendation for Email Outreach Dynamics. In *Proceedings of the
34th ACM International Conference on Information and Knowledge Management (CIKM '25)*, November 10–14, 2025, Seoul,
Republic of Korea. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3746252.3760832