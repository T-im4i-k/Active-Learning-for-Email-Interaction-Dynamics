# Refinements to the SAE–Bandit Framework for Active Email Recommendation

**Working draft — extends** Žid, C., Alves, R., and Kordík, P. (2025). *Active Recommendation for Email Outreach
Dynamics.* CIKM '25. https://doi.org/10.1145/3746252.3760832

*Tymofii Ivanov, Faculty of Information Technology, CTU Prague*

---

## Abstract

Žid et al. (2025) frame cold-start email recipient selection as an iterative multi-armed bandit, where each arm is a
recipient and Thompson Sampling parameters are derived from a shallow autoencoder (SAE) trained on the
recipient–template open matrix. In this draft we revisit several components of that framework — the fixed
exploration/exploitation coefficient $\alpha$, the historic open-rate term $\phi_j$, the definitions of the influence
term $p_j$ and the confidence term $f_j (t)$, the linear form of the score $s_j (t)$, the exclusive reliance on binary
open/no-open labels, and the shallow architecture of the autoencoder itself — and propose concrete, mathematically
grounded refinements to each. These include dynamic $\alpha$-scheduling (send-volume-based and open-rate-based), a
recency-weighted $\phi_j$, a forward-pass definition of $f_j (t)$, a variance-based definition of $p_j$ motivated by
information gain rather than magnitude, a factored score $s_j (t) = \pi_j (t)u_j (t)$ that separates the "probability of
opening" axis from the "value of opening" axis, two complementary ways of exposing the SAE to the previously unused
time-to-open (TTO) signal — a soft, decay-weighted reconstruction target and a hard cutoff that censors training inputs
to match what is actually observable mid-campaign — and a two-layer deep autoencoder that reuses the forward-pass
primitives introduced along the way. We report preliminary grid-search results for each implemented proposal against a
reproduced baseline. Most variants leave the model's high-percentile recall (Recall@25% and above) close to baseline,
but several — most notably the variance-based $p_j$ and the recency-weighted $\phi_j$ — visibly improve recall at very
small sending fractions (Recall@5%), which is arguably the regime of greatest practical interest. TTO cutoff
thresholding and the deep autoencoder yield smaller, more mixed gains, the latter with markedly higher run-to-run
variance. This document is an early-stage technical report intended as a shared working basis rather than a finished
paper: the soft TTO-weighted training variant, and its further extensions, are presented with derivations but not yet
validated experimentally.

---

## 1. Introduction

Žid et al. propose an active-learning framework for email outreach in which a shallow autoencoder, trained on historical
open/no-open data, supplies the arm scores of a Thompson Sampling bandit that selects which recipients receive each new
campaign batch. The design is attractive because it requires no retraining during the active-learning phase and remains
interpretable, but several of its components were fixed by the original authors for simplicity — a constant blending
coefficient $\alpha$, an unweighted historical open-rate, a linear (rather than autoencoder-native) aggregation for the
current-state confidence term, training on binary opens only, and a shallow autoencoder architecture. Each of these
choices is a reasonable starting point but also a natural place to look for improvement.

This document collects a set of refinements explored on top of the original formulation, together with the reasoning
behind each, and reports preliminary experimental results. We keep the notation of the original paper wherever possible
so that each proposal can be understood as a drop-in replacement for one primitive of the original model, rather than a
redesign of the framework as a whole.

### 1.1 Contributions

1. **Dynamic $\alpha$-scheduling.** We reinterpret the fixed coefficient $\alpha$ as an explore/exploit dial and propose
   three schedules that decay it over the course of active learning: a send-volume-based linear schedule, a
   send-volume-based geometric (log-linear) schedule, and an open-rate-based ("confidence") hyperbolic schedule.
2. **Recency-weighted historical open rate.** We generalize $\phi_j$ from an unweighted average over past templates to
   an exponentially recency-weighted average, of which the original definition is a limiting case.
3. **Forward-pass definition of $f_j (t)$.** We replace the linear-combination definition of the current-state
   confidence term with one that passes the full current open-state through the trained autoencoder, capturing joint
   (nonlinear) interactions between simultaneous openers rather than only their individually averaged effects.
4. **Variance-based definition of $p_j$.** We replace the mean-based influence score with a variance-based one, arguing
   that the quantity of interest for information gain is uncertainty *reduction*, not the magnitude of the induced
   shift.
5. **Factored score function.** We decompose $s_j (t)$ into a probability-of-opening term $\pi_j (t)$ and a utility
   term $u_j (t)$, each independently schedulable, so that the "historic vs. current data" axis and the "exploration vs.
   exploitation" axis — conflated in the original linear score — can be controlled separately.
6. **Time-to-open signal.** We fold the previously unused time-to-open signal into SAE training via two complementary
   constructions: a soft, exponentially decayed reconstruction target that discounts slow opens without altering the
   SAE's input space, and a hard cutoff that instead censors the training input itself to match what is actually
   observable mid-campaign.
7. **Deep autoencoder.** We generalize the SAE to a two-layer nonlinear encoder/decoder. Because $p_j$ and $f_j (t)$ can
   already be expressed purely as forward passes through the trained model, both definitions carry over unchanged under
   this substitution, with a denoising-style training procedure standing in for the original diagonal constraint.

Sections 2–9 develop each proposal in turn; Section 10 describes the experimental protocol used to evaluate the ones we
have tested so far; Section 11 reports results; Section 12 discusses open questions; Section 13 concludes.

---

## 2. Background

### 2.1 Problem Setup and Notation

We observe a matrix $X \in \{0,1\}^{n \times m}$, where $n$ is the number of email templates, $m$ is the number of
recipients, and $X_{i,j} = 1$ if recipient $j$ opened template $i$. Given a new template $n+1$ sent in $b$ batches over
a window $[0, T]$, the goal is to select, at each batch, which small subset of the remaining recipients should receive
the message, so as to maximize how many eventual openers are reached early — before the sending budget is exhausted.

| Symbol                            | Meaning                                                             |
|-----------------------------------|---------------------------------------------------------------------|
| $n$, $m$                          | number of templates, number of recipients                           |
| $X \in \{0,1\}^{n\times m}$       | historical open matrix                                              |
| $E, D \in \mathbb{R}^{m\times d}$ | SAE encoder / decoder matrices, bottleneck size $d$                 |
| $\Sigma = \sigma(B_{E,D})$        | recipient–recipient influence matrix                                |
| $\phi_j$                          | historic open rate of recipient $j$                                 |
| $p_j$                             | influence score of recipient $j$                                    |
| $f_j(t)$                          | current-state confidence score of recipient $j$ at time $t$         |
| $s_j(t)$                          | arm score for recipient $j$ at time $t$                             |
| $\alpha$                          | historic/current (alternatively, explore/exploit) blend coefficient |

### 2.2 Shallow Autoencoder for Collaborative Filtering

A shallow autoencoder is trained on $X$ as

$$\min_{E,D \in \mathbb{R}^{m\times d}} \ell\big (X,\ \sigma (XB_{E,D})\big), \qquad B_{E,D} = ED^\top - \mathrm{diag}\big ([E\odot D]\mathbf{1}\big), \tag{1}$$

with $\ell$ the element-wise binary cross-entropy loss and the diagonal constraint on $B_{E,D}$ (standard in this family
of models) preventing the trivial identity solution. We write $f_{SAE} (x) = \sigma (x^\top B_{E,D})$ for the map from a
binary open-vector to a vector of predicted opening probabilities, and $\Sigma = \sigma (B_{E,D})$ for the corresponding
matrix, so that

$$\Sigma_{i,j} = e_i^\top \Sigma e_j = \big[f_{SAE} (e_i)\big]_j. \tag{2}$$

**Remark (indexing convention).** $B_{E,D}$, and hence $\Sigma$, is not symmetric in general, since $E \neq D$.
Throughout this document the row index of $\Sigma$ is the recipient who has *already opened* (the conditioning
recipient) and the column index is the recipient whose opening probability is being *predicted*. This is the convention
under which $p_j$ (Section 2.3, which reads along row $j$) is naturally an *influence* score and $f_j (t)$ (which reads
along column $j$) is naturally a *received-influence* score; the original paper's prose description of $\Sigma_{i,j}$
appears to state the two indices in the opposite order, but its own formulas for $p_j$ and $f_j (t)$ are only consistent
with the convention adopted here.

### 2.3 Original Arm-Score Function

The original score for recipient $j$ at time $t$ is

$$s_j (t) = \alpha\,\phi_j p_j + (1-\alpha)\,f_j (t), \tag{3}$$

where

$$\phi_j = \frac{1}{n}\sum_{i=1}^n X_{i,j} \quad\text{ (historic open rate)}, \qquad p_j = \frac{1}{m-1}\sum_{i \ne j} \Sigma_{j,i} \quad\text{ (influence)}, \qquad f_j (t) = \bar x (t)^\top \Sigma_{:,j} \quad\text{ (current-state confidence)}, \tag{4}$$

and $\bar x (t) = x (t)/n_t$ is the normalized vector of observed opens for template $n+1$ up to time $t$ ($n_t$ being
the number of recipients who have opened so far, or 1 if none have).

The score $s_j (t)$ is a measure of the *usefulness* of sending an email to recipient $j$, which feeds the parameters of
a Beta distribution ($\alpha_j (t) = Gs_j (t)$, $\beta_j (t) = G (1-s_j (t))$, with $G$ a confidence modifier) used by
Thompson Sampling to select the next batch.

---

## 3. Dynamic $\alpha$-Scheduling

### 3.1 Motivation

The term $\phi_j p_j$ in Eq. (3) is computed purely from historic data and rewards recipients whose opening would be
*informative* about others — i.e., it drives exploration. The term $f_j (t)$ is computed from interactions observed on
the current template and rewards recipients who are *directly likely* to open given what has been observed so far —
i.e., it drives exploitation. Under this reading, $\alpha$ is an explore/exploit dial: a higher $\alpha$ favors
exploration, a lower $\alpha$ favors exploitation. It is natural to schedule $\alpha$ to decay over the course of active
learning, shifting the model from exploration toward exploitation as more of the sending budget is consumed.

### 3.2 Send-Volume-Based Schedules

Let $N (t)$ be the number of emails sent by time $t$ and $N$ the total number of emails that will be sent, and
let $\mu = N (t)/N \in [0,1]$. A larger $\mu$ means less of the sending window remains in which to capitalize on
information gained from exploration, and should correspond to a smaller $\alpha$.

For endpoints $l, r \in [0,1]$, we propose two simple schedules for $\alpha (\mu)$:

**Linear schedule:**

$$\alpha (\mu) = l (1-\mu) + r\mu. \tag{5}$$

**Geometric (log-linear) schedule:**

$$\alpha (\mu) = l^{\,1-\mu}\, r^{\,\mu}. \tag{6}$$

### 3.3 Open-Rate-Based ("Confidence") Schedule

A complementary motivation looks not at how much of the budget has been spent, but at how much has actually been
*learned*: $f_j (t)$ is itself a mean of $\Sigma_{:,j}$ over recipients who have already opened, so its reliability
should increase with the number of observed opens rather than the number of emails sent.
Let $o (t) = x (t)^\top \mathbf{1}$ be the number of opens observed by time $t$ and $\tilde o = o (t)/m$. We propose a
hyperbolic decay in $\tilde o$,

$$\alpha (\tilde o) = \frac{\kappa}{\tilde o + \kappa}, \tag{7}$$

with dampening factor $\kappa$ (so that $\alpha (\kappa) = 1/2$).

*Practical note.* $\tilde o$ normalizes by the full recipient pool $m$, not by the number of recipients who have
received template $n+1$ so far. Given the dataset's baseline open rate of roughly 9% and partial rollout during active
learning, $\tilde o$ stays small for most of the active-learning phase, so $\kappa$ likely needs to be tuned to a
comparably small value for this schedule to move appreciably.

### 3.4 Combined Scheduling

Combining both views — making $\alpha$ a joint function of $\mu$ and $\tilde o$ — is a natural next step, though it
introduces additional hyperparameters and has not yet been explored empirically.

---

## 4. Recency-Weighted Historical Open Rate

The original $\phi_j = \frac{1}{n}\sum_i X_{i,j}$ weights every past template equally, disregarding the fact that
templates are sent at different times and recipient behavior can drift. We generalize this to an exponentially
recency-weighted average. Assuming templates are indexed in send order, define the relative recency of template $i$
as $d_i = (n-i)/n$ and its weight as

$$\omega_i = 2^{-d_i/h}, \tag{8}$$

with half-life $h > 0$. The weighted historic open rate is then

$$\phi_j = \frac{\omega^\top X_{:,j}}{\sum_{i=1}^n \omega_i}, \qquad \omega = (\omega_1,\dots,\omega_n)^\top. \tag{9}$$

Normalizing by $\sum_i \omega_i$ ensures the effective weights form a convex combination, so $\phi_j \in [0,1]$
regardless of $h$. The original, unweighted $\phi_j$ is recovered as $h \to \infty$ (every $\omega_i \to 1$).

---

## 5. Forward-Pass Definition of $f_j (t)$

The original $f_j (t) = \bar x (t)^\top \Sigma_{:,j}$ is a mean of $\Sigma_{:,j}$ over recipients who have opened —
i.e., an average of *individual* influences. The definition cannot capture any joint or nonlinear interaction between
openers, and thus has a limited expressive power.

We propose instead passing the current binary open-state directly through the trained autoencoder:

$$f_j (t) = \sigma\big (x (t)^\top B_{E,D}\big)_j = \big[f_{SAE} (x (t))\big]_j. \tag{10}$$

Because $\sigma (\cdot)$ is nonlinear, $\sigma (x (t)^\top B_{E,D}) \ne x (t)^\top \sigma (B_{E,D})$ in general — the
two coincide only when $x (t)$ is one-hot (exactly one opener so far). Eq. (10) therefore reflects the probability
of $j$ opening given the *joint* current state, incorporating the combined effect of all openers together, rather than a
per-recipient average.

**Remark.** Eq. (10) deliberately uses the raw $x (t) \in \{0,1\}^m$, not the normalized $\bar x (t)$. This differs from
Eq. (4), where normalizing was benign because it converted a sum of already-bounded, already-sigmoided quantities into a
mean. Here the sigmoid is applied *after* aggregation, to a raw logit; scaling that logit by $1/n_t$ before the sigmoid
does not correspond to any meaningful normalization — it simply shrinks the logit's magnitude and pushes the output
toward $0.5$ as more recipients open, i.e., toward *maximum* uncertainty, which is the opposite of the intended effect
of accumulating more observations. $x (t)$ is also the more faithful input in distributional terms: it is binary,
exactly like the rows of $X$ the autoencoder was trained on, whereas $\bar x (t)$'s fractional entries are out of
distribution relative to training.

---

## 6. Variance-Based Definition of $p_j$

$p_j$ is meant to measure the information gain from a recipient $j$ opening their email — i.e., how much observing $j$'s
email-opening would sharpen our estimate of $f_i (t)$ for other recipients $i$. The original definition,

$$p_j = \frac{1}{m-1}\sum_{i \ne j} \Sigma_{j,i}, \tag{11}$$

is the mean of row $j$ of $\Sigma$ (excluding $\Sigma_{j,j}$): it measures the overall *level* to which other
recipients' $f_i (t)$ scores would shift if $j$ opens. This rewards recipients whose opening would push others' scores
up the most, which is not the same thing as rewarding recipients whose opening would most reduce *uncertainty* about
others' scores — the latter being the actual objective of information gain.

We propose instead rewarding recipients who induce the greatest *separation* (variance) of $\Sigma_{j,:}$
(excluding $\Sigma_{j,j}$), i.e., recipients whose outcome most strongly discriminates between "will open" and "won't
open" among the rest of the population:

$$p_j = \frac{4}{m-1}\sum_{i \ne j}\big (\Sigma_{j,i} - \bar\Sigma_{j:}\big)^2, \qquad \bar\Sigma_{j:} = \frac{1}{m-1}\sum_{i \ne j}\Sigma_{j,i}. \tag{12}$$

This is the (population) variance of row $j$ of $\Sigma$ (excluding $\Sigma_{j,j}$), scaled by 4 so that the maximum
possible variance of a Bernoulli-distributed quantity (0.25) maps to 1, keeping $p_j \in [0,1]$ as in the original
definition.

---

## 7. Factored Score Function

The two terms of $s_j (t)$ carry two orthogonal interpretations: $\phi_j p_j$ vs. $f_j (t)$ is a
*historic-vs-current-data* distinction, while (per Section 3.1) it is also an *exploration-vs-exploitation* distinction.
The original linear combination (Eq. 3) conflates the two — for instance, it is not obvious why exploration should be
based solely on historic data while exploitation is based solely on current data.

We propose factoring $s_j (t)$ into a probability-of-opening estimate and a utility term, each blending historic and
current information (or influence and direct value) independently:

$$\pi_j (t) = \alpha (t)\phi_j + (1-\alpha (t))f_j (t) \qquad \text{ (estimated probability of opening)}, \tag{13}$$

$$u_j (t) = \beta (t)p_j + (1-\beta (t))\cdot 1 \qquad \text{ (utility of } j \text{ opening)}, \tag{14}$$

$$s_j (t) = \pi_j (t)\, u_j (t) = \Big (\alpha (t)\phi_j + [1-\alpha (t)]f_j (t)\Big)\Big (\beta (t)p_j + [1-\beta (t)]\Big). \tag{15}$$

In $u_j (t)$, the constant term $1$ represents the direct value of $j$ opening, while $p_j$ represents the indirect
value of $j$ opening through information gained about other recipients; $\beta (t)$
trades these off. With $\alpha (t)$ and $\beta (t)$ scheduled independently — for instance, $\alpha (t)$ via the
open-rate-based (confidence) schedule of Section 3.3 and $\beta (t)$ via a send-volume-based schedule from Section 3.2 —
the historic/current axis and the exploration/exploitation axis can be tuned separately.

A practical advantage of this factoring is that it is purely a recombination of existing primitives
($\phi_j$, $p_j$, $f_j (t)$, and the $\alpha$-schedules of Section 3), so other proposals in this document
(recency-weighted $\phi_j$, forward-pass $f_j (t)$, variance-based $p_j$) can be substituted in directly.

**Notation note.** The original paper already overloads $\alpha$: the score-blend coefficient of Eq. (3) and the
Beta-distribution shape parameter $\alpha_j (t) = Gs_j (t)$ used by Thompson Sampling share a symbol but denote
unrelated quantities. The $\beta (t)$ introduced here in Eq. (14) is a *third*, again unrelated, quantity from the
Beta-distribution parameter $\beta_j (t) = G (1-s_j (t))$.

---

## 8. Incorporating the Time-to-Open Signal

The SAE of Eq. (1) is trained purely on the binary matrix $X$, discarding a TTO signal available for every observed
open. Two recipients who both eventually open a template are treated identically regardless of their TTO — even though,
given the short operational window between batches, only fast opens are actually actionable during active learning. We
explore two complementary ways of exploiting this signal. Section 8.2 *softly*
reweights how much a given open counts toward the SAE's reconstruction target, without touching its input space. Section
8.3 instead *hard-censors* the SAE's training input to mimic the partial visibility the model actually has at query time
during active learning.

### 8.1 TTO Matrix and Decayed Labels

Let $\Delta \in (\mathbb{R}_{\ge 0} \cup \{+\infty\})^{n\times m}$ record the observed time-to-open,
with $\Delta_{i,j} = +\infty$ if recipient $j$ never opened template $i$.
Note $X_{i,j} = \mathbb{1}[\Delta_{i,j} < \infty]$, so $\Delta$ is strictly more expressive than $X$. We define a
continuous, exponentially decayed open label (matching the exponential form already used for template recency in Section
4),

$$Y_{i,j} = 2^{-\Delta_{i,j}/h_\delta}, \tag{16}$$

with TTO half-life $h_\delta > 0$ and the convention $2^{-\infty} = 0$, so $Y_{i,j} = 0$ for non-openers
and $Y_{i,j} \in (0,1]$ for openers, smoothly discounting slow opens. As $h_\delta \to \infty$, $Y_{i,j} \to X_{i,j}$
for every opener, recovering the original binary model exactly — the same limiting relationship used for the recency
weights of Section 4.

Because the two decays (template recency, opening speed) are orthogonal, $Y$ can be composed directly with the
recency-weighted $\phi_j$ of Section 4:

$$\phi_j = \frac{\omega^\top Y_{:,j}}{\sum_{i=1}^n \omega_i}. \tag{17}$$

Though this option remains to be explored emprically.

### 8.2 Soft TTO-Weighted Targets

The simplest way to use $Y$ keeps the SAE's input space binary and only replaces the reconstruction target:

$$\min_{E,D}\ \ell\big (Y,\ \sigma (XB_{E,D})\big). \tag{18}$$

Since the input space is unchanged, $p_j$ (Eq. 11 or 12) carries over mechanically unchanged, evaluated
on $\Sigma^Y = \sigma (B_{E,D})$ trained under Eq. (18). The interpretation shifts, however: $\Sigma^Y_{j,i}$ no longer
represents a plain probability that $i$ opens given only $j$ opened — it is now a TTO-decayed, "speed-weighted" version
of that quantity, blending *whether* $i$ would open with *how fast*. This shift propagates through to $p_j$, $f_j (t)$,
and $s_j (t)$. At inference, $f_j (t)$ can still be computed exactly as in Section 5, using the binary $x (t)$.

Note: Moving to a TTO-informed target $Y$ — shifts the numeric range and distribution of $s_j (t)$
relative to the original target $X$, so $G$, originally tuned against the binary model, likely needs re-tuning
under this construction.

### 8.3 Hard TTO Cutoff Thresholding

The training rows of $X$ (Section 2.2) reflect every recipient who *eventually* opened a template, however long that
took. The state vector $x (t)$ that the bandit actually queries during active learning, however, is necessarily
incomplete: for any $t < T$, a recipient who opens after $t$ still appears as a non-opener. Training rows are therefore
"converged" patterns, while active-learning queries are "still-evolving" ones — a mismatch that Eq. (18) does not by
itself address, since it only reweights the target while leaving the training input $X$ untouched.

We address this mismatch by censoring the SAE's training *input* the same way active learning censors $x (t)$.
Reusing $\Delta$ from Section 8.1, we fix a cutoff threshold $\delta_c \in [0, +\infty)$ and define a thresholded binary
matrix

$$C_{i,j} = \mathbb{1}[\Delta_{i,j} \le \delta_c]. \tag{19}$$

$C$ keeps only the opens that happened fast enough to plausibly be observed during an active operational window, and
treats every slower open as a non-open — exactly the censoring a recipient's true eventual behavior is subject to during
active learning. The SAE is then trained to recover the true, eventual open pattern from this censored view:

$$\min_{E,D}\ \ell\big (X,\ \sigma (CB_{E,D})\big). \tag{20}$$

As $\delta_c \to \infty$, $C \to X$ and the model reduces exactly to the original formulation (Eq. 1).
As $\delta_c \to 0$, $C$ collapses toward the zero matrix, discarding all training signal. The useful range in between
should be anchored to timescales the model already has available — the batch interval $T/b$ and the overall
deadline $T$.

### 8.4 Further TTO Variants (Future Work)

Several further extensions of the constructions above remain at the derivation stage and are left for future work:

- **Partial-state interpolation.** Rather than using only the binary $x (t)$ at inference under the soft-target model of
  Section 8.2, a continuous state vector $z_j (t)$, equal to $2^{-\delta_j/h_\delta}$ for recipients who have already
  opened template $n+1$ (with $\delta_j$ their observed TTO) and $0$ otherwise, could be passed through $f_{SAE}$ in
  place of $x (t)$, letting opens' *speed* — not just their occurrence — inform $f_j (t)$ mid-campaign.
  Because $f_{SAE}$'s only nonlinearity is a bounded, smooth sigmoid, this is mathematically well-posed regardless of
  whether $B_{E,D}$ was fit on binary or continuous inputs; whether it is empirically useful remains to be tested.
- **Fully continuous (TTO-to-TTO) training.** The input side could also be replaced by $Y$,
  training $\min_{E,D} \ell (Y, \sigma (YB_{E,D}))$ so the SAE both sees and predicts TTO-decayed values throughout.
  Under this variant the one-hot vector $e_j$ no longer represents a typical training point (it corresponds to an
  instantaneous open, a boundary case), so $p_j$ would need to be re-queried at a representative non-one-hot magnitude —
  e.g., scaled by recipient $j$'s own mean decayed-open rate, or by the population mean — rather than at $e_j$ directly,
  with each choice trading off a cleaner separation between $\phi_j$ and $p_j$ against a more literal query.
- **Alternative decay shapes.** A hyperbolic form, $Y_{i,j} = \kappa_\delta/ (\Delta_{i,j}+\kappa_\delta)$ (mirroring
  Eq. 7), has a heavier tail than the exponential decay of Eq. (16) and has not yet been compared against it
  empirically.

These are noted here as concrete next steps rather than developed further in this draft.

---

## 9. Deep Autoencoder

The SAE of Eq. (1) reconstructs $\sigma (xB_{E,D})$, where $B_{E,D} = ED^\top - \mathrm{diag} ([E\odot D]\mathbf{1}_m)$
is a single, fixed $m \times m$ matrix that does not depend on $x$. Every predicted entry, before the final sigmoid, is
therefore a linear combination of the input entries — the model can only capture pairwise, linear relationships between
recipients.

As already noted in Section 5, however, neither $p_j$ nor $f_j (t)$ actually requires $\Sigma$ as an explicit matrix;
under the forward-pass definitions, both are expressible purely as queries to the trained autoencoder:

$$p_j = \frac{1}{m-1}\sum_{i \ne j} \big[f_{SAE} (e_j)\big]_i, \qquad f_j (t) = \big[f_{SAE} (x (t))\big]_j. \tag{21}$$

(We adopt the forward-pass definition of $f_j (t)$ from Section 5 throughout this section, rather than the original Eq.
4.) Since $p_j$ and $f_j (t)$ depend only on the ability to query $f_{SAE}$, and not on any explicit property
of $B_{E,D}$ itself, a deeper, nonlinear autoencoder can be substituted for the shallow one without changing either
definition.

### 9.1 Two-Layer Encoder/Decoder

We propose a two-layer encoder and decoder, with a ReLU nonlinearity between the two linear layers on each side:

$$\text{Encoder:}\ \mathrm{Linear} (m, 2d) \to \mathrm{ReLU} \to \mathrm{Linear} (2d, d)$$
$$\text{Decoder:}\ \mathrm{Linear} (d, 2d) \to \mathrm{ReLU} \to \mathrm{Linear} (2d, m)$$

where $d$ retains its role as the bottleneck size from Section 2.2. The resulting reconstruction function is

$$f_{DAE} (x) = (\sigma \circ \mathrm{Decoder} \circ \mathrm{Encoder})(x). \tag{22}$$

We optionally add dropout and/or layer normalization at the bottleneck, consistent with common regularization practice
for deeper collaborative-filtering autoencoders, to control overfitting given the added capacity relative to the
original SAE.

**On the diagonal constraint.** The original SAE explicitly zeroes the diagonal of $B_{E,D}$ to rule out the trivial
solution of a recipient predicting their own entry. A deep network has no single weight matrix to constrain this way, so
we substitute a statistical safeguard instead: at each training step, we randomly mask a subset of the input entries
to $0$ and require the network to reconstruct the *full*, unmasked $X$. This makes the autoencoder denoising in the
classic sense, and forces every prediction to depend on other recipients' entries rather than a shortcut through the
recipient's own.

The training objective becomes

$$\min_{\theta}\ \ell\big (X, f_{DAE} (X;\theta)\big), \tag{23}$$

where $\theta$ collects the parameters of both the encoder and decoder layers. Under this substitution, $p_j$
and $f_j (t)$ retain their forward-pass definitions exactly, with $f_{SAE}$ replaced by $f_{DAE}$:

$$p_j = \frac{1}{m-1}\sum_{i \ne j} \big[f_{DAE} (e_j)\big]_i, \qquad f_j (t) = \big[f_{DAE} (x (t))\big]_j. \tag{24}$$

---

## 10. Experimental Setup

We evaluate each proposal via grid search over its hyperparameters, selecting the best-performing configuration on a
validation set, then reporting test-set performance. As our validation and reporting metric we use the area under the
recall curve,

$$\mathrm{AUC} = \int_0^1 \mathrm{Recall} (\tau)\, d\tau, \tag{25}$$

where $\mathrm{Recall} (\tau)$ is recall at sending fraction $\tau$ (the fraction of recipients who ultimately receive
the message). For each experiment we report Recall-AUC together with Recall@5%, @15%, @25%, and @35%, plus a smoothed
recall curve on the test set. This differs from the original paper's reported operating points (25%/50%/75%); we
additionally probe low sending fractions (5%, 15%) since that is the regime in which reducing the number of sent emails
while preserving recall is most consequential in practice, and where most of the framework's headroom for improvement is
likely to be found once recall is already high at 50–75%.

---

## 11. Results

### 11.1 Reproduction Check

Before testing modifications, we reproduced the original model on our evaluation pipeline and compared against the
original paper's reported operating points (Recall@25/50/75%):

|                  | Recall@25%     | Recall@50%     | Recall@75%     |
|------------------|----------------|----------------|----------------|
| Original paper   | 0.923 ± 0.0005 | 0.975 ± 0.0002 | 0.989 ± 0.0004 |
| Our reproduction | 0.923 ± 0.014  | 0.975 ± 0.008  | 0.989 ± 0.004  |

Point estimates match the published results exactly; our reproduction shows larger variance, consistent with a smaller
number of simulation runs for each test set template.

### 11.2 Baseline and Proposed Variants

All results below use our evaluation protocol (Section 10: Recall@5/15/25/35% and Recall-AUC), reported as mean ±
standard deviation over repeated simulations. Soft TTO-weighted training (Section 8.2) and its further extensions
(Section 8.6) remain presented as derivations pending empirical validation. TTO cutoff thresholding (Section 8.3) and
the deep autoencoder (Section 9) have been evaluated and are included below.

| Variant                      | Selected hyperparameters                                                         | Recall@5%     | Recall@15%    | Recall@25%    | Recall@35%    | AUC           |
|------------------------------|----------------------------------------------------------------------------------|---------------|---------------|---------------|---------------|---------------|
| Baseline                     | —                                                                                | 0.385 ± 0.076 | 0.821 ± 0.028 | 0.923 ± 0.014 | 0.958 ± 0.011 | 0.894 ± 0.012 |
| Linear $\alpha$-schedule     | $l=0.1,\ r=0.05$                                                                 | 0.383 ± 0.073 | 0.823 ± 0.023 | 0.927 ± 0.012 | 0.960 ± 0.010 | 0.894 ± 0.011 |
| Geometric $\alpha$-schedule  | $l=0.3,\ r=0.05$                                                                 | 0.377 ± 0.050 | 0.823 ± 0.017 | 0.927 ± 0.012 | 0.959 ± 0.010 | 0.894 ± 0.009 |
| Confidence $\alpha$-schedule | $\kappa=0.003$                                                                   | 0.341 ± 0.031 | 0.818 ± 0.022 | 0.926 ± 0.014 | 0.960 ± 0.010 | 0.891 ± 0.009 |
| Recency-weighted $\phi_j$    | $h=0.3$                                                                          | 0.447 ± 0.041 | 0.837 ± 0.018 | 0.928 ± 0.014 | 0.960 ± 0.011 | 0.903 ± 0.009 |
| Forward-pass $f_j(t)$        | —                                                                                | 0.375 ± 0.049 | 0.841 ± 0.029 | 0.937 ± 0.018 | 0.966 ± 0.010 | 0.901 ± 0.011 |
| Factored $s_j(t)$            | $\alpha(t)$: confidence, $\kappa=0.005$; $\beta(t)$: geometric, $l=0.1,\ r=0.05$ | 0.425 ± 0.026 | 0.821 ± 0.017 | 0.922 ± 0.013 | 0.959 ± 0.010 | 0.900 ± 0.008 |
| Variance-based $p_j$         | —                                                                                | 0.479 ± 0.034 | 0.837 ± 0.017 | 0.926 ± 0.013 | 0.960 ± 0.010 | 0.905 ± 0.009 |
| TTO cutoff thresholding      | $\delta_c=720$                                                                   | 0.390 ± 0.038 | 0.824 ± 0.016 | 0.924 ± 0.012 | 0.960 ± 0.010 | 0.895 ± 0.008 |
| Deep autoencoder             | $d=16$                                                                           | 0.400 ± 0.090 | 0.815 ± 0.085 | 0.936 ± 0.019 | 0.963 ± 0.011 | 0.898 ± 0.019 |

![Baseline Model](images/baseline_recalls_0_1.png)

![Linear Alpha Scheduling](images/linear_alpha_recalls_0_1.png)

![Geometric Alpha Scheduling](images/geometric_alpha_recalls_0_1.png)

![Confidence Alpha Scheduling](images/confidence_based_alpha_recalls_0_1.png)

![Recency-weighted Phi](images/exp_template_weight_recalls_0_1.png)

![Forward-pass F](images/forward_pass_f_recalls_0_1.png)

![Factored S](images/alternative_s_recalls_0_1.png)

![Variance-based P](images/variance_based_p_recalls_0_1.png)

![TTO Cutoff Thresholding](images/tto_cutoff_recalls_0_1.png)

![Deep Autoencoder](images/deep_autoencoder_recalls_0_1.png)

---

## 12. Discussion and Open Questions

**Where the gains concentrate.** Across variants, differences from baseline are largest at the lowest sending fraction
 and shrink as the sending fraction grows, which is expected: baseline recall already exceeds 0.92 by 25%,
leaving little headroom for any variant to improve on at that point. The largest Recall@5% improvements come from the
variance-based $p_j$ (0.479 vs. 0.385 baseline) and the recency-weighted $\phi_j$ (0.447), followed by the
factored $s_j (t)$ (0.425). TTO cutoff thresholding and the deep autoencoder both produce small positive shifts at
Recall@5% (0.390 and 0.400, respectively) and comparable AUC gains (0.895 and 0.898 vs. 0.894 baseline), but neither is
competitive with the strongest variants at this operating point. The deep autoencoder's Recall@15% (0.815 ± 0.085) is,
notably, both slightly *below* baseline and substantially more variable than every other variant in the table — roughly
3–5$\times$ the standard deviation of the next-noisiest entries — suggesting the added capacity is not yet
well-regularized at this dataset size; the denoising mask rate and dropout settings of Section 9.1 warrant a wider
search before drawing conclusions about the architecture itself. The $\alpha$-scheduling variants alone show little to
no improvement at 5% and, for the confidence schedule, a *decrease* (0.341) — consistent with the practical note in
Section 3.3 that this schedule's effective range may be too narrow given the dataset's baseline open rate; the choice
of $\kappa$ likely needs revisiting.

**Open questions carried over from the derivations above:**

- Whether combined $\alpha$-scheduling (joint function of $\mu$ and $\tilde o$, Section 3.4) is worth its added
  hyperparameter count.
- Empirical comparison of the soft-target (Section 8.2) and hard-cutoff (Section 8.3) TTO constructions against each
  other.
- Whether $\phi_j$ should still appear separately in $s_j (t)$/$\pi_j (t)$ under the recipient-scaled TTO-to-TTO
  redefinition of $p_j$ (Section 8.6), given the activity-level double-counting risk noted there.
- Whether the deep autoencoder's high Recall@15% variance is a regularization artifact or a genuine instability of the
  architecture; a wider sweep over masking rate, dropout, and bottleneck width $d$ is needed before drawing conclusions.
- Combining the deep autoencoder (Section 9) with either TTO construction (Section 8), since the two modifications touch
  different parts of the model — architecture vs. training signal — and are not mutually exclusive.
- All experiments to date test one modification at a time; joint search over combinations (e.g., variance-based $p_j$
  together with recency-weighted $\phi_j$) has not yet been run and may compound the individual gains seen here.

---

## 13. Conclusion and Future Work

We have proposed ten refinements to the SAE–Thompson Sampling framework of Žid et al. (2025), spanning the score
function's blend coefficient, its historic and influence terms, its current-state confidence term, its overall
factorization, the training signal available to the underlying autoencoder — both a soft time-to-open reweighting and a
hard censoring of training inputs — and the autoencoder's architecture itself. Preliminary experiments across the nine
modifications evaluated so far (all but the soft TTO-weighted target) suggest the largest practical gains are
concentrated at low sending fractions, with the variance-based influence score and recency-weighted open rate the
strongest individual candidates; TTO cutoff thresholding and the deep autoencoder provide smaller, more mixed
improvements, the latter alongside a stability concern that needs addressing before the architecture can be recommended
on its own. Immediate next steps are: (1) implement and evaluate the soft TTO-weighted target of Section 8.2, and
compare it against the hard cutoff of Section 8.3; (2) stabilize the deep
autoencoder via a wider regularization sweep; (3) run combined-modification experiments rather than isolated ablations,
including deep-autoencoder $\times$ TTO combinations.

---

## References

Žid, Č., Alves, R., and Kordík, P. 2025. Active Recommendation for Email Outreach Dynamics. In *Proceedings of the 34th
ACM International Conference on Information and Knowledge Management (CIKM '25)*, November 10–14, 2025, Seoul, Republic
of Korea. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3746252.3760832