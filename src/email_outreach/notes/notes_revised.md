# Active Recommendation for Email Outreach Dynamics Improvement Notes

This document contains notes on possible improvements and refinements of the Contextual Multi Arm Bandit with Thompson
Sampling model established
by [Active Recommendation for Email Outreach Dynamics](https://dl.acm.org/doi/10.1145/3746252.3760832) paper

We explore different practical and theoretical refinements to the model, including but not limited to:

- Alpha coefficient scheduling
- Adjusted template weight
- Refined f definition

## Recap

TBA

### Dataset

We consider dataset $\mathbb{X} \in \{0, 1\}^{n,m}$ where n is a number of templates, m is a number of recepients
and $X_{i,j} = 1$ if user $j$
opened email template $i$ and $0$ otherwise.

### Autoencoder

We train a shallow autoencoder (SAE) as follows:

$$\min_{E,D \in \mathbb{R}^{m, d}} l (X, \sigma (X B_{E,D}))$$

where

$$B_{E,D} = E D^T - diag[(E \odot D) \mathbf{1}]$$

and $l$ is element-wise BCE loss

Given the trained autoencoder, we define a function
$f_{SAE} (x) = \sigma (x B_{E,D})$ which maps a binary vector of opened templates to a vector of probabilities of
opening other templates.

Additionally, we define $\Sigma = \sigma (B_{E,D})$. Note that
$\Sigma_{i,j} = e_i^T \Sigma e_j = f_{SAE} (e_i)e_j$. Thus,
$\Sigma_{i,j}$ represents a probability of user $j$ opening a template if only user $i$ has already opened it - a
measure of influence of user
$i$ on user $j$.

> **Note on direction/asymmetry:** $B_{E,D}$ (and hence $\Sigma$) is
> *not* symmetric in general, since $E \neq D$, so
> $\Sigma_{i,j} \neq \Sigma_{j,i}$ in general. Throughout this document
> the convention is: **row index = the recipient who has already opened
> (the "source"/conditioner), column index = the recipient whose opening
> probability is being predicted (the "target")**. This is the
> convention that makes $p_j$ (below, uses row $j$) an *influence* score
> and $f_j (t)$ (below, uses column $j$) a
> *received-influence/confidence* score. It's worth flagging that the
> paper's own prose description of $\Sigma_{i,j}$ ("probability that $i$
> opens given only $j$ opened") reads with $i$/$j$ reversed relative to
> this convention — but the paper's own formulas for $p_j$ and $f_j (t)$
> only work under the convention above, so this appears to be a swap in
> the paper's descriptive text rather than an error in this reasoning.

### Arms score function

The core part of the model is the arms score function, which is defined as:
$$s_j (t) = \alpha \cdot \phi_j p_j + (1 - \alpha) \cdot f_j (t)$$

Where:

$$\phi_j = \frac{1}{n}\sum_{i=1}^{n} X_{i,j}$$

- The historic probability of user $j \in \{1,2,...,m\}$ opening an email

$$p_j = \frac{1}{m-1}\sum_{i=1, i\ne j}^{m} \Sigma_{j,i}$$

- Measure of influence of user $j$ opening an email from a template on other users opening their email from this
  template.

$$f_j (t) = \bar x (t)^T \Sigma_{:j}$$

- Measure of probability of user $j$ opening an email from template at time $t$ given the current state of opened
  templates $x (t)$.

> **Note on normalization:** the published paper's Eq. (4) writes this
> term with the *unnormalized* $x_{n+1} (t)$ rather than
> $\bar x_{n+1} (t)$. We use the normalized
> $\bar x_{n+1} (t) = x_{n+1} (t)/n_t$ throughout, consistent with the
> paper's Figure 1 and with the paper's own description of $f_j (t)$ as a
> *mean* (point-estimate) over recipients who have opened — a mean
> requires normalizing by the number of openers, which the raw
> $x_{n+1} (t)$ does not do.

## Alpha Scheduling

The model presented in the paper uses a fixed alpha coefficient
$\alpha \in [0,1]$. We further explore the possibility of using a dynamic alpha coefficient $\alpha (t)$ that changes
over time.

The overall score $s_j (t)$ for a given user $j$ is not a measure of probability of user $j$ opening an email, but
rather a measure of
"usefulness" of sending an email to user $j$.

### Sent Mail - Based Alpha Scheduling

As was noted in previous sections $\phi_j$ represents a historic estimate of probability of user $j$ opening an email,
while $p_j$
represents a measure of "influence" of user $j$ on other users opening their emails. Thus, the part $\phi_j p_j$
represents the informativeness of sending an email to user $j$ in terms of gaining more information about other users -
which corresponds to exploration.

On the other hand, $f_j (t)$ represents a measure of direct probability of user $j$ opening an email given the current
state of opened templates - which corresponds to exploitation.

Thus $\alpha$ can be interpreted as a measure of exploration vs exploitation trade-off. A higher $\alpha$ corresponds to
more exploration, while a lower $\alpha$ corresponds to more exploitation.

In this spirit, it is reasonable to use alpha scheduling with decreasing value of alpha, representing gradual shift from
exploration to exploitation.

Let $N (t)$ represent the number of total emails sent at time $t$ and
$N$ - total number of emails we send overall.

> **Renamed from** $m (t)$/$M$: the original draft used $m (t)$ and $M$
> for these quantities, but $m$ is already fixed throughout this
> document as the total recipient count (and reappears below in
> $\tilde o = o (t)/m$). Renaming to $N (t)$/$N$ avoids that collision —
> no formula changes here, just notation.

In this context, we consider $\alpha$ to be a function of ratio
$\mu = \frac{N (t)}{N}$. Greater fraction represents smaller window for capitalizing on information gained from
exploration - and should result in a smaller value of alpha.

#### Linear Alpha Scheduling

We consider a linear (in terms of $\mu$) decrease of $\alpha (\mu)$ from
$l \in [0,1]$ to $r \in [0,1]$:
$$\alpha (\mu) = l \cdot (1-\mu) + r \cdot \mu$$

In terms of $t$:

$$\alpha (t) = l \cdot (1-\frac{N (t)}{N}) + r \cdot \frac{N (t)}{N}$$

#### Geometric (Log-Linear) Alpha Scheduling

We consider a geometric/log-linear (in terms of $\mu$) decrease of
$\alpha (\mu)$ from $l \in [0,1]$ to $r \in [0,1]$:
$$\alpha (\mu) = l ^ { (1-\mu)} \cdot r ^ {\mu}$$

In terms of $t$:

$$\alpha (t) = l ^ { (1-\frac{N (t)}{N})} \cdot r ^ \frac{N (t)}{N}$$

### Open Mail - Based Alpha Scheduler

An alternative (complementary) view on the 2 parts of a score function
$s_j (t)$ is that the first part $\phi_j p_j$ is calculated completely based on historic data (notice it does not depend
on $t$), while
$f_j (t)$ is based on actual observed interactions in the current template (notice the dependence on $t$ and the fact
that $f_j (0) = 0$).

Thus, another motivation for decreasing alpha comes from increasing
"trust" in $f_j (t) = \bar x (t)^T \Sigma_{:j}$ - which itself is a mean (point-estimate) of values in $\Sigma_{:j}$
among recipients who have already opened an email.

Let $o (t)$ represent number of opened emails at time $t$
($o (t) = x (t) ^ T \mathbf{1}$)

Thus, $\alpha (t)$ should be a decreasing function of
$\tilde o = o (t)/m$ - proportion of opened emails at time $t$ to the number of users - representing a transition from
historic data based estimates to actual observed interactions - based estimates.

#### Confidence Alpha Scheduling

We consider a Hyperbolic (in terms of $\tilde o$) decrease of alpha

$$\alpha (\tilde o) = \frac{\kappa}{\tilde o + \kappa}$$
Here $\kappa$ - is a dampening factor ($\alpha (\kappa) = \frac{1}{2}$)

In terms of time:
$$\alpha (t) = \frac{\kappa}{o (t)/m + \kappa}$$

> **Practical note:** $\tilde o$ normalizes by the *total* recipient
> pool $m$, not by how many recipients have actually received template
> $n+1$ so far. Given a \~9% baseline open rate and partial rollout
> during active learning, $\tilde o$ will stay quite small for most of
> the active learning phase, so $\alpha (\tilde o)$ may barely move
> unless $\kappa$ is tuned to a comparably small value (well below
> typical eventual open-rate levels). Worth validating empirically, or
> considering normalizing by the number of recipients sent-so-far
> instead of $m$.

#### Combined Alpha Scheduling

It might be interesting to combine both approaches to alpha scheduling, where $\alpha (t)$ is a function of both $\mu$
and $\tilde o$, though this idea remains to be explored (excessive number of hyperparameters might be a problem).

## Template Weights

The original model definition of historic probability of opening an
email $$\phi_j = \frac{1}{n} \sum_{i=1}^{n} X_{i,j}$$ gives equal weight to each template $X_{i:}$

However, this disregards the fact that the templates were sent at different times, and that user behavior can change
over time.

Thus, it is reasonable to consider a weighted average of opened templates, where more recent templates have higher
weights than older templates.

### Exponential Template Weights

We assume that templates are sent at uniform time intervals, ordered by their index $i$ (that is, template $X_{1:}$ was
sent first, template
$X_{2:}$ was sent second, and so on).

For a given template $X_{i:}$ we define its weight as:
$$\omega_i = 2 ^ { (\frac{-d_i}{h})}$$

where
$$d_i = \frac{n-i}{n}$$

- relative time since template $X_{i:}$ was sent

Where $h \gt 0$ is the half-life of the weight decay.

The weighted average of opened templates is then defined as:
$$\phi_j = (\omega^T X_{:j}) / \sum_{i=1}^{n} \omega_i$$

Where

$$\omega = \begin{bmatrix} \omega_1 \\ \omega_2 \\ ... \\ \omega_n \end{bmatrix}$$

Dividing by $\sum_i \omega_i$ ensures the *effective* weights
$\omega_i / \sum_k \omega_k$ form a proper convex combination (they sum to 1), which guarantees $\phi_j \in [0,1]$
regardless of the raw magnitude of the $\omega_i$.

Note that the original definition of $\phi_j$ is a special case of this weighted average with uniform weights
(as $h \to \infty$, every
$\omega_i \to 1$).

## Improved Definition of f

The current definition of $f_j (t) = \bar x (t)^T \Sigma_{:j}$ is a mean of values in $\Sigma_{:j}$ among recipients who
have already opened an email - thus representing mean influence of users who have already opened an email on user $j$
opening an email.

However, this definition considers only individual recipients' influences and disregards their interactions.
Because $\Sigma_{:j}$ is precomputed and $\bar x (t)^T \Sigma_{:j}$ is a linear combination of its entries, the combined
effect of multiple simultaneous openers is just the average of their individual effects — it cannot capture any
joint/nonlinear interaction between openers.

We propose a refined definition of $f_j (t)$ that instead passes the full current state through the trained autoencoder
directly, taking into consideration the combined influence of users who opened an email on user $j$ opening an email, as
well as the influence of users who have not opened an email on user $j$ opening an email:

$$f_j (t) = \sigma (x (t)^T B_{E,D}) e_j = f_{SAE} (x (t)) e_j$$

> **Deliberately using raw** $x (t)$, not $\bar x (t)$, here. This is the
> and the distinction matters:
>
> - In the *original* $f_j (t) = \bar x (t)^T \Sigma_{:j}$ above, dividing
>   by $n_t$ converts a sum of already-bounded, already-sigmoided
>   quantities ($\Sigma_{i,j} \in [0,1]$) into a mean. That's a benign,
>   well-justified normalization.
> - Here, the sigmoid is applied *after* aggregation, to the raw logit
>   $x (t)^T B_{E,D}$. Scaling that logit by $1/n_t$ before the sigmoid
>   doesn't "normalize" anything meaningful — it just shrinks the
>   logit's magnitude, which pushes $\sigma (\cdot)$ toward $0.5$
>   regardless of the true underlying signal (and regardless of the sign
>   of entries in $B_{E,D}$, which can be positive or negative). As more
>   recipients open ($n_t$ grows), this would systematically drag
>   $f_j (t)$ *toward maximum uncertainty* — the opposite of what more
>   observed opens should do to confidence.
> - $x (t)$ is also simply the more faithful input: it's defined to be
>   binary, $\{0,1\}^m$, exactly like the rows of $X$ the autoencoder
>   was trained on. $\bar x (t)$, with fractional entries like $1/n_t$,
>   is the input that's out of distribution relative to training — not
>   the other way around.

Because $\sigma (\cdot)$ is nonlinear,
$\sigma (x (t)^T B_{E,D}) \neq x (t)^T \sigma (B_{E,D}) = x (t)^T \Sigma$ in general (they coincide only when $x (t)$ is
one-hot, i.e. exactly one opener). So $f_{SAE} (x (t))$ is a vector of probabilities of opening templates given the
*joint* current state of opened templates $x (t)$, and the refined $f_j (t)$ represents a probability of user $j$
opening an email given the current state of opened templates, taking into account both individual and combined
influences of users who have opened an email, as well as the influence of users who have not opened an email.

# Improved Definition of p

As was noted in previous sections, it is desirable that $p_j$ represents a measure of information gain from sending an
email to user $j$ in terms of gaining more information about other users, given that user $j$ opens their email.

The only way this information can be utilized is a change in $f_j (t)$ scores of other users.

However, the current definition of
$$p_j = \frac{1}{m-1}\sum_{i=1, i\ne j}^{m} \Sigma_{j,i}$$
is a mean of values in $\Sigma_{j:}$ among all recipients except user $j$ - thus representing an overall change of level
of estimated $f_j (t)$ values for other users if user $j$ opens an email. This definition rewards users, who, if they
open their email, will lead to a maximal $f_j (t)$ scores level increase, which is not consistent with the goal of
maximizing information gain.

The real property we want from users with high $p_j$ is not maximal increase in $f_j (t)$ scores of other users, but
rather maximal uncertainty reduction in $f_j (t)$ scores of other users. In other words, we want to reward users who, if
they open their email, will lead to a maximal reduction in uncertainty of $f_j (t)$ scores of other users. This is
consistent with the goal of maximizing information gain.

Thus, we propose a refined definition of $p_j$ that rewards maximal separation of $f_j (t)$ scores of other users if
user $j$ opens an email, which is a measure of uncertainty reduction in $f_j (t)$ scores of other users:

$$
p_j = \frac{4}{m-1}\sum_{i=1, i\ne j}^{m} (\Sigma_{j,i} - \bar\Sigma_{j:})^2
$$

where $\bar\Sigma_{j:} = \frac{1}{m-1}\sum_{i=1, i\ne j}^{m} \Sigma_{j,i}$ is the mean of values in $\Sigma_{j:}$ among
all recipients except user $j$.

Effectively, this is the variance of values in $\Sigma_{j:}$ among all recipients except user $j$, scaled to be
in $[0,1]$ (the factor of 4 ensures that the maximum possible variance of a Bernoulli variable, which is 0.25, scales to
1).

# Improved definition of s

As was noted in previous sections, 2 terms of $s_j (t)$: $\phi_j p_j$ and $f_j (t)$ have 2 orthogonal interpretations:
exploration vs exploitation and historic vs current data.

The current definition of $s_j (t)$ as a linear combination of $\phi_j p_j$ and $f_j (t)$  is not consistent with those
2 different views. For instance, it is not clear why should exploration be based solely on historic data, while
exploitation is based solely on current data.

Thus, we propose a refined definition of $s_j (t)$:

$$
s_j (t) = \pi_j (t) u_j (t)
$$

Where

$$
\pi_j (t) = \alpha (t)\phi_j + (1-\alpha (t))f_j (t)
$$
An estimate of the probability of user $j$ opening an email at time $t$, based on both historic and current data.

$$
u_j (t) = \beta (t)p_j + (1-\beta (t))1
$$

Utility of user $j$ opening an email, accounting for both exploration and exploitation. The term $1$ represents the
direct utility of user $j$ opening an email, while $p_j$ measures indirect utility of user $j$ opening an email in terms
of information gain about other users.

With $\pi_j (t)$ and $u_j (t)$ defined as above, the refined score function can be expressed as:
$$
s_j (t) = \biggl (\alpha (t)\phi_j + [1-\alpha (t)]f_j (t)\biggr) \biggl (\beta (t)p_j + [1-\beta (t)]1\biggr)
$$

Note that the refined definition of $s_j (t)$ has 2 hyperparameters $\alpha (t)$ and $\beta (t)$, which can be scheduled
independently, allowing for more flexibility in controlling the exploration vs exploitation trade-off and historic vs
current data trade-off.

> It makes sense to use confidence-based scheduling for $\alpha (t)$, as discussed in previous sections, while using
> sent-mail-based scheduling for $\beta (t)$, as discussed in previous sections.

A strong advantage of the refined definition of $s_j (t)$ is reusal of the primitives of the original definition
of $s_j (t)$, which allows for plug-and-play reusal of the of the improvements and modifications proposed in previous
sections.

## Incorporating Time-to-Open (TTO)

The autoencoder is currently trained purely on $X$, the binary open/no-open matrix. This discards a second signal we
have available for every observed open: the time-to-open (TTO). Two recipients who both eventually open a template are
treated identically by the SAE, whether one opened in five minutes or five days - even though, given the short
operational window ($w$ between batches, $T$ overall), only fast opens are actually actionable for the active learning
process. We explore how to fold TTO into training of the SAE (and, by extension, $\phi_j$), without touching the
definitions of $s_j (t)$, $p_j$, or $f_j (t)$ themselves any further than necessary.

### TTO matrix and decayed label

We introduce $\Delta \in (\mathbb{R}_{\ge 0} \cup \{+\infty\})^{n,m}$, the matrix of observed TTOs, where $\Delta_{i,j}$
is the time between sending and opening of template $i$ by recipient $j$. For recipients who did not open template $i$,
we set $\Delta_{i,j} = +\infty$.

Note that $\Delta$ is strictly more expressive than $X$, since $X_{i,j} = \mathbb{1}[\Delta_{i,j} < \infty]$ - i.e. $X$
can be recovered from $\Delta$, but not vice versa.

We define a decayed open label, exponential in $\Delta$ (consistent with the exponential form already used for template
recency in the Template Weights section):

$$Y_{i,j} = 2 ^ { (\frac{-\Delta_{i,j}}{h_\delta})}$$

where $h_\delta \gt 0$ is a TTO half-life hyperparameter, and we adopt the convention $2 ^ {-\infty} = 0$, so that
$Y_{i,j} = 0$ for recipients who did not open. For recipients who did open, $Y_{i,j} \in (0, 1]$, smoothly discounting
slow opens rather than counting them identically to fast ones.

> **Special case:** as $h_\delta \to \infty$, $Y_{i,j} \to X_{i,j}$ for every opener, recovering the original binary
> model exactly - the same "special case" relationship used to justify the template weights generalization above. A
> hard "cutoff" (treat any open slower than some threshold $\delta_c$ as a non-open) is the opposite limit,
> $h_\delta \to 0$, which turns $Y$ into a step function of $\Delta$.


> A hyperbolic form,
> $Y_{i,j} = \kappa_\delta / (\Delta_{i,j} + \kappa_\delta)$, mirroring the Confidence Alpha Scheduling section, is a
> reasonable alternative with a heavier tail than the exponential form above; both remain to be validated empirically
> against each other.

> This composes directly with the weighted $\phi_j$ from the Template Weights section, since the two decays (recency of
> template, speed of open) are orthogonal:
>
> $$\phi_j = (\omega^T Y_{:j}) / \sum_{i=1}^{n} \omega_i$$

We consider three ways of using $Y$ in place of $X$ when training the SAE, differing in whether the *input* to the
autoencoder stays binary ($X$) or is replaced by the continuous $Y$.

### Model A: $X \to Y$

The simplest option keeps the SAE's input space unchanged (binary $x$), and only replaces the reconstruction target:

$$\min_{E,D} l (Y, \sigma (X B_{E,D}))$$

Since the input space is untouched, $e_j$ remains an entirely ordinary training-time input (a one-hot row is still
exactly what a template with a single opener looks like), so $p_j$ carries over **mechanically unchanged**:

$$p_j = \frac{1}{m-1}\sum_{i=1, i \ne j}^{m} \Sigma^{Y}_{j,i}, \quad \Sigma^Y = \sigma (B_{E,D})$$

> **Interpretation shift:** $\Sigma^Y_{j,i}$ no longer represents a probability of $i$ opening given only $j$ opened -
> it now represents a TTO-decayed, "speed-weighted" version of that quantity, blending *whether* $i$ would open with
> *how fast*. This shift propagates to $p_j$ and $f_j (t)$ as well, and ultimately to $s_j (t)$.

At inference, $f_j (t)$ can still be computed exactly as in the Improved Definition of f section (or a classic
definition, it is also possible here), using the binary
$x (t)$. Optionally, we can additionally exploit the TTO of recipients who have *already* opened template $n+1$ by time
$t$, via the following extension.

#### The $z (t)$ interpolation extension

Define a continuous, partially TTO-informed state vector at decision time $t$:

$$z_j (t) = \begin{cases} 2 ^ { (\frac{-\delta_j}{h_\delta})} & \text{recipient } j \text{ has opened by } t \\ 0 & \text{otherwise} \end{cases}$$

where $\delta_j$ is the observed TTO of recipient $j$ for template $n+1$. We then define:

$$f_j (t) = \sigma (z (t) ^ T B_{E,D}) e_j = f_{SAE} (z (t)) e_j$$

> **Note on validity:** $x ^ T B_{E,D}$ is linear in $x$, and $z (t) \in [0,1]^m$ always, regardless of
> whether $B_{E,D}$
> was fit only on binary corners of the hypercube. The only nonlinearity in $f_{SAE}$ is the final sigmoid, which is
> smooth and bounded everywhere. So evaluating $f_{SAE}$ at $z (t)$ is a mathematically tame extension of a model
> trained on binary rows, not a leap into genuinely undefined territory - but whether it is empirically *useful*
> (i.e. whether the TTO of early openers carries predictive signal the binary $x (t)$ does not) remains to be
> validated.

### Model B: $Y \to Y$

A more expressive option lets the SAE also *see* TTO information on the input side:

$$\min_{E,D} l (Y, \sigma (Y B_{E,D}))$$

Here the input space becomes continuous, $Y_{i,j} \in [0,1)$.

> This means $e_j$ - corresponding to $\delta_j = 0$, an
> instantaneous open - is now a boundary point of the training distribution rather than a typical one, and querying
> $\Sigma^Y_{j,:}$ at $e_j$ answers "if $j$ opened as fast as physically possible", not "if $j$ opened". We therefore
> should **not** compute $p_j$ at $e_j$ under this model.

#### $p_j$ under Model B

We instead query $f_{SAE}$ at a point representative of the training distribution, scaled by a magnitude in the range
the model actually saw during training. Two variants:

**Recipient-scaled:**

$$p_j = \frac{1}{m-1}\sum_{i=1, i \ne j}^{m} f_{SAE} (\bar Y_j \, e_j) e_i, \quad \bar Y_j = \frac{1}{n}\sum_{i=1}^{n} Y_{i,j}$$

This asks "what if $j$ opens the way $j$ typically does" rather than "what if $j$ opens instantly" - a realistic
expected effect rather than a best-case, counterfactual one.

> **Note on coupling with $\phi_j$:** $\bar Y_j$ is, up to the choice of decay applied, the same quantity as $\phi_j$
> (or its weighted generalization from the Template Weights section). This means $p_j$ now has recipient $j$'s own
> activity level baked into the query magnitude, while $s_j (t)$ (and $\pi_j (t)$, in the refined definition above)
> still multiplies by / combines with $\phi_j$ separately. The original design deliberately kept activity ($\phi_j$)
> and magnitude-free influence ($p_j$) as independent, multiplicatively-combined axes; this recipient-scaled variant
> partially collapses that separation and risks double-counting activity. Whether $\phi_j$ should still appear
> separately in $s_j (t)$ / $\pi_j (t)$ under this variant is an open question, to be resolved empirically.

**Population-mean (decoupled):**

$$p_j = \frac{1}{m-1}\sum_{i=1, i \ne j}^{m} f_{SAE} (\bar Y \, e_j) e_i, \quad \bar Y = \frac{1}{nm}\sum_{k=1}^{n}\sum_{l=1}^{m} Y_{k,l}$$

Querying at the *population* mean rather than recipient $j$'s own mean avoids the coupling above entirely, at the cost
of asking a coarser question ("what if $j$ opens at a typical rate for *any* recipient" rather than "...for $j$
specifically"). This preserves the original separation of concerns between $\phi_j$ and $p_j$ cleanly, and is the safer
default of the two variants pending empirical comparison.

$f_j (t)$ under Model B can reuse the same $z (t)$ construction as Model A; if anything, $z (t)$ is more in-distribution
here, since the model was trained on continuous inputs rather than only binary ones.


### Downstream considerations

- **$G$ calibration.** $s_j (t)$ (or $\pi_j (t)$ in the refined definition) feeds $\alpha_j (t) = G s_j (t)$ and
  $\beta_j (t) = G (1 - s_j (t))$. Regardless of how $s_j (t)$ is interpreted, this remains a Bernoulli-flavored
  construction; moving to a TTO-blended $\Sigma^Y$ shifts the numeric range/distribution of $s_j (t)$, so $G$
  (tuned against the original $X$-based $\Sigma$) likely needs re-tuning under any of Models A/B.