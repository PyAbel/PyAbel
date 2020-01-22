.. _rBasexmath:

rBasex: mathematical details
============================


Coordinates
-----------

The coordinate systems used here are defined such that the image is in the
:math:`xy` plane, with the :math:`y` axis being the (vertical) axis of
symmetry. Thus the image-space polar coordinates are

.. math::
    \begin{aligned}
        r &= \sqrt{x^2 + y^2}, \\
        \theta &= \arctan \frac{y}{x}, \\
    \end{aligned}

with the polar angle :math:`\theta` measured from the :math:`y` axis. The
corresponding coordinate system for the underlying 3D distribution has the same
:math:`x` and :math:`y` axes, plus a perpendicular :math:`z` axis, so the
distribution-space spherical coordinates are

.. math::
    \begin{aligned}
        \rho &= \sqrt{x^2 + y^2 + z^2}, \\
        \theta' &= \arctan \frac{y}{\sqrt{x^2 + z^2}}, \\
        \phi' &= \arctan \frac{z}{x},
    \end{aligned}

with the polar angle :math:`\theta` also measured from the :math:`y` axis (the
symmetry axis). The Abel transform performs a projection along the :math:`z`
axis:

.. math::
    (x, y, z) \mapsto (x, y),

as shown by the dashed line:

.. plot:: transform_methods/rbasex-coord.py
    :align: center

This figure also illustrated important relations between the 3D and 2D radii:

.. math::
    \rho = \sqrt{r^2 + z^2}.

and polar angles:

.. math::
    \left.\begin{aligned}
        \cos\theta = y / r \\
        \cos\theta' = y / \rho
    \end{aligned}\right\}
    \quad \Rightarrow \quad
    \cos\theta' = \frac{r}{\rho} \cos\theta.


Basis functions
---------------

The 3D distribution basis consists of direct products of radial and angular
basis functions. The radial functions are triangular functions centered at
integer radii (whole pixels), spanning ±1 pixel:

.. plot:: transform_methods/rbasex-bR.py
    :align: center

.. math::
    b_R(\rho) = \begin{cases}
        \rho - (R - 1), & R - 1 < \rho < R, \\
        (R + 1) - \rho, & R \leqslant \rho < R + 1, \\
        0 & \text{otherwise}
    \end{cases}

(and :math:`b_0(\rho)` does not have the inner part, :math:`R - 1 < \rho < R`,
since :math:`\rho \geqslant 0`). These functions actually form a basis of the
piecewise linear approximation with nodes at each pixel.

The angular basis functions are just integer powers of :math:`\cos\theta'` from
0 up to the highest order expected in the distribution. Hence the overall 3D
distribution basis functions are

.. math::
    b_{R,n}(\rho, \theta', \varphi') = b_R(\rho) \cos^n \theta'

(due to cylindrical symmetry, there is no dependence on the azimuthal angle
:math:`\varphi'`).

The 2D image basis functions are, correspondingly, the projections of these
distribution basis functions along the :math:`z` axis:

.. math::
    p_{R,n}(r, \theta) =
        \int_{-\infty}^\infty b_{R,n}(\rho, \theta', \varphi') \,dz =
        2 \int_0^\infty b_{R,n}(\rho, \theta', \varphi') \,dz.


Basis projections
-----------------

As mentioned earlier, :math:`\cos\theta' = \tfrac{r}{\rho} \cos\theta`, so we
can write

.. math::
    \begin{aligned}
        p_{R,n}(r, \theta) &= 2 \int_0^\infty b_R(\rho) \cos^n\theta' \,dz = \\
            &= 2 \int_0^\infty b_R(\rho) \left(\frac{r}{\rho}\right)^n \cos^n\theta \,dz = \\
            &= 2 \int_0^\infty b_R(\rho) \left(\frac{r}{\rho}\right)^n \,dz \cdot \cos^n\theta.
    \end{aligned}

In other words, basis projection are also separable into radial and angular
parts:

.. math::
    p_{R,n}(r, \theta) = p_{r;n}(r) \cos^n\theta,

with the same angular dependence, but their radial parts are different for
different angular orders (thus projections of functions with angular dependence
different from a singe cosine power, for example, :math:`\sin^2\theta'` or
Legendre polynomials, would be *not* separable).

Since :math:`b_R(\rho)` consists of two segments that are linear functions of
:math:`\rho`, the integrals above can be expressed in terms of the integrals

.. math::
    \int\limits_{z_\text{min}}^{z_\text{max}}
        \rho \left(\frac{r}{\rho}\right)^n \,dz =
    r \int\limits_{z_\text{min}}^{z_\text{max}}
        \left(\frac{r}{\rho}\right)^{n-1} \,dz

and

.. math::
    \int\limits_{z_\text{min}}^{z_\text{max}}
        R \left(\frac{r}{\rho}\right)^n \,dz =
    R \int\limits_{z_\text{min}}^{z_\text{max}}
        \left(\frac{r}{\rho}\right)^n \,dz

with appropriate lower and upper limits. That is, only the antiderivatives of
the form

.. math::
    F_n(r, z) = \int \left(\frac{r}{\rho}\right)^n \,dz

with integer :math:`n` from −1 to the highest angular order are needed. They
all can be computed analytically and are listed in the following table (as a
reminder, :math:`\rho = \sqrt{r^2 + z^2}`):

.. list-table::
    :header-rows: 1
    :widths: auto
    :align: center

    * - :math:`n`
      - :math:`F_n(r, z)`
    * - :math:`-1`
      - :math:`\frac12 z \left(\frac{r}{\rho}\right)^{-1} +
        \frac12 r \ln(z + \rho)`
    * - :math:`\phantom{-}0`
      - :math:`z`
    * - :math:`\phantom{-}1`
      - :math:`r \ln(z + \rho)`
    * - :math:`\phantom{-}2`
      - :math:`r \arctan\frac{z}{r}`
    * - :math:`\phantom{-}3`
      - :math:`z \left(\frac{r}{\rho}\right)`
    * - :math:`\phantom{-}4`
      - :math:`\frac12 z \left(\frac{r}{\rho}\right)^2 +
        \frac12 r \arctan\frac{z}{r}`
    * - :math:`\phantom{-}5`
      - :math:`\frac13 z \left(\frac{r}{\rho}\right)^3 +
        \frac23 z \left(\frac{r}{\rho}\right)`
    * - :math:`\phantom{-}6`
      - :math:`\frac14 z \left(\frac{r}{\rho}\right)^4 +
        \frac38 z \left(\frac{r}{\rho}\right)^2 +
        \frac38 r \arctan\frac{z}{r}`
    * - :math:`\phantom{-}\vdots`
      - :math:`\vdots`
    * - :math:`2m \geqslant 2`
      - :math:`z \sum\limits_{k=1}^{m-1} a_k \left(\frac{r}{\rho}\right)^{2k} +
        a_1 r \arctan\frac{z}{r}, \quad
        a_k = \dfrac{\prod_{l=k+1}^{m-1} (2l - 1)}{\prod_{l=k}^{m-1} (2l)}`
    * - :math:`2m + 1 \geqslant 3`
      - :math:`z \sum\limits_{k=0}^{m-1} a_k \left(\frac{r}{\rho}\right)^{2k+1},
        \quad
        a_k = \dfrac{\prod_{l=k+1}^{m-1} (2l)}{\prod_{l=k}^{m-1} (2l + 1)}`

(The general expression assume the usual convention that an empty product
equals 1, and an empty sum equals 0.) A simple recurrence relation exists for
:math:`n \ne 0`:

.. math::
    F_{n+2}(r, z) = \frac{1}{n} z \left(\frac{r}{\rho}\right)^n +
                    \frac{n - 1}{n} F_n(r, z).


The integration limits have the form

.. math::
    z_R = \begin{cases}
        \sqrt{R^2 - r^2}, & r < R, \\
        0 & \text{otherwise}
    \end{cases}

and are :math:`[z_{R-1}, z_R]` for the inner part :math:`b_R\big(\rho \in [R -
1, R]\big) = \rho - (R - 1)` and :math:`[z_R, z_{R+1}]` for the outer part
:math:`b_R\big(\rho \in [R, R + 1]\big) = (R + 1) - \rho`:

.. plot:: transform_methods/rbasex-limits.py
    :align: center

The :math:`\rho` values corresponding to the integration limits (for
substitution in the antiderivatives :math:`F_n`) have an even simpler form:

.. math::
    \rho|_{z=z_R} = \sqrt{r^2 + z_R^2} = \max(r, R),

and hence

.. math::
    \left.\left(\frac{r}{\rho}\right)\right|_{z=z_R} =
        \min\left(\frac{r}{R}, 1\right).

The :math:`\arctan\frac{z_R}{r}` terms can also be “simplified” to
:math:`\left.\arccos\frac{r}{\rho}\right|_{z=z_R} = \arccos\frac{r}{R}` for
:math:`r < R` and 0 otherwise, or :math:`\arccos\left[\min\left(\frac{r}{R},
1\right)\right]`. This seems to be more computationally efficient on modern
systems, although previously it was the other way around, since :math:`\arccos`
was implemented in libraries through :math:`\operatorname{arctan2}` (FPATAN),
square root (FSQRT) and arithmetic operations.

Collecting all the pieces together, we get the following expression for the
radial parts of the projections:

.. math::
    \begin{aligned}
        p_{R;n}(r) &= 4 [r F_{n-1}(r, z_R) - R F_n(r, z_R)] - {} \\
                   &- 2 [r F_{n-1}(r, z_{R-1}) - (R - 1) F_n(r, z_{R-1})] - {} \\
                   &- 2 [r F_{n-1}(r, z_{R+1}) - (R + 1) F_n(r, z_{R+1})].
    \end{aligned}

Like :math:`b_0(\rho)`, the :math:`p_{0;n}(r)` functions do not have the inner
part, so for them (:math:`R = 0`, :math:`z_R = 0`, :math:`R + 1 = 1`) the expression is

.. math::
    \begin{aligned}
        p_{0;n}(r) &= 2 [r F_{n-1}(r, 0) - F_n(r, 0)] -
                     2 [r F_{n-1}(r, z_1) - F_n(r, z_1)] = \\
                   &= 2 [F_n(r, z_1) - F_n(r, 0)] -
                      2 r [F_{n-1}(r, z_1) - F_{n-1}(r, 0)].
    \end{aligned}

However, in practice :math:`R = 0` corresponds to the single central pixel, and
at the integer grid we have :math:`p_{0;0}(r) = \delta_{r,0}` and
:math:`p_{0;n>0}(r) = 0`, that is the intensity at :math:`r = 0` must be
assumed isotropic.

Here are examples of :math:`p_{R;n}(r)` plotted for :math:`R = 6` and :math:`n
= 0, 1, 2`, together with the radial part of the distribution basis function
:math:`b_R(r)`:

.. plot:: transform_methods/rbasex-pRn.py
    :align: center

The projection functions have a large curvature near :math:`r \approx R` and
thus are not well represented by piecewise linear approximations at the integer
grid, as illustrated below (the solid red line is the same :math:`p_{6;2}(r)`
as above):

.. plot:: transform_methods/rbasex-peak.py
    :align: center

This was not a problem for the reconstruction method developed in [1]_, since
it samples these functions at each pixel, with their :math:`r = \sqrt{x^2 +
y^2}` values not limited to integers. But expanding piecewise linear radial
distributions over the basis of these curved :math:`p_{R;n}` might be
problematic. However, as the green curves illustrate, even for a peak with just
3 nonzero points, its projection is represented by linear segments
significantly better. Therefore, for real experimental data with adequate
sampling (peak widths > 2 pixels), the piecewise linear approximation should
work reasonably well.


Transform
---------

The initial 3D distribution has the form

.. math::
    I(\rho, \theta') = \sum_n I_n(\rho) \cos^n \theta',

where :math:`I_n(\rho)` are the radial distributions for each angular order.
They are represented as a linear combination of the radial basis functions:

.. math::
    I_n(\rho) = \sum_R c_{R,n} b_R(\rho).

The forward Abel transform of this 3D distribution (in other words, its
projection, or the experimentally recorded image) then has the form

.. math::
    P(r, \theta) = \sum_n P_n(\rho) \cos^n \theta,

where :math:`P_n(r)` are its radial distributions for each angular order (not
to be confused with Legendre polynomials) and are represented as linear
combinations of the radial projected basis functions:

.. math::
    P_n(r) = \sum_R c_{R,n} p_{R;n}(r)

with the same coefficients :math:`c_{R,n}`.

If the radial distributions of both the initial distribution and its projection
are sampled at integer radii, these linear combinations can be written in
vector-matrix notation as

.. math::
    \begin{aligned}
        I_n(\boldsymbol \rho) &= \mathbf B^{\rm T} \mathbf c_n,
        & \mathbf B_{ij} &= b_{R=i}(\rho = j), \\
        P_n(\mathbf r) &= \mathbf P_n^{\rm T} \mathbf c_n,
        & (\mathbf P_n)_{ij} &= p_{R=i;n}(r = j)
    \end{aligned}

for each angular order :math:`n`.

It is obvious from the definition of :math:`b_R(\rho)` that :math:`\mathbf B`
is an identity matrix, so the expansion coefficients are simply :math:`\mathbf
c_n = I_n(\boldsymbol \rho)`. Thus the forward and inverse Abel transforms can
be computed as

.. math::
    \begin{aligned}
        P_n(\mathbf r) &= \mathbf P_n^{\rm T} I_n(\boldsymbol \rho), \\
        I_n(\boldsymbol \rho) &= \big(\mathbf P_n^{\rm T}\big)^{-1} P_n(\mathbf r)
    \end{aligned}

for each angular order separately. Since all projected basis functions satisfy
:math:`p_{R;n}(r \geqslant R + 1) = 0` (see the plots above), the matrices
:math:`\mathbf P_n^{\rm T}` are upper triangular, and their inversions
:math:`\big(\mathbf P_n^{\rm T}\big)^{-1}` are also upper triangular for all
:math:`n`, which additionally facilitates the computations. (This triangularity
makes the inverse Abel transform similar to the “onion peeling” procedure
written in a matrix form, but based on linear interpolation instead of midpoint
rectangular approximation.)

Overall, the transforms proceed as follows:

1. Radial distributions for each angular order are extracted from the input
   data using :class:`abel.tools.vmi.Distributions`. This takes
   :math:`O(N\,R_\text{max}^2)` time, where :math:`N` is the number of angular
   terms, and :math:`R_\text{max}` is the largest analyzed radius (assuming
   :math:`N \ll R_\text{max}`).
2. Radial projected basis functions are computed to construct the
   :math:`\mathbf P_n` matrices, also in :math:`O(N\,R_\text{max}^2)` total
   time.
3. For the inverse Abel transform, the :math:`\mathbf P_n^{\rm T}` matrices are
   inverted, in :math:`O(N\,R_\text{max}^3)` total time. This step is not
   needed for the forward Abel transform.
4. The radial distributions from step 1 are multiplied by the transform
   matrices :math:`\mathbf P_n^{\rm T}` or :math:`\big(\mathbf P_n^{\rm
   T}\big)^{-1}` to obtain the reconstructed radial distributions, in
   :math:`O(N\,R_\text{max}^2)` total time.
5. If the transformed image is needed, it is constructed from its radial
   distributions obtained in step 4 using the first formula in this section.
   This takes :math:`O(N\,R_\text{max}^2)` time.

That is, only step 3 has time complexity that scales cubically with the image
size, and all other steps have quadratic complexity. However, for the forward
Abel transform, step 3 is not needed at all, and for the inverse Abel
transform, its results can be cached. Thus processing a sequence of images
takes time linearly proportional to the total number of processed pixels. In
other words, the throughput is independent on the image size.


Regularizations
---------------

The matrix equation

.. math::
    \mathbf y = \mathbf A \mathbf x

(in our case the vector :math:`\mathbf x` represents the radial part of the
sought 3D distribution, the matrix :math:`\mathbf A` represents the forward
Abel transform, and the vector :math:`\mathbf y` represents the radial part of
the recorded projection) can be solved as

.. math::
    \mathbf x = \mathbf A^{-1} \mathbf y

if :math:`\mathbf A` is invertible. However, if the problem is ill-conditioned,
computing :math:`\mathbf A^{-1}` might be problematic, and the solution might
have undesirably amplified noise.

Regularization methods try to replace the ill-conditioned problem with a
related better-conditioned one and use its solution as a well-behaved
approximation to the solution of the original problem.


Tikhonov
^^^^^^^^

Instead of inverting :math:`\mathbf A` explicitly, the solution of
:math:`\mathbf y = \mathbf A \mathbf x` can be found as

.. math::
    \mathbf x = \mathop{\rm arg\,min}\limits_{\mathbf x}
                (\mathbf y - \mathbf A \mathbf x)^2,

from a quadratic minimization (“least-squares”) problem, which is equivalent to
the original problem, but makes evident that for ill-conditioned problems the
minimum is very “flat”, and many different :math:`\mathbf x` can be accepted as
a solution.

The idea of `Tikhonov regularization
<https://en.wikipedia.org/wiki/Tikhonov_regularization>`_ is to add some small
“regularizing” term to this minimization problem:

.. math::
    \tilde{\mathbf x} = \mathop{\rm arg\,min}\limits_{\mathbf x} \left[
                            (\mathbf y - \mathbf A \mathbf x)^2 + g[\mathbf x]
                        \right]

that will help to select the “best” solution by imposing larger penalty on
undesirable solutions. If this term is also a quadratic form

.. math::
    g[\mathbf x] = (\mathbf \Gamma \mathbf x)^2

with some matrix :math:`\mathbf \Gamma` (not necessarily square), then the
quadratic minimization problem is reduced back to a linear matrix equation and
has the solution

.. math::
    \tilde{\mathbf x} = \mathbf A^{\rm T}
                        \left(\mathbf A \mathbf A^{\rm T} +
                              \mathbf \Gamma \mathbf \Gamma^{\rm T}\right)^{-1}
                        \mathbf y.

In practice, it is convenient to define :math:`\mathbf \Gamma = \varepsilon
\mathbf \Gamma_0` with some fixed :math:`\mathbf \Gamma_0` and change the
“Tikhonov factor” :math:`\varepsilon` to adjust the regularization “strength”.
The form of :math:`\mathbf \Gamma_0` selects the regularization type:


:math:`L_2` norm
""""""""""""""""

This is the simplest case, with :math:`\mathbf \Gamma_0 = \mathbf I`, the
identity matrix. That is, the penalty functional :math:`g[\mathbf x] =
\varepsilon^2 \mathbf x^2` is the quadratic norm of the solution scaled by the
regularization parameter.

The idea is that, in a continuous limit, if we have a well-behaved function
:math:`f(r)` and some random noise :math:`\delta(r)`, then

.. math::
    \begin{aligned}
        g[f(r) + \delta(r)] &=
            \int [f(r) + \delta(r)]^2 \,dr = \\
            &= \underbrace{\int [f(r)]^2 \,dr}_{\textstyle g[f(r)]} +
               \underbrace{2 \int f(r) \delta(r) \,dr}_{\textstyle \approx 0} +
               \underbrace{\int [\delta(r)]^2 \,dr}_{\textstyle > 0}.
    \end{aligned}

In other words, a noisy solution will have a larger penalty than a smooth
solution, unless the noise is correlated, and a smooth solution will be
preferred as long as the noise forward transforms is close to zero
(:math:`\|\mathcal A \delta(r)\| < \|\delta(r)\|`).

Notice, however, that for very large Tikhonov factors the regularization term
starts to dominate in the minimization problem, which tend to

.. math::
    \tilde{\mathbf x} = \varepsilon^2
                        \mathop{\rm arg\,min}\limits_{\mathbf x} \mathbf x^2

and thus has the solution :math:`\tilde{\mathbf x} \to 0`. For reasonable
regularization strengths this intensity suppression effect is small, but the
solution is always biased towards zero.


Finite differences
""""""""""""""""""

Here the first-order finite difference operator is used as the Tikhonov matrix:

.. math::
    \mathbf \Gamma_0 = \begin{pmatrix}
        -1 &  1 &  0 & 0 & \cdots \\
         0 & -1 &  1 & 0 & \cdots \\
         0 &  0 & -1 & 1 & \ddots \\
        \vdots & \vdots & \ddots & \ddots & \ddots
    \end{pmatrix}.

It is the discrete analog of the differentiation operator, so in a continuous
limit this regularization corresponds to using the penalty functional of the
form

.. math::
    g[f(r)] = \int \left(\frac{df(r)}{dr}\right)^2 \,dr.

Noisy functions obviously have larger RMS derivatives that smooth functions and
thus are penalized more.

Unlike the :math:`L_2`-norm regularization, which tends to avoid sign-changing
functions and oscillating functions in general, this regularization can produce
noticeably overshoots (including negative) around sharp features in the
distribution. However, it tends to preserve the overall intensity.


Truncated SVD
^^^^^^^^^^^^^

This is the method used in pBasex. The idea is that since the `condition number
<https://en.wikipedia.org/wiki/Condition_number>`_ of a matrix equals the ratio
of its maximal and minimal `singular values
<https://en.wikipedia.org/wiki/Singular_value>`_, performing the singular value
decomposition (SVD),

.. math::
    \mathbf U \mathbf \Sigma \mathbf V^{\rm T} = \mathbf A,

inverting :math:`\mathbf \Sigma` (which is diagonal), then excluding its
largest values values and assembling the pseudoinverse

.. math::
    \tilde{\mathbf A}^{-1} = \mathbf V \tilde{\mathbf \Sigma}^{-1}
                             \mathbf U^{\rm T}

gives a better-conditioned matrix approximation of :math:`\mathbf A^{-1}`,
which is then used to obtain the approximate solution

.. math::
    \tilde{\mathbf x} = \tilde{\mathbf A}^{-1} \mathbf y.

This approach can be helpful when the left singular vectors (columns of
:math:`\mathbf V`, which become linear contributions to :math:`\mathbf x`) are
physically meaningful and different for the useful signal and the undesirable
noise. Then removing the singular values corresponding to the undesirable
vectors excludes them from the solution, while retaining the useful
contributions. However, this is not the case for our problem. Here are the
singular values :math:`\sigma_i` of :math:`\mathbf A^{-1}` plotted together
with some representative left singular vectors :math:`\mathbf v_i`:

.. plot:: transform_methods/rbasex-SVD.py
    :align: center

As can be seen, all these vectors are oscillatory, with negative values, and
most of them are delocalized over the whole radial range. That is, they do not
have a clear physical meaning for practical applications of the Abel transform.

The only potentially useful observation is that the first vectors,
corresponding to the largest singular values, have the highest spacial
frequencies and contribute mostly to the lower :math:`r` range. Thus excluding
them might reduce the high-frequency noise near the center of the transformed
image. It should be noted, however, that a simple SVD truncation leads to the
same problems with delocalized oscillations and the `Gibbs phenomenon
<https://en.wikipedia.org/wiki/Gibbs_phenomenon>`_, as in truncated Fourier
series. (From this perspective, soft attenuation, like in the Tikhonov
regularization, is a more appropriate approach.)

So this method is not recommended for practical applications and is provided
here mostly for completeness.


Non-negative components
^^^^^^^^^^^^^^^^^^^^^^^

This is the simplest *nonlinear* regularization method proposed in [1]_. The
idea is that the linear matrix equation

.. math::
    \mathbf y = \mathbf A \mathbf x

is replaced by the minimization problem

.. math::
    \tilde{\mathbf x} = \mathop{\rm arg\,min}\limits_{\mathbf x\ \geqslant\ 0}
                        (\mathbf y - \mathbf A \mathbf x)^2

with a physically meaningful constraint that the solution (the intensity
distribution) must be non-negative everywhere. If the linear solution happens
to be non-negative, this modified problem has exactly the same solution.
Otherwise the minimization problem gives the closest (in the least-squares
sense) non-negative approximation to the original problem.

Unfortunately, applying non-negativity constraint to trigonometric polynomials,

.. math::
    I(\theta) = \sum a_n \cos^n \theta \geqslant 0\ \text{for all}\ \theta,

generally leads to a system of nonlinear equations on their coefficients, which
cannot be solved efficiently.

However, if the polynomial has no more that two terms, that is its order is 0,
1, or 2 with even powers only, the constraints are linear and can be linearly
transformed into nonnegativity constraints on the coefficients:

.. math::
    \begin{aligned}
        I(\theta) &= c_0 \cos^0 \theta \geqslant 0
        &&\Leftrightarrow& c_0 \geqslant 0; \\
        I(\theta) &= c_0 \cos^0 \theta + c_1 \cos^1 \theta = \\
                  &= a_0 (\cos^0 \theta + \cos^1 \theta) + {} \\
                  &+ a_1 (\cos^0 \theta - \cos^1 \theta) \geqslant 0
        &&\Leftrightarrow& a_i \geqslant 0; \\
        I(\theta) &= c_0 \cos^0 \theta + c_2 \cos^2 \theta = \\
                  &= a_0 (\cos^0 \theta - \cos^2 \theta) + {} \\
                  &+ a_2 \cos^2 \theta \geqslant 0
        &&\Leftrightarrow& a_i \geqslant 0.
    \end{aligned}

Notice that in the last case the term :math:`a_0 (\cos^0 \theta - \cos^2
\theta) = a_0 \sin^2 \theta` corresponds to perpendicular transitions, whereas
:math:`a_2 \cos^2 \theta` corresponds to parallel transitions, so the
inequalities :math:`a_i \geqslant 0` have a direct physical meaning that both
transition components must be non-negative.

The quadratic minimization problem with linear constraints reduces to a
sequence of linear problems and is soluble exactly in a finite number of steps.

In some cases the non-negative solution can be positively biased, since it does
not allow negative noise, but can have some positive noise. Nevertheless, this
bias is smaller than the positive bias introduced by zeroing negative values in
solutions obtained by linear methods (*never do this*!).


The idea of non-negative transition components can be extended to multiphoton
processes *without interference between different channels*, so that

.. math::
    \begin{aligned}
        I(\theta) &=
            \left(a_0^{(1)} \sin^2 \theta + a_2^{(1)} \cos^2 \theta\right) \times \\
        &\times \left(a_0^{(2)} \sin^2 \theta + a_2^{(2)} \cos^2 \theta\right) \times \\
        &\times \dots \times \\
        &\times \left(a_0^{(m)} \sin^2 \theta + a_2^{(m)} \cos^2 \theta\right),
            \quad a_i^{(j)} \geqslant 0,
    \end{aligned}

which also leads to a linear combination with non-negative coefficients:

.. math::
    I(\theta) = \sum b_n \sin^m \theta \cdot \cos^n \theta,
    \quad b_n \geqslant 0.

These constraints, however, are stronger than the intensity non-negativity: for
example, the angular distribution

.. math::
    \sin^4 \theta - 2 \sin^2 \theta \cdot \cos^2 \theta + \cos^4 \theta =
    \left(\sin^2 \theta - \cos^2 \theta\right)^2

is non-negative everywhere, but contains a negative coefficient for the
:math:`\sin^2 \theta \cdot \cos^2 \theta` term. So even though this
regularization is not always valid for multiphoton processes, it can be useful
in some cases and is easy to implement. To remind that it is not “truly
non-negative”, this regularization is called “positive” here.


A general advice applicable to all regularization methods is that when a
relevant model is available, it is better to fit it directly to non-regularized
results, thus avoiding additional assumptions and biases introduced by
regularizations.


Examples
^^^^^^^^

.. warning::
    Absolute and relative efficiencies of these regularization methods and
    their optimal parameters depend on the image size, the amount of noise and
    the distribution itself. Therefore *do not assume* that the examples shown
    here are directly relevant to *your* data.

Some properties of the regularization methods described above are demonstrated
here by applying them to a synthetic example. The test distribution from the
BASEX article is forward Abel-transformed to obtain its projection, and then
Poissonian noise is added to it to simulate experimental VMI data with
relatively low signal levels (such that the noise is prominent):

.. comment
    the only purpose of ":scale: 1" in the plots below is to make them clickable

.. plot:: transform_methods/rbasex-sim.py
    :scale: 1

In order to characterize the regularization performance, all the methods are
applied at various strengths to this simulated projection, and the relative
root-mean-square error :math:`\big\|\tilde I(r) - I_\text{src}(r)\big\| \big/
\big\|I_\text{src}(r)\big\|`, where :math:`I_\text{src}(r)` is the “true”
radial intensity distribution, and :math:`\tilde I(r)` is the reconstructed
distribution, is calculated in each case. The following plot shows how this
reconstruction error changes with the regularization strength (the
non-parameterized “pos” method is shown by a dashed line):

.. plot:: transform_methods/rbasex-regRMSE.py
    :align: center

(Note that the horizontal axis in the left plot is nonlinear, and that the
vertical axis in the right plot does not start at zero and actually spans a
very small range.)

These plots demonstrate that the Tikhonov methods have some optimal strength
value, at which the reconstruction error is minimized. At smaller values the
noise is not suppressed enough (zero strength corresponds to the
non-regularized transform), and at larger values the reconstructed distribution
is smoothed too much.

The SVD plot has steps corresponding to successive removal of singular values.
The reconstruction error does not decrease monotonically, but exhibits several
local minima before starting to grow. Notice that even the global minimum is
only slightly better than no regularization.

The actual reconstructed images for each regularization method at its optimal
strength are shown below with their radial intensity distributions:


Using ``reg=None``
""""""""""""""""""

.. plot::
    :scale: 1

    from rbasex_reg import plot
    plot(None)

The non-regularized transform results are shows as a reference. The image has
red colors for positive intensities and blue colors for negative intensities.
The upper plot shows the reconstructed radial intensity distribution in black
and the “true” distribution in red behind it. The lower plot shows the the
difference between these two distributions in blue (red is the zero line).


Using ``reg=('L2', 100)``
"""""""""""""""""""""""""

.. plot::
    :scale: 1

    from rbasex_reg import plot
    plot('L2', 100)

The noise level is generally reduced, but the peaks near the origin are
noticeably broadened, which actually increases deviations in this region.


Using ``reg=('diff', 115)``
"""""""""""""""""""""""""""

.. plot::
    :scale: 1

    from rbasex_reg import plot
    plot('diff', 115)

The noise is reduced even more, especially its high-frequency components. The
peaks near the origin also suffer, but somewhat differently.


Using ``reg=('SVD', 0.03)``
"""""""""""""""""""""""""""

.. plot::
    :scale: 1

    from rbasex_reg import plot
    plot('SVD', 0.03)

The only noticeable difference from no regularization is some noise reduction
near the origin.


Using ``reg='pos'``
"""""""""""""""""""

.. plot::
    :scale: 1

    from rbasex_reg import plot
    plot('pos')

The most prominent feature is the absence of negative intensities. The noise is
reduced significantly in the areas of low intensity, where it is constrained
from attaining negative values, which also reduces its positive amplitudes, as
the distribution should be reproduced on average. The peaks, being strongly
positive, do not have noticeable noise reduction. However, in contrast to other
methods, the peaks near the origin are not broadened, while the off-peak noise
near them is reduced.


References
----------

.. [1] \ M. Ryazanov,
       “Development and implementation of methods for sliced velocity map
       imaging. Studies of overtone-induced dissociation and isomerization
       dynamics of hydroxymethyl radical (CH\ :sub:`2`\ OH and
       CD\ :sub:`2`\ OH)”,
       Ph.D. dissertation, University of Southern California, 2012.
       (`ProQuest <https://search.proquest.com/docview/1289069738>`_,
       `USC <http://digitallibrary.usc.edu/cdm/ref/collection/p15799coll3/id/
       112619>`_).
