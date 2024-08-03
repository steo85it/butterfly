# Exact Covariance Test

``` 
zsh exact_cov_test.sh mesh.obj
```

Computes the **full dense** eigendcomposition of the linear FEM Laplace-Beltrami
operator. 

For various Matern covariance parameter pairs $(\kappa, \nu)$:
- Computes the true and estimated truncation errors for a increasing numbers of
computed eigenvectors $m$.
- Plots truncation errors and sample from random field.

The tested parameter pairs are specified in `exact_cov_test.sh` by
```
# KAPPA, NU pairs to test
PARAMS=(1e-4,0.0 1e-1,4.0 3e-1,0.5)
```

# Fast Covariance Test

``` 
zsh fast_cov_test.sh mesh.obj
```

For various Matern covariance parameter pairs $(\kappa, \nu)$:
- For various tolerances $\varepsilon$:
    - Computes the butterfly-compressed eigendecomposition of the linear FEM
Laplace-Beltrami operator stopping when the estimated truncation error in the
covariance operator is less than $\varepsilon$.
    - Computes samples and matvecs with the covariance operator.
    - Estimates covariance operator error compared to reference butterfly
      tolerance $\varepsilon_{\text{ref}}$ from matvecs.
    - Plots sample from random field.
- For various Chebyshev polynomial orders $p$:
    - Computes matrix-free samples and matvecs with the order $p$ approximate
      covariance operator using [1]
    - Estimates covariance operator error compared to reference butterfly
      tolerance $\varepsilon_{\text{ref}}$ from matvecs.
    - Plots sample from random field.
- Plots covariance operator errors for both schemes.

The tested parameter pairs, number of computed samples and matvecs, butterfly
tolerances $\varepsilon$, and log Chebyshev orders $\log_2(p)$ are specified in `fast_cov_test.sh` by
```
# KAPPA, NU pairs to test
PARAMS=(1e-4,0.0 1e-1,4.0 3e-1,0.5)

NUM_SAMPLES=100

REFTOL=1e-4
TOLS=(1e-1 1e-2 1e-3 $REFTOL)
LOGPS=(2 4 6)
```



> [1] Lang, Annika, and Mike Pereira. "Galerkin–Chebyshev approximation of
> Gaussian random fields on compact Riemannian manifolds." BIT Numerical
> Mathematics 63, no. 4 (2023): 51.