# Quantum Circuit Overhead
Numerical package for computation of $\delta$-approximate $t$-designs used in the article [Quantum Circuit Overhead](https://arxiv.org/abs/2505.00683) by O. Słowik, P. Dulian and A. Sawicki.
## Usage
### Sampling $\delta$
To sample $\delta$ for random gate sets write in the console:
```console
> python main.py [arguments]
```
where arguments specify the ensemble. For $N$ gate sets, weights from $t$-design and typical ensembles these arguments are:
- Haar-random gate sets with $n$ elemnts from $U(2)$ (in article denoted by $\mathcal S_{\mu, n, \infty}$):
```console
> python main.py -n_of_generators n -d 2 -t t -sample_size N
```
- Haar-random gate sets with $n$ elemnts from $U(2)$ with order $r$ (in article denoted by $\mathcal S_{\mu, n, r}$):
```console
> python main.py -n_of_generators n -d 2 -t t -sample_size N -gate_order r
```
- gates derived from a finite group $C$ stored in the text file `file.txt`  with order $r$ (in article denoted by $\mathcal C_{\mu, r}$):
```console
> python main.py -n_of_generators n -d 2 -t t -sample_size N -gate_order r -gates_path file.txt
```
File `file.txt` should start with the group dimension `d` and then each row is a row of the consecutive $c_i$ e.g.:   
2   
$(c_1)^1_1$ $(c_1)^1_2$   
$(c_1)^2_1$ $(c_1)^2_2$   
$(c_2)^1_1$ $(c_2)^1_2$   
$(c_2)^2_1$ $(c_2)^2_2$   
...

The program will save the result in two files one containg norms and the other containg random gates.

All possible arguments are:
- `-sample_size N` sets the number of sampled gate sets to n.
- `-n_of generators n` sets the number of independent random gates in each gate set. If set to `start-stop-step` performs computation for every n in `range(start, stop, step) = [start, start+step, ..., stop]`.
- `-d d` sets the dimension of the unitary group $U(d)$ which gate sets are sampled from.
- `-gates_path txt_file_path` sets path to the text file containing set of gates $\mathcal C$. If this argument is given the sampled gate sets will be of the form $c^\dagger U_i c$ for every $c\in\mathcal C$ and $i=1, ..., n$ instead of the default Haar-random form $U_i$.
- `-gate_order r` - sets order of the random gates to $r$.
- `-weights_gen x` sets method for generating representation to `x` which should be one of:
    - `t-design` - weghts appearing in a $t$-design where $t$ is given as `-t t`. Defult option.
    - `norm2` - $SU(d)$ weights $\lambda=(\lambda_1, ..., \lambda_{d-1})$ such that $||\lambda||_2 = \sum_i |\lambda_i|^2 \le J$ where $J$ is given as `-J J`.
    - `dim` - weights such that representation dimension is $\dim \pi_\lambda \le J$ where $J$ is given as `-J J`.

    For options different than `t-design` one can additionally filter weights by using `-weights_filter y` where `y` is:
    - `PU` - leaves only weights of projective representations i.e.: $\pi_\lambda(e^{i\varphi}U) = \pi_\lambda(U)$ for all $\varphi$.
    - `Q` - leaves only weights of quaternionic representations.
- `-save_spectrum` - will save the whole spectrum (not just the norms) of $T_{\nu_\mathcal{S}, \lambda}$ operators.
- `-symmetric` - will compute norm of symmetric gate set $U_1, U_2, ..., U_n, U_1^\dagger, U_2^\dagger, ..., U_n^\dagger$
- `-v x` - add `x` to the created file names.
### Other features
Package contains also simple classes for computation of finite groups `FiniteGroup` in `finite_groups.py` and for computation of $SU(d)$ representations `SURepresentation` in `representation.py`.
### Parallel computation
Package was written using [mpi4py](https://mpi4py.readthedocs.io/en/stable/) library and can be run on multiple cores using `mpiexec`:
```console
> mpiexec python main.py [arguments]
```