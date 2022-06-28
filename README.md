# Thresholded Lasso Bandit
Code for reproducing results in the paper "[Thresholded Lasso Bandit](https://arxiv.org/abs/2010.11994)".

## About
In this paper, we revisit the regret minimization problem in sparse stochastic contextual linear bandits, where feature vectors may be of large dimension $d$, but where the reward function depends on a few, say $s_0\ll d$, of these features only.
We present Thresholded Lasso bandit, an algorithm that (i) estimates the vector defining the reward function as well as its sparse support, i.e., significant feature elements, using the Lasso framework with thresholding, and (ii) selects an arm greedily according to this estimate projected on its support.
The algorithm does not require prior knowledge of the sparsity index $s_0$ and can be parameter-free.
For this simple algorithm, we establish non-asymptotic regret upper bounds scaling as $\mathcal{O}( \log d + \sqrt{T} )$ in general, and as $\mathcal{O}( \log d + \log T)$ under the so-called margin condition (a probabilistic condition on the separation of the arm rewards).
The regret of previous algorithms scales as $\mathcal{O}( \log d + \sqrt{T \log (d T)})$ and $\mathcal{O}( \log T \log d)$ in the two settings, respectively.
Through numerical experiments, we confirm that our algorithm outperforms existing methods. 

## Installation
This code is written in Python 3.
To install the required dependencies, execute the following command:
```bash
$ pip install -r requirements.txt
```

### For Docker User
Build the container:
```bash
$ docker build -t thresholded-lasso-bandit .
```
After build finished, run the container:
```bash
$ docker run -it thresholded-lasso-bandit
```

## Run Experiments
In order to investigate the performance of TH Lasso bandit on features drawn from a Gaussian distribution, execute the following command:
```bash
$ python run_gaussian_experiment.py
```
In this experiment, the following options can be specified:
* `--K`: Number of arms. The default value is `2`.
* `--T`: Number of rounds to be played. The default value is `1000`.
* `--d`: Dimension of feature vectors. The default value is `1000`.
* `--s0`: Sparsity index. The default value is `20`.
* `--x_max`: Maximum l2-norm of feature vectors. The default value is `10`.
* `--rho_sq`: Correlation level between feature vectors of arms. The default value is `0.7`.
* `--num_trial`: Number of trials to run experiments. The default value is `20`.

To evaluate TH Lasso bandit via an experiment with a feature distribution other than the Gaussian distribution (uniform, elliptical, hard instance), execute the following command:
```bash
$ python run_uniform_experiment.py
``` 
```bash
$ python run_elliptical_experiment.py
``` 
```bash
$ python run_hard_instance_experiment.py
``` 
