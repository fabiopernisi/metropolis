# Exploration of Non-Zero-Temperature Adaptive Monte Carlo in Neural Network Training

This repository hosts a notebook that extends the research presented in "Training neural networks using Metropolis Monte Carlo and an adaptive variant" by Whitelam et al. (2022), accessible at [arXiv:2205.07408](https://arxiv.org/abs/2205.07408).

## Background
The referenced paper investigates the application of the zero-temperature Metropolis Monte Carlo method for neural network training. This approach aims to minimize a loss function through an algorithm that initially resembles Gradient Descent. In the original version, as described by Whitelam and colleagues, the algorithm updates each network weight $$\( x_i \)$$ by adding a Gaussian random number $$\( \epsilon_i \)$$, as follows:

$$
x_i \rightarrow x_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

Here, $$\( \sigma^2 \)$$ is a uniform variance across all weights. The update is accepted only if it results in a non-increasing loss, consistent with the zero-temperature Metropolis algorithm.

## Adaptive Monte Carlo Extension
The paper introduces an adaptive version of the Monte Carlo algorithm (aMC), which proposes weight updates in a more nuanced manner:

$$
x_i \rightarrow x_i + \epsilon_i \mathcal{N}(\mu_i, \sigma_i^2)
$$

In this adaptive approach, both the mean $$\( \mu_i \)$$ and variance $$\( \sigma_i^2 \)$$ of the Gaussian distribution vary for each weight. The mean $$\( \mu_i \)$$ is adjusted after every accepted move:

$$
\mu_i \rightarrow \mu_i + \epsilon (\epsilon_i - \mu_i)
$$
$
Here, $$\( \epsilon \)$$ is a model hyperparameter. The variance $$\( \sigma_i \)$$ is adapted using a simple learning-rate schedule: after every $$\( n_S \)$$ consecutive rejections (denoted `n_reset` in the code), the following updates occur:

$$
\sigma_i \rightarrow \sigma_i \times 0.95 \quad \text{and} \quad \mu_i = 0
$$

Initially, the step-size $$\( \sigma \)$$ is set to a constant value $$\( \sigma_0 \)$$.

## Our Focus
In this project, we delve into exploring a non-zero-temperature variant of the aMC algorithm. Our aim is to understand its implications and effectiveness in neural network training under different temperature settings.

---

**Note**: This README is intended to provide a brief overview of our work in this repository. For a comprehensive understanding, please refer to the original paper and the detailed documentation within the code.
