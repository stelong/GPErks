---
title: 'GPErks: a Python library to (bene)fit Gaussian Process Emulators'
tags:
  - Python
  - Gaussian process emulator
  - surrogate model
  - uncertainty quantification
  - global sentitivity analysis
authors:
  - name: Stefano Longobardi
    orcid: 0000-0002-0111-0047
    affiliation: 1
  - name: Gianvito Taneburgo
    affiliation: 2
affiliations:
 - name: Cardiac Electromechanics Research Group, School of Biomedical Engineering and Imaging Sciences, King's College London, London, UK
   index: 1
 - name: Independent Researcher
   index: 2
date: 30 July 2021
bibliography: paper.bib

---
# Summary
`GPErks` library does this this and that


# Statement of need
Many libraries for Gaussian process modelling exists out there [@scikit-learn:2011; @GPy:2014; @PyMC3:2016; @GPflow:2017; @PyStan:2021]. Making a Gaussian process model a useful tool in applications requires model hyperparameters fitting to observed data. Common practice to model fitting involves maximizing the marginal likelihood with respect to the hyperparameters [@Rasmussen:2005]. An optimization algorithm is commonly run to directly reach a global minimum of the negative marginal log-likelihood. However, according to the specific application in mind for the GP model, one may want to stop training at a specific moment when the model performances are satisfactory enough for inference purposes. The flexibility to train a GP model by explicitly customising the training for loop while controlling losses is offered by GPyTorch library [@GPyTorch:2019], where the user defines the model, the loss function, the parameters to be optimized, the optimization algorithm and manually makes an optimizer gradient descent step at every training epoch while every time resetting the gradients. However, with great power comes great responsibility: it is not easy to train "by hands" and the resulting trained model could have not captured the data structure correctly.

We leveraged GPyTorch capabilities by creating a powerful yet simple API around it to make the GPE model fitting and utilisation accessible to everyone while still retaining all the customisation capabilities offered by the out of the box library.

In particular, this code aims at fast and robustly emulate a computer code input/output, using a Gaussian process model: the so called Gaussian process emulator (GPE).  This code aims at simplifying computer scientists life in computer model analysis and uncertainty quantification with global sensitivity analysis that comes in for free when the model is replaced by a fast evaluating probabilistic surrogate.

# Model description
GPEs $f(\mathbf{x})$ were defined as the sum of a deterministic mean function $h(\mathbf{x})$ and a stochastic process $g(\mathbf{x})$ [@OHagan:2006]:
\begin{equation}
    f(\mathbf{x}) = h(\mathbf{x}) + g(\mathbf{x})
\end{equation}

The mean function is a linear regression model with first-order degree polynomials:
\begin{equation}
    h(\mathbf{x}) := \beta_0 + \beta_1 x_1 + \dots + \beta_{d} x_{d}
\end{equation}

where $\beta_i\in\mathbb{R}\,\,\text{for}\,\,i=0,\dots,d$ are the weights, and $d\in\mathbb{N}$ represents the problem input dimension. The stochastic process is a centred (zero-mean) Gaussian process with the stationary squared exponential kernel as covariance function:
\begin{align}
    &g(\mathbf{x})\sim\mathcal{GP}(\mathbf{0},\,k_{\text{SE}}(d(\mathbf{x},\,\mathbf{x}'))) \\
    &k_{\text{SE}}(d(\mathbf{x},\,\mathbf{x}')) := \sigma_f^2\, e^{-\frac{1}{2}\,d(\mathbf{x},\,\mathbf{x}')} \\
    &d(\mathbf{x},\,\mathbf{x}') := (\mathbf{x}-\mathbf{x}')^\mathsf{T}\,\Lambda\,(\mathbf{x}-\mathbf{x}')
\end{align}

where $\sigma_f^2\in\mathbb{R}^{+}$ is the signal variance and $\Lambda:=\text{diag}(\ell_1^2,\dots,\ell_D^2)$, $\ell_i\in\mathbb{R^{+}}$ for $i=1,\dots,d$ are the characteristic length-scales of the process. The model likelihood was taken to be Gaussian, i.e. the learning sample observations ($y$) are modelled to be affected by an additive, independent and identically distributed noise:
\begin{equation}
    y=f(\mathbf{x}) + \varepsilon,\quad \varepsilon\sim\mathcal{N}(0,\,\sigma_n^2)
\end{equation}

where $\sigma_n^2\in\mathbb{R}^{+}$ is the noise variance. The mean function and kernel described here are used as defaults within our emulation framework, although they can both be changed to any other different formulation offered by GPyTorch

<!-- 
# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

We want to say thank you to...

# References