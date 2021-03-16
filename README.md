# GPErks

A Python library to (bene)fit Gaussian Process Emulators.

# TODO: update

This library contains the tools needed for constructing a univariate Gaussian process emulator (GPE) as a surrogate model of a generic map *X -> y*. The map (e.g. a computer code input/output) is simply described by the (*N x D*) *X* matrix of input parameters and the respective (*N x 1*) *y* vector of one output feature, both provided by the user. GPEs are implemented as the sum of a mean function given by a linear regression model (with first-order degreed polynomials) and a centered (zero-mean) Gaussian process regressor with RBF kernel as covariance function.

The GPE training can be performed either against a validation set (by validation loss) or by itself (by training loss), using an "early-stopping" criterion to stop training at the point when performance on respectively validation dataset/training dataset starts to degrade. The entire training process consists in firstly performing a *K*-fold cross-validation training by validation loss, producing a set of *K* GPEs. Secondly, a final additional GPE is trained on the entire dataset by training loss, using an early-stopping patience level and maximum number of allowed epochs both equal to the average stopping epoch number previously obtained across the cross-validation splits. Each single training is performed by restarting the loss function optimization algorithm from different initial points in the hyperparameter high-dimensional space.

At each training epoch, it is possible to monitor training loss, validation loss and a metric of interest (the last two only if applicable). Available metrics are MAPE (corrected variant), MSE and R2Score. Other metrics can be easily integrated if needed. Losses over epochs plots can be automatically outputed. It is also possible to switch between GPE's noise-free and noisy implementations. Also, data can be standardized before the training.

The entire code runs on both CPU and GPU. The cross-validation training loop is implemented to run in parallel with multiprocessing.

---
## Information

**Status**: `Occasionally developed`

**Type**: `Personal project`

**Development years**: `2020-2021`

**Author**: [stelong](https://github.com/stelong)

---
## Getting Started

```
git clone https://github.com/stelong/GPErks.git
```

---
## Prerequisites

* [Python3](https://www.python.org/) (>=3.6)
* [virtualenv](https://pypi.org/project/virtualenv/) (optional)

---
## Installing

```
cd GPErks/
```
```
# (this block is optional)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```
```
pip install .
```

---
## Usage

```
cd example-scripts/
```
To run the example scripts showing the full library capabilities you need to format your dataset as plain text without commas into two files: `X.txt` and `Y.txt`. Additionally, you need to provide labels for each input parameter and output feature as plain text into two separate files: `xlabels.txt` and `ylabels.txt`. An example dataset is provided in `data/`.


### Script 1: emulation step-by-step
This first script guides you through common steps (0)-(7) to make towards a complete emulation of the map *X -> Y[:, IDX]*, from dataset loading to actual training to emulator testing. `IDX` is an integer representing the column index (counting from 0) of the selected feature to be emulated. The input dataset is automatically split such that 80% of it is used for training while the remaining 20% is used for validation/testing. 

To run the script, type:
```
python3 1_emulation_step_by_step.py /absolute/path/to/input/ IDX /absolute/path/to/output/
```
Notice that in our specific case, `/absolute/path/to/input/` is `data/`. After the run completes, folder `IDX/` will be created in `/absolute/path/to/output/` and filled with a trained emulator object `gpe.pth` and training dataset files `X_train.txt`, `y_train.txt`.

The emulator base class is `GPEmul`. An emulator object can be instantiated as follows:
```
from GPErks.gpe import GPEmul

emulator = GPEmul(X_train, y_train)
```
Additional keyword arguments with default values are:
* `device=torch.device("cuda" if torch.cuda.is_available() else "cpu")`
* `learn_noise=False`
* `scale_data=True`

By changing `learn_noise` to `True`, you can easily switch to a noisy formulation (an additional hyperparameter will be fitted, correspondig to the standard deviation of a zero-mean normal distribution). Data are automatically standardised before the training. To disable this option simply set `scale_data` to `False`.

The training is performed via the command:
```
emulator.train(X_val, y_val)
```
Additional keyword arguments with default values are:
* `learning_rate=0.1`
* `max_epochs=1000`
* `n_restarts=10`
* `patience=20`
* `savepath="./"`
* `save_losses=False`
* `watch_metric="R2Score"`

`learning_rate` is a tuning parameter for the employed *Adam* optimization algorithm that determines the step size at each iteration while moving toward a minimum of the loss function. `max_epochs` is the maximum number of allowed epochs (iterations) in the training loop. `n_restarts` is the number of times you want to repeat the optimization algorithm starting from a different point in the hyperparameter high-dimensional space. `patience` is the maximum number of epochs you want to wait without seeing any improvement on the validation loss (if called as above: `emulator.train(X_val, y_val)`) or on the training loss (if called with empty arguments: `emulator.train([], [])`). `savepath` is the absolute path where the code will store training `checkpoint.pth` files and the final trained emulator object `gpe.pth`. To output figures of monitored quantities such as training loss, validation loss and metric of interest over epochs, set `save_losses` to `True`. `watch_metric` can be set to any metric name chosen among the available ones (currently `"MAPE"`, `"MSE"`, `"R2Score"`).

Finally, the trained emulator object can be saved as:
```
emulator.save()
```
Additional keyword argument is `filename` which defaults to `"gpe.pth"`.

Once you have a trained emulator object, this can be easily loaded as:
```
emulator = GPEmul.load(X_train, y_train)
```
Additional keyword arguments are `loadpath` which defaults to the training `savepath` (default is `"./"`) and `filename` which defaults to the saving `filename` (default is `"gpe.pth"`). Notice that you need exactely the same dataset used during the training (`X_train` and `y_train`) to load a trained emulator.

The emulator (either loaded or freshly trained) can be now used to make predictions (inference) at a new (never observed) set of points. This can be performed through the `predict` command:
```
X_test, y_test = ... # load the testing dataset here
y_predicted_mean, y_predicted_std = emulator.predict(X_test)
```
The returned vectors have shape `(X_test.shape[0],)`.

To check to emulator accuracy, you can evaluate the chosen metric function at the true and predicted values:
```
from GPErks.utils.metrics import R2Score

print( R2Score(emulator.tensorize(y_test), emulator.tensorize(y_predicted_mean)) )
```
Notice that you have to first make the `numpy.ndarray` vectors be tensors because of the metric function being specifically written for `torch` tensors.

It is also possible to draw samples from the emulator full posterior distribution via the `sample` command:
```
Y_samples = emulator.sample(X_test)
```
Additional keyword argument is `n_draws`, the number of samples to draw, which defaults to `1000`. The returned matrix has shape `(X_test.shape[0], n_draws)`.


### Script 2: K-fold cross-valiadation training
This script automatically performs a K-fold cross-validation training and a final training on the entire dataset. Default number of splits is `FOLD=5` while default metric used is `METRIC="R2Score"`.

To run the script, type:
```
python3 2_kfold_cross_validation_training.py /absolute/path/to/input/ IDX /absolute/path/to/output/
```
After the run completes, folder `IDX/` will be created in `/absolute/path/to/output/`. In `/absolute/path/to/output/IDX/`, other 5 folders (numbered from 0) will be created and filled with a trained emulator object `gpe.pth` and dataset files `X_train.txt`, `y_train.txt` corresponding to the specific four fifth of the dataset the emulator has been trained on. Moreover, in `/absolute/path/to/output/`, the same three files will be produced with the only difference being the emulator this time being trained on the entire dataset. An additional file `R2Score_cv.txt` will be outputed, containing the accuracy obtained on each left-out part of the dataset during the cross-validation process.


### Script 3: GPE-based Sobol' global sensitivity analysis (GSA)
This scripts represents an additional, very useful tool to understand input parameters' impact on each model's scalar output total variance. Sobol' sensitivity indices, namely first-order, second-order and total effects are calculated for this purpose. The code uses Saltelli's and Jansen's estimators in combination with samples from the full GPE posterior distribution to get an entire distribution for each of the Sobol' indices.

Default emulator used is the one trained on the entire dataset (`EMUL_TYPE="full"`). This is useful when the training dataset is small. Other available option is to use the best-performing emulator in the cross-validation process (`EMUL_TYPE="best"`), i.e. the emulator which achieved the highest/lowest metric score when trained on a specific four fifth of the dataset and tested against the respective left-out part. The Sobol' indices estimators have cost of `M x (2*D + 2)` model evaluations, which becomes computationally tractable when, as in this case, the model is replaced by a fast-evaluating emulator. `M=1000` is the number of initial points the algorithm samples from a low-discrepancy, quasi-random Sobol' sequence. If you do not want to calculate second-order indices, this can be switched-off by setting `CALC_SECOND_ORDER=False`. In this case the total cost will be `M x (D + 2)`. The default number of GPE posterior distribution samples used is `N_DRAWS=1000`. Parameters whose indices have distributions with mean or mean minus 3 standard deviations below `THRE=0.01` are considered to be negligible (no effect).

To run the script, you first need to have run **Script 2** with the same input/output paths and same `IDX`. Then type:
```
python3 3_global_sobol_sensitivity_analysis.py /absolute/path/to/input/ IDX /absolute/path/to/output/
```
After the run completes, folder `/absolute/path/to/output/IDX/` (or `/absolute/path/to/output/IDX/BESTSPLIT_IDX/` if you used `EMUL_TYPE="best"`) will be filled with indices' distributions files `STi.txt`, `Si.txt`, `Sij.txt` and two summary plot files `*idxlabelname*_box.pdf`, `*idxlabelname*_donut.pdf`.

### Script 4: GSA parameters ranking
This last script is an add-on functionality for global sensitivity analysis. If you have performed GSA across one or many different scalar output features (i.e. different columns of the output matrix *Y*) you may want to know which parameters resulted to be the most influencing. By adding in your `/absolute/path/to/input/` dataset-containing folder (in our example case, `data/`) an extra file called `features_idx_list.txt` which contains indices of output features for which you previously performed GSA, you can rank parameters from the most important to the least important.

Ranking is performed according to first-order effects (`CRITERION="Si"`). This could have already been guessed by visually counting how many times a parameter shows up in the different `*idxlabelname*_donut.pdf` plots. If you want to rank parameters according to total effects, set `CRITERION="STi"`.

To run the script, you first need to have run **Script 3** with the same input/output paths for each `IDX` listed in your `/absolute/path/to/input/features_idx_list.txt` file. Then type:
```
python3 4_gsa_parameters_ranking.py /absolute/path/to/input/ /absolute/path/to/output/ > params_ranking.txt
```

---
## Contributing

Stefano Longobardi is the only maintainer. Any contribution is welcome.

---
## License

This project is licensed under the MIT license.
Please refer to the [LICENSE.md](LICENSE.md) file for details.

---
*This README.md complies with [this project template](
https://github.com/ShadowTemplate/project-template). Feel free to adopt it
and reuse it.*
