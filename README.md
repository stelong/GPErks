<img src="notebooks/data/images/GPErks_logo.png" width=120 height=120 />

# GPErks

A Python library to (bene)fit Gaussian Process Emulators.

---
## Information

**Status**: `Actively developed`

**Type**: `Personal project`

**Development years**: `2020 - Present`

**Authors**: [stelong](https://github.com/stelong), [ShadowTemplate](https://github.com/ShadowTemplate)

---
## Getting Started

### Prerequisites

* [Python3](https://www.python.org/) (>=3.10)
* [virtualenv](https://pypi.org/project/virtualenv/) (optional)

### Installing

1. Pull the source code from the project repository:
```
git clone https://github.com/stelong/GPErks.git
cd GPErks/
```
2. (optional) Create a Python3 virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```
3. Install PyTorch package first in order to satisfy your custom installation requirements (e.g., a CPU-only installation or a CUDA installation with a specific version that matches your machine NVIDIA drivers):

(CPU-only)
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
(CUDA 11.8)
```
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
Note: please check PyTorch [website](https://pytorch.org/get-started/locally/) to customize your installation.

4. Install GPErks:
```
pip install .
```

5. (optional) Install Jupyter Notebook to be able to run the example notebooks:
```
pip install notebook
```

### Usage

The full documentation is under construction. For the moment, please refer to the example [notebooks](https://github.com/stelong/GPErks/tree/master/notebooks) and [tutorial](https://youtu.be/e4kYIIrcAHA) while noting that this is based on the old GPErks v0.1.0 API (it will provide a good overview on all package functionalities). The available notebooks are also available as plain Python [scripts](https://github.com/stelong/GPErks/tree/master/examples).

---
## Contributing

[stelong](https://github.com/stelong) and [ShadowTemplate](https://github.com/ShadowTemplate) are the only maintainers. Any contribution is welcome!

---
## License

This project is licensed under the MIT license.
Please refer to the [LICENSE](LICENSE) file for more details.

---
*This README.md complies with [this project template](
https://github.com/ShadowTemplate/project-template). Feel free to adopt it
and reuse it.*