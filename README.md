# Master Thesis
> Allows reproduction of results in my master thesis.

[![Downloads][pypi-downloads]][pypi-url]
![License][pypi-license]
![Python versions][pypi-python-versions]

The package supports testing and evaluating SSD and Bayesian SSD. The results
can be visualised.

## Installation

```sh
pip install twomartens.masterthesis
pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI
```

The second line is important as Git dependencies cannot be specified in the `setup.py`
file.

Please refer to [GPU support][tf-gpu-support] for instructions on
installing the non-Python dependencies for `tensorflow`.

Type the following to create the configuration file and to see the options:
```sh
tm-masterthesis config list
```

Especially the paths have to be set to the correct values.

## Usage example

```sh
tm-masterthesis --help
```

Lists all available commands. As most commands are nested, it is advisable to 
request the help at different nesting levels.

```sh
tm-masterthesis config {get,set,list}
```

Allows for the modification and retrieval of the configuration values.

```sh
tm-masterthesis test {ssd,bayesian_ssd} iteration train_iteration
```

Tests the selected network, using `iteration` as identifier for the test run
and `train_iteration` as identifier for the training iteration. If the config
parameter `ssd_test_pretrained` is `True` then the training iteration is
not relevant. 

```sh
tm-masterthesis evaluate {ssd,bayesian_ssd} iteration
```

Runs the evaluation process using the test results identified by `iteration`,
evaluation results are saved under `iteration` under the evaluation path.

```sh
tm-masterthesis visualise_metrics {ssd,bayesian_ssd} iteration
```

Uses the evaluation results stored under `iteration` and visualises
it. The score JSON and the figure images are stored under `iteration`
in a `visualise` folder under the output path.

There are more commands but the rest can be very tightly linked to requirements
in the master thesis and might therefore not be of interest generally.

## Development setup

Clone the repository locally. Then execute the following commands inside
the repository:

```sh
git submodule init
git submodule update
pip install -e .
pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI
```

## Release History

* 0.1.0
    * first release

## Meta

Jim Martens – [@2martens](https://twitter.com/2martens) – github@2martens.de

Distributed under the Apache 2.0 license. See ``LICENSE`` for more information.
The package contains the [ssd_keras][ssd_keras] implementation of Pierluigi Ferrari.

[https://github.com/2martens/](https://github.com/2martens/)

## Contributing

1. Fork it (<https://github.com/2martens/masterthesis/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[dependencies]:https://img.shields.io/librariesio/release/pypi/twomartens.masterthesis.svg
[pypi-license]: https://img.shields.io/pypi/l/twomartens.masterthesis.svg
[pypi-url]: https://pypi.org/project/twomartens.masterthesis/
[pypi-downloads]: https://img.shields.io/pypi/dm/twomartens.masterthesis.svg
[pypi-python-versions]: https://img.shields.io/pypi/pyversions/twomartens.masterthesis.svg
[tf-gpu-support]: https://www.tensorflow.org/install/gpu
[ssd_keras]: https://github.com/pierluigiferrari/ssd_keras
