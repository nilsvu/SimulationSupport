# SimulationSupport

Python routines that are shared between SpEC and SpECTRE, such as eccentricity
control.

## Getting started

Install the package (and development extras) into a virtual environment:

```bash
python -m pip install .[dev]
```

## Documentation

Build the Sphinx documentation locally with the docs extra:

```bash
python -m pip install .[docs]
sphinx-build -b html docs docs/_build/html
```

## Formatting and testing

The project uses `black` and `isort` for formatting. Run the checks locally
together with the tests:

```bash
black .
isort .
pytest
```

Continuous integration via GitHub Actions enforces these checks on pushes and
pull requests.
