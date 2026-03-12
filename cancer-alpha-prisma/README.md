# Cancer Alpha PRISMA Figure Generation

This project provides tools for generating a PRISMA figure for the Cancer Alpha data preprocessing pipeline. The PRISMA figure visually represents the flow of data through the preprocessing steps, ensuring transparency and reproducibility in the data handling process.

## Overview

The Cancer Alpha PRISMA figure generation project includes the following components:

- **src/prisma_figure**: The main module containing the logic for generating the PRISMA figure.
  - `__init__.py`: Initializes the prisma_figure module.
  - `cli.py`: Command-line interface for generating the PRISMA figure.
  - `parser.py`: Responsible for parsing input data related to the Cancer Alpha data preprocessing pipeline.
  - `renderer.py`: Handles the rendering of the PRISMA figure.

- **notebooks**: Contains Jupyter notebooks for interactive figure generation.
  - `01_prisma_figure_generation.ipynb`: An interactive notebook demonstrating the figure generation process.

- **tests**: Contains unit tests to ensure the functionality of the rendering logic.
  - `test_renderer.py`: Unit tests for the functions and classes defined in `renderer.py`.

- **pyproject.toml**: Configuration file specifying project metadata, dependencies, and build system requirements.

- **requirements.txt**: Lists the Python packages required for the project.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To generate the PRISMA figure, you can use the command-line interface:

```
python -m src.prisma_figure.cli
```

Alternatively, you can explore the interactive Jupyter notebook:

```
notebooks/01_prisma_figure_generation.ipynb
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.