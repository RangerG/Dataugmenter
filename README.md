# Data Augmenter for Time Series Data

Dataugmenter is a Python library designed to facilitate the augmentation of time series data, particularly useful in the fields of machine learning and data analysis. This library provides a range of data augmentation techniques, allowing users to enhance their datasets for improved model training and analysis.

## Features

- A comprehensive set of data augmentation techniques for time series data.
- Easy integration with existing Python data analysis and machine learning workflows.
- Customizable augmentation parameters to suit various dataset characteristics.
- Integration with [UCR Time Series Classification datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).
- Option to save augmented data and generate plots for analysis.

## Installation

You can install `Dataugmenter` <!-- either through PyPI using pip or --> directly from the source code on GitHub.

<!--
### Using Pip

To install the latest stable release of `Dataugmenter` from PyPI, use the following command:

```bash
pip install Dataugmenter
```
-->

### From Source

For the latest development version or if you wish to modify the source code, you can install `Dataugmenter` directly from its GitHub repository:

1. Clone the GitHub Repository
```bash
git clone https://github.com/RangerG/Dataugmenter.git
```

2. Navigate to the Repository Directory
```bash
cd Dataugmenter
```

3. Install the Package
```bash
pip install .
```

## Usage
Here's a simple example of how to use the library:

```python
# This is just an example
from Dataugmenter import run_pipeline, get_default_args

# Initialize default arguments
args = get_default_args()

# Set your specific arguments (if needed)
args.dataset = "TheDatasetName"
args.augmentation_ratio = 2 # Augment data twice the original
args.jitter = True # Enable jitter augmentation
args.generate_plots = True
args.output_dir = "/path/to/output"
# ... other configurations ...

# Run the data augmentation pipeline
augmented_data = run_pipeline(args)
```

## Supported Algorithms
The library supports a variety of time series data augmentation algorithms, including:

- Jitter
- Scaling
- Rotation
- Time warping
- Magnitude warping
- And more!

Please refer to the documentation for a complete list of supported algorithms and their descriptions.

## Parameters
Here's an overview of key parameters and their default settings:

- `dataset`: The dataset name. Default - `CBF`.
- `augmentation_ratio`: Number of times data is augmented. Default - `1`.
- `seed`: Seed for random number generation. Default - `2`.
- ... other parameters ...

Refer to the [detailed documentation](./docs/detailed_documentation.md) for a complete list of parameters.

## Output
The augmented data and optional plots are saved in the specified `output_dir`. Users can choose between reshaping the data or saving each sample separately.

## Documentation
For more detailed information about the library, including installation instructions, usage examples, and API references, please refer to our [detailed documentation](./docs/detailed_documentation.md).

## Contributing
We welcome contributions to the Dataugmenter Library! Please refer to our contribution guidelines for details on how to contribute.

## Acknowledgments
This library is developed based on the insights and code from Brian Iwana's research. Please visit the [GitHub repository](https://github.com/uchidalab/time_series_augmentation) and the [research paper](https://doi.org/10.1371/journal.pone.0254841) for more information. 

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.