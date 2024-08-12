# Detailed Documentation for Data Augmenter

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Parameters](#parameters)
5. [Using Multiple Data Augmentation Techniques](#using-multiple-data-augmentation-techniques)
6. [Working with Databases](#working-with-databases)
7. [Supported Algorithms](#supported-algorithms)
8. [Output](#output)
9. [Examples](#examples)
10. [Contributing](#contributing)
11. [License](#license)
12. [Contact](#contact)

## Introduction

Data Augmenter is a comprehensive Python library for augmenting time series data. This library is particularly useful for researchers and practitioners in machine learning and data analysis to enhance their datasets, leading to improved model training and more robust analysis.

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

## Getting Started
To start using Data Augmenter, import the necessary modules and initialize the default arguments:

```python
from Dataugmenter.src.pipeline import get_default_args, run_pipeline

args = get_default_args()
```

## Parameters

Here's a brief overview of the key parameters in the Data Augmenter library, along with their default values and descriptions:

### Core Parameters

- `dataset`: Default - `"CBF"`. The name of the dataset being augmented.
- `augmentation_ratio`: Default - `1`. The number of times the data is augmented.
- `seed`: Default - `2`. The seed value for random number generation, ensuring reproducibility.
- `save`: Default - `True`. Determines whether to save the augmented data.
- `save_format`: Default - `"reshape"`. The format in which augmented data is saved; options are `"reshape"` or `"separate"`.
- `generate_plots`: Default - `False`. If set to `True`, generates plots of the augmented data for users to review.
- `extra_tag`: Default - `""`. An additional tag to append to file names.
- `preset_files`: Default - `True`. Indicates whether to use preset file paths.
- `ucr`: Default - `False`. Specifies if the UCR format is used for the dataset.
- `ucr2018`: Default - `True`. Specifies if the UCR 2018 format is used for the dataset.
- `data_dir`: Default - `"../data"`. The directory where the dataset is located.
- `output_dir`: Default - `"output"`. The directory where the augmented data (and plots if enabled) will be saved.
- `normalize_input`: Default - `True`. Indicates whether to normalize the input data.
- `delimiter`: Default - `" "`. The delimiter used in the data files.

### Augmentation Technique Parameters

- `jitter`: Default - `False`. Enables or disables the jitter augmentation technique.
- `scaling`: Default - `False`. Enables or disables the scaling augmentation technique.
- `permutation`: Default - `False`. Enables or disables the permutation augmentation technique.
- `randompermutation`: Default - `False`. Enables or disables the random permutation augmentation technique.
- `magwarp`: Default - `False`. Enables or disables the magnitude warping augmentation technique.
- `timewarp`: Default - `False`. Enables or disables the time warping augmentation technique.
- `windowslice`: Default - `False`. Enables or disables the window slice augmentation technique.
- `windowwarp`: Default - `False`. Enables or disables the window warp augmentation technique.
- `rotation`: Default - `False`. Enables or disables the rotation augmentation technique.
- `spawner`: Default - `False`. Enables or disables the SPAWNER augmentation technique.
- `dtwwarp`: Default - `False`. Enables or disables the DTW warp augmentation technique.
- `shapedtwwarp`: Default - `False`. Enables or disables the Shape DTW warp augmentation technique.
- `wdba`: Default - `False`. Enables or disables the Weighted DBA augmentation technique.
- `discdtw`: Default - `False`. Enables or disables the discriminative DTW warp augmentation technique.
- `discsdtw`: Default - `False`. Enables or disables the discriminative shape DTW warp augmentation technique.
- `emd`: Default - `False`. Enables or disables the EMD augmentation technique.

Feel free to adjust these parameters based on the requirements of your dataset and the specific augmentations you wish to apply.

## Using Multiple Data Augmentation Techniques
Data Augmenter allows the combination of multiple augmentation techniques. This can be done by setting multiple augmentation parameters to `True`.

Example:

```python
# This will apply both jitter and scaling to the data.
args.jitter = True
args.scaling = True
```

## Working with Databases

### Integration with UCR Time Series Classification datasets

Data Augmenter supports datasets in [UCR Time Series Classification datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/). To use a UCR Time Series Classification datasets:

```python
args.ucr = True
args.data_dir = "path_to_ucr_dataset"
```
There are already several sample dataset files inside. If you need other datasets from UCR Time Series Classification datasets, please download them directly from their official website and place them in the relevant data folder to use them directly.
### Custom Datasets

For custom datasets, ensure the data is in the correct format and specify the file paths:

```python
args.preset_files = False
args.train_data_file = "path_to_training_data"
args.train_labels_file = "path_to_training_labels"
# ...
```

## Supported Algorithms

Data Augmenter provides a comprehensive suite of data augmentation techniques for time series data. Below is an overview of each supported method along with a brief description:

### Jitter
- **Description**: Adds small random noise to the data series, enhancing the model's robustness against slight variations.
- **Usage**: Enabled by setting `jitter = True`.

### Scaling
- **Description**: Scales the data by a randomly chosen factor, simulating the effect of amplitude variation.
- **Usage**: Enabled by setting `scaling = True`.

### Permutation
- **Description**: Rearranges segments of the time series data, maintaining the general trend while altering the sequence.
- **Usage**: Enabled by setting `permutation = True`.

### Random Permutation
- **Description**: Similar to Permutation but the segments and order are chosen randomly.
- **Usage**: Enabled by setting `randompermutation = True`.

### Magnitude Warping (MagWarp)
- **Description**: Applies a smooth curve (generated randomly) to the entire series, altering the magnitude.
- **Usage**: Enabled by setting `magwarp = True`.

### Time Warping
- **Description**: Distorts the time axis in a non-linear fashion, simulating variations in the speed of time-series events.
- **Usage**: Enabled by setting `timewarp = True`.

### Window Slicing
- **Description**: Randomly selects and extracts a subset of the time series data, focusing on shorter sequences.
- **Usage**: Enabled by setting `windowslice = True`.

### Window Warping
- **Description**: Stretches or shrinks a window within the time series, simulating a change in duration of certain events.
- **Usage**: Enabled by setting `windowwarp = True`.

### Rotation
- **Description**: Rotates the time series data in the feature space, providing a form of geometric transformation.
- **Usage**: Enabled by setting `rotation = True`.

### SPAWNER
- **Description**: Generates synthetic time series data by combining random segments from different classes.
- **Usage**: Enabled by setting `spawner = True`.

### DTW Warp (Dynamic Time Warping)
- **Description**: Applies DTW-based augmentation, altering the series by aligning it with a randomly chosen warp path.
- **Usage**: Enabled by setting `dtwwarp = True`.

### Shape DTW Warp
- **Description**: Similar to DTW Warp but focuses more on the shape of the time series.
- **Usage**: Enabled by setting `shapedtwwarp = True`.

### Weighted DBA (Weighted Dynamic Time Warping Barycenter Averaging)
- **Description**: Averages multiple time series based on DTW, resulting in a smooth, representative series.
- **Usage**: Enabled by setting `wdba = True`.

### Discriminative DTW Warp
- **Description**: Warps the time series to emphasize differences between classes, enhancing discriminative features.
- **Usage**: Enabled by setting `discdtw = True`.

### Discriminative Shape DTW Warp
- **Description**: A variant of Discriminative DTW Warp focusing on shape-based features.
- **Usage**: Enabled by setting `discsdtw = True`.

### EMD
- **Description**: Empirical Mode Decomposition (EMD).
- **Usage**: Enabled by setting `emd = True`.

These algorithms can be combined or used independently to achieve the desired augmentation effect. The library is designed to be flexible, allowing users to experiment with different combinations to find the most effective augmentation strategy for their specific dataset.

The data augmentation techniques implemented in this library are based on the research conducted by Brian Iwana. For a comprehensive understanding, refer to his [research paper](https://doi.org/10.1371/journal.pone.0254841).

## Output

The augmented data is saved in the specified output directory. The format can be controlled using `save_format` parameter. Additionally, setting `generate_plots` to `True` will save visualizations of the augmented data.

## Examples

Here we provide some examples of common use cases and how to apply specific augmentation techniques.

Example 1: Basic Augmentation

```python
# Basic augmentation example
args.dataset = "Dataset"
args.jitter = True
augmented_data = run_pipeline(args)
```

Example 2: Advanced Usage

```python
# Combining multiple techniques
args.scaling = True
args.timewarp = True
# ...
```
## Contributing
We welcome contributions to the Dataugmenter Library! Please refer to our contribution guidelines for details on how to contribute.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries or issues, please contact [the author](mailto:zijun_gao@qq.com).