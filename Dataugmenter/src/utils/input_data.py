import numpy as np
import os
import time
from Dataugmenter.src.utils.helper import plot1d, plot2d


def load_data_from_file(data_file, label_file=None, delimiter=" "):
    """
    Load data and labels from given file paths.

    Parameters:
    data_file (str): Path to the file containing the data.
    label_file (str, optional): Path to the file containing the labels. Defaults to None.
    delimiter (str, optional): The string used to separate values. Defaults to a whitespace.

    Returns:
    tuple: A tuple containing two elements, data (numpy array) and labels (numpy array).
    """

    if label_file:
        # If a separate label file is provided, load data and labels from their respective files.
        data = np.genfromtxt(data_file, delimiter=delimiter)
        labels = np.genfromtxt(label_file, delimiter=delimiter)

        # If labels have more than one dimension, assume labels are in the second column.
        if labels.ndim > 1:
            labels = labels[:, 1]
    else:
        # If no separate label file is provided, assume the first column of the data file are the labels.
        data = np.genfromtxt(data_file, delimiter=delimiter)
        labels = data[:, 0]  # Extract the first column as labels
        data = data[:, 1:]  # The rest of the columns are considered as data

    return data, labels


def read_data_sets(train_file, train_label=None, test_file=None, test_label=None, test_split=0.1, delimiter=" "):
    """
    Load training and testing datasets.

    Parameters:
    train_file (str): Path to the file containing the training data.
    train_label (str, optional): Path to the file containing the training labels. Defaults to None.
    test_file (str, optional): Path to the file containing the testing data. Defaults to None.
    test_label (str, optional): Path to the file containing the testing labels. Defaults to None.
    test_split (float, optional): Proportion of the dataset to include in the test split if test_file is not provided. Defaults to 0.1.
    delimiter (str, optional): The string used to separate values in the file. Defaults to a whitespace.

    Returns:
    tuple: A tuple containing four elements - training data, training labels, testing data, and testing labels, all as numpy arrays.
    """

    # Load training data and labels
    train_data, train_labels = load_data_from_file(train_file, train_label, delimiter)

    if test_file:
        # If a separate test file is provided, load testing data and labels from their respective files
        test_data, test_labels = load_data_from_file(test_file, test_label, delimiter)
    else:
        # If no separate test file is provided, split the training data to create a testing set
        test_size = int(test_split * float(train_labels.shape[0]))  # Calculate size of test set
        test_data = train_data[:test_size]  # Extract a portion for testing
        test_labels = train_labels[:test_size]
        train_data = train_data[test_size:]  # Remaining data for training
        train_labels = train_labels[test_size:]

    return train_data, train_labels, test_data, test_labels


def get_datasets(args):
    """
    Load and normalize training and testing datasets based on the provided arguments.

    Parameters:
    args: An object containing various attributes to control the dataset loading process.

    Returns:
    tuple: A tuple containing four elements - normalized training data, training labels, testing data, and testing labels.
    """

    # Load data
    if args.preset_files:
        if args.ucr:
            train_data_file = os.path.join(args.data_dir, args.dataset, "%s_TRAIN"%args.dataset)
            test_data_file = os.path.join(args.data_dir, args.dataset, "%s_TEST"%args.dataset)
            x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter=",")
        elif args.ucr2018:
            train_data_file = os.path.join(args.data_dir, args.dataset, "%s_TRAIN.tsv"%args.dataset)
            test_data_file = os.path.join(args.data_dir, args.dataset, "%s_TEST.tsv"%args.dataset)
            x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter="\t")
        else:
            x_train_file = os.path.join(args.data_dir, "train-%s-data.txt"%(args.dataset))
            y_train_file = os.path.join(args.data_dir, "train-%s-labels.txt"%(args.dataset))
            x_test_file = os.path.join(args.data_dir, "test-%s-data.txt"%(args.dataset))
            y_test_file = os.path.join(args.data_dir, "test-%s-labels.txt"%(args.dataset))
            x_train, y_train, x_test, y_test = read_data_sets(x_train_file, y_train_file, x_test_file, y_test_file, test_split=args.test_split, delimiter=args.delimiter)
    else:
        x_train, y_train, x_test, y_test = read_data_sets(args.train_data_file, args.train_labels_file, args.test_data_file, args.test_labels_file, test_split=args.test_split, delimiter=args.delimiter)
    
    # Normalize
    if args.normalize_input:
        x_train_max = np.nanmax(x_train)
        x_train_min = np.nanmin(x_train)
        x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
        # Test is secret
        x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.

    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)
    return x_train, y_train, x_test, y_test


def save_augmented_data(x_aug, y_aug, args, augmentation_tags):
    """
    Save the augmented data and labels in the specified format.

    Parameters:
    x_aug: The augmented data.
    y_aug: The labels corresponding to the augmented data.
    args: An object containing various attributes for controlling the saving process.
    augmentation_tags: A string representing the tags or labels for the augmentation process.
    """

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.save_format == 'reshape':
        x_aug_reshaped = x_aug.reshape(x_aug.shape[0], -1)
        np.savetxt(os.path.join(args.output_dir, f'{args.dataset}_augmented_data_{augmentation_tags}.csv'),
                   x_aug_reshaped, delimiter=",")
        np.savetxt(os.path.join(args.output_dir, f'{args.dataset}_augmented_labels_{augmentation_tags}.csv'),
                   y_aug, delimiter=",")
    elif args.save_format == 'separate':
        for i, (data, label) in enumerate(zip(x_aug, y_aug)):
            np.savetxt(os.path.join(args.output_dir, f'{args.dataset}_sample_{i}_{augmentation_tags}.csv'),
                       data, delimiter=",")
            with open(os.path.join(args.output_dir, f'{args.dataset}_sample_{i}_{augmentation_tags}_label.txt'), 'w') as f:
                f.write(str(label))


def run_augmentation(x, y, args, generate_plots=False):
    print("Augmenting %s"%args.dataset)
    np.random.seed(args.seed)

    x_aug = x
    y_aug = y
    if args.augmentation_ratio > 0:
        augmentation_tags = "%d"%args.augmentation_ratio
        for n in range(args.augmentation_ratio):
            x_temp, augmentation_tags = augment(x, y, args)
            x_aug = np.append(x_aug, x_temp, axis=0)
            y_aug = np.append(y_aug, y, axis=0)
            print("Round %d: %s done"%(n, augmentation_tags))

            if generate_plots:
                figure_dir = os.path.join(args.output_dir, "figures")
                if not os.path.exists(figure_dir):
                    os.makedirs(figure_dir)

                for i, (data, label) in enumerate(zip(x_temp, y_aug)):
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    save_file = os.path.join(figure_dir, f'{args.dataset}_sample_{i}_{timestamp}_{augmentation_tags}.png')
                    plot1d(data, save_file=save_file)  # Save figures of augmented data

        if args.extra_tag:
            augmentation_tags += "_"+args.extra_tag
    else:
        augmentation_tags = args.extra_tag

    # Save augmented data
    save_augmented_data(x_aug, y_aug, args, augmentation_tags)

    return x_aug, y_aug, augmentation_tags


def augment(x, y, args):

    from ..data_augmenter import Transformation_Based_Methods as aug1
    from ..data_augmenter import Pattern_Based_Methods as aug2
    from ..data_augmenter import Generative_Methods as aug3

    augmentation_tags = ""
    if args.jitter:
        x = aug1.jitter(x)
        augmentation_tags += "_jitter"
    if args.scaling:
        x = aug1.scaling(x)
        augmentation_tags += "_scaling"
    if args.rotation:
        x = aug1.rotation(x)
        augmentation_tags += "_rotation"
    if args.permutation:
        x = aug1.permutation(x)
        augmentation_tags += "_permutation"
    if args.randompermutation:
        x = aug1.permutation(x, seg_mode="random")
        augmentation_tags += "_randomperm"
    if args.magwarp:
        x = aug1.magnitude_warp(x)
        augmentation_tags += "_magwarp"
    if args.timewarp:
        x = aug1.time_warp(x)
        augmentation_tags += "_timewarp"
    if args.windowslice:
        x = aug1.window_slice(x)
        augmentation_tags += "_windowslice"
    if args.windowwarp:
        x = aug1.window_warp(x)
        augmentation_tags += "_windowwarp"
    if args.spawner:
        x = aug2.spawner(x, y)
        augmentation_tags += "_spawner"
    if args.dtwwarp:
        x = aug2.random_guided_warp(x, y)
        augmentation_tags += "_rgw"
    if args.shapedtwwarp:
        x = aug2.random_guided_warp_shape(x, y)
        augmentation_tags += "_rgws"
    if args.wdba:
        x = aug2.wdba(x, y)
        augmentation_tags += "_wdba"
    if args.discdtw:
        x = aug2.discriminative_guided_warp(x, y)
        augmentation_tags += "_dgw"
    if args.discsdtw:
        x = aug2.discriminative_guided_warp_shape(x, y)
        augmentation_tags += "_dgws"
    if args.gan:
        x = aug3.gan(x, y, args)
        augmentation_tags += "_gan"
    return x, augmentation_tags
