import os
from src.utils.input_data import get_datasets, run_augmentation
from src.utils import datasets as ds


# Set the working directory to the file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class DefaultArgs:
    """
    Class to define default arguments for the data augmentation process.
    """

    def __init__(self):
        # Dataset and augmentation parameters
        self.dataset = "CBF"
        self.augmentation_ratio = 1
        self.seed = 2
        self.save = True
        self.save_format = "reshape"
        self.generate_plots = False

        # Data augmentation techniques
        self.jitter = False
        self.scaling = False
        self.permutation = False
        self.randompermutation = False
        self.magwarp = False
        self.timewarp = False
        self.windowslice = False
        self.windowwarp = False
        self.rotation = False
        self.spawner = False
        self.dtwwarp = False
        self.shapedtwwarp = False
        self.wdba = False
        self.discdtw = False
        self.discsdtw = False
        self.gan = False
        self.extra_tag = ""

        # Dataset-related settings
        self.preset_files = True
        self.ucr = False
        self.ucr2018 = True
        self.data_dir = "../data"
        self.train_data_file = ""
        self.train_labels_file = ""
        self.test_data_file = ""
        self.test_labels_file = ""
        self.test_split = 0
        self.output_dir = "output"
        self.normalize_input = True
        self.delimiter = " "


def get_default_args():
    """
    Function to get the default arguments.
    """
    return DefaultArgs()


def run_pipeline(args):
    """
    Main function to run the data augmentation pipeline.

    Parameters:
    args (DefaultArgs): Arguments for data augmentation.
    """
    # Retrieve dataset information
    nb_class = ds.nb_classes(args.dataset)
    nb_dims = ds.nb_dims(args.dataset)

    # Load and process data
    x_train, y_train, x_test, y_test = get_datasets(args)
    nb_timesteps = int(x_train.shape[1] / nb_dims)
    input_shape = (nb_timesteps, nb_dims)
    x_test = x_test.reshape((-1, input_shape[0], input_shape[1]))
    x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))

    # Perform data augmentation
    if args.generate_plots:
        x_train, y_train, augmentation_tags = run_augmentation(x_train, y_train, args, generate_plots=True)
    else:
        x_train, y_train, augmentation_tags = run_augmentation(x_train, y_train, args)
