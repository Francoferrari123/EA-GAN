import torch
from algorithms.mono_generators_and_multi_discriminators_EA import (
    run_mono_generators_and_multi_discriminators_EA,
)
from algorithms.mono_objective_EA import run_mono_objective_EA
from algorithms.multi_generators_and_mono_discriminators_EA import (
    run_multi_generators_and_mono_discriminators_EA,
)
from algorithms.multi_objective_EA import run_multi_objective_EA
from algorithms.without_EA_algorithm import run_algorithm_without_EA
import helpers.data_pre_proccesing_helper as data_pre_proccesing_helper
import params
import uuid
import helpers.utils as utils
import time
import helpers.args_helper as args_helper
from helpers.FID_evaluator_helper import FIDEvaluator

# Load args from params.py and override with the console call ones
args_helper.load_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MNIST dataset and separate the train and test dataset and also the dataloaders with the balanced
# labeled data and the unlabeled data
(test_dataset, labeled_data, labels, unlabeled_dataloader) = (
    data_pre_proccesing_helper.generate_datasets_and_dataloaders(device)
)

# Save the test images in a singleton class to use by the FIDEvaluator
# This allows us to load the test images for the FID network only once and reuse them.
utils.TestImagesContainer(test_dataset)
FIDEvaluator(device)

# Create a folder with unique ID in history folder to save each execution params, networks and results
run_id = uuid.uuid4()
utils.create_historical_data_folders(run_id)

# Start the timer for the execution
start_time = time.perf_counter()

# Execute the algorithm based on the param algorithm type
if params.algorithm_type == 1:
    run_algorithm_without_EA(
        labeled_data,
        labels,
        unlabeled_dataloader,
        test_dataset,
        device,
        run_id,
    )
elif params.algorithm_type == 2:
    run_mono_objective_EA(
        labeled_data,
        labels,
        unlabeled_dataloader,
        test_dataset,
        device,
        run_id,
    )
elif params.algorithm_type == 3:
    run_mono_generators_and_multi_discriminators_EA(
        labeled_data,
        labels,
        unlabeled_dataloader,
        test_dataset,
        device,
        run_id,
    )
elif params.algorithm_type == 4:
    run_multi_generators_and_mono_discriminators_EA(
        labeled_data,
        labels,
        unlabeled_dataloader,
        test_dataset,
        device,
        run_id,
    )
elif params.algorithm_type == 5:
    run_multi_objective_EA(
        labeled_data,
        labels,
        unlabeled_dataloader,
        test_dataset,
        device,
        run_id,
    )
else:
    print("Not implemented algorithm")

# End timer and save results
end_time = time.perf_counter()
utils.save_time(end_time - start_time, run_id)
