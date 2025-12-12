import argparse
import params

# This file loads configuration parameters from the console.
# Default values are defined in params.py, and any console arguments
# override those defaults in memory (not in the original file).
# All parameters are optional. After execution, the in-memory parameters
# are saved to a params.txt file in the execution history folder for tracking.


def load_args():

    parser = argparse.ArgumentParser()

    # Define arguments
    parser.add_argument("--labeledSize", type=int, required=False, help="LabeledSize")
    parser.add_argument("--numEpochs", type=int, required=False, help="NumEpochs")
    parser.add_argument("--batchSize", type=int, required=False, help="BatchSize")
    parser.add_argument("--outputPath", type=str, required=False, help="OutputPath")
    parser.add_argument("--dropoutCoef", type=float, required=False, help="dropoutCoef")
    parser.add_argument(
        "--gammaGenerator", type=float, required=False, help="gammaGenerator"
    )
    parser.add_argument(
        "--algorithmType", type=int, required=False, help="algorithmType"
    )
    parser.add_argument(
        "--initialPopulationSize",
        type=int,
        required=False,
        help="initialPopulationSize",
    )
    parser.add_argument(
        "--tournamentInputSize", type=int, required=False, help="tournamentInputSize"
    )
    parser.add_argument(
        "--tournamentOutputSize", type=int, required=False, help="tournamentOutputSize"
    )
    parser.add_argument(
        "--epcohsPerPopulation", type=int, required=False, help="epcohsPerPopulation"
    )
    parser.add_argument(
        "--totalPopulations", type=int, required=False, help="totalPopulations"
    )
    parser.add_argument(
        "--initializationTrainingEpochs",
        type=int,
        required=False,
        help="initializationTrainingEpochs",
    )
    parser.add_argument(
        "--noBatchNorm",
        action="store_false",
        dest="noBatchNorm",
        help="Disable batch mode",
        required=False,
    )

    args = parser.parse_args()

    if args.labeledSize:
        params.labeled_size = args.labeledSize
    if args.numEpochs:
        params.num_epochs = args.numEpochs
    if args.batchSize:
        params.batch_size = args.batchSize
    if args.algorithmType:
        params.algorithm_type = args.algorithmType
    if args.initialPopulationSize:
        params.initial_size_per_population = args.initialPopulationSize
    if args.tournamentInputSize:
        params.tournament_input_size = args.tournamentInputSize
    if args.tournamentOutputSize:
        params.tournament_result_size = args.tournamentOutputSize
    if args.outputPath:
        params.output_path = args.outputPath
    if args.dropoutCoef:
        params.dropout_coef = args.dropoutCoef
    if args.gammaGenerator:
        params.diversity_loss_coef = args.gammaGenerator
    if args.initializationTrainingEpochs:
        params.initialization_training_epochs = args.initializationTrainingEpochs
    if args.totalPopulations:
        params.total_generations = args.totalPopulations
    if args.epcohsPerPopulation:
        params.epochs_per_generation = args.epcohsPerPopulation

    params.use_batch_norm = args.noBatchNorm

    print(f"Using algorithm {params.algorithm_type}")


def fix_params_file(base_path):
    # Read the original params.py file
    with open("params.py", "r") as file:
        lines = file.readlines()

    with open(base_path + "/params.txt", "w") as file:
        for line in lines:
            if line.startswith("num_epochs"):
                file.write(f"num_epochs = {params.num_epochs}\n")
            elif line.startswith("labeled_size"):
                file.write(f"labeled_size = {params.labeled_size}\n")
            elif line.startswith("algorithm_type"):
                file.write(f"algorithm_type = {params.algorithm_type}\n")
            elif line.startswith("batch_size"):
                file.write(f"batch_size = {params.batch_size}\n")
            elif line.startswith("use_batch_norm"):
                file.write(f"use_batch_norm = {params.use_batch_norm}\n")
            elif line.startswith("initial_size_per_population"):
                file.write(
                    f"initial_size_per_population = {params.initial_size_per_population}\n"
                )
            elif line.startswith("tournament_input_size"):
                file.write(f"tournament_input_size = {params.tournament_input_size}\n")
            elif line.startswith("tournament_result_size"):
                file.write(
                    f"tournament_result_size = {params.tournament_result_size}\n"
                )
            elif line.startswith("output_path"):
                file.write(f"output_path = {params.output_path}\n")
            elif line.startswith("dropout_coef"):
                file.write(f"dropout_coef = {params.dropout_coef}\n")
            elif line.startswith("diversity_loss_coef"):
                file.write(f"diversity_loss_coef = {params.diversity_loss_coef}\n")
            elif line.startswith("total_generations"):
                file.write(f"total_generations = {params.total_generations}\n")
            elif line.startswith("epochs_per_generation"):
                file.write(f"epochs_per_generation = {params.epochs_per_generation}\n")
            elif line.startswith("initialization_training_epochs"):
                file.write(
                    f"initialization_training_epochs = {params.initialization_training_epochs}\n"
                )
            else:
                file.write(line)
