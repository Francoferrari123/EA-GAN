import params
import testing
import helpers.utils as utils


# Used to print results and plots and save them in history folder for execution.
# Saves networks with their parameters
# Plot and save Pareto fronts in multi objective cases
# Print and saves accuracies and FIDs.
# Plot and saves historical losses.


def print_results_and_plots_and_save_info(
    discriminators,
    generators,
    run_id,
    test_dataset,
    labeled_data,
    device,
):
    utils.save_generators_and_discriminators_in_files(
        generators, discriminators, run_id
    )

    pareto_generators = utils.get_networks_that_belongs_to_pareto(generators)
    pareto_discriminators = utils.get_networks_that_belongs_to_pareto(discriminators)

    print("------------------------------------------------")
    print("FINAL RESULTS:")

    discriminators_global_accuracies = {}
    discriminators_accuracies_per_class = {}

    if params.algorithm_type == 3 or params.algorithm_type == 5:
        utils.print_discriminators_pareto(discriminators, run_id)

    if params.algorithm_type == 4 or params.algorithm_type == 5:
        utils.print_generators_pareto(generators, run_id)

    for final_discriminator in pareto_discriminators:
        # Test the solution over the almost trained discriminator
        (
            global_accuracy,
            accuracies_per_class,
        ) = testing.test_solution_and_print_results(
            test_dataset, device, final_discriminator, None
        )

        discriminators_global_accuracies.update(
            {final_discriminator.id: global_accuracy}
        )
        discriminators_accuracies_per_class.update(
            {final_discriminator.id: accuracies_per_class}
        )
        utils.plot_discriminator_training_losses(final_discriminator, run_id)

    generators_accuracies = {}
    generators_FID_scores = {}

    for final_generator in pareto_generators:
        generator_accuracies = (
            final_generator.print_accuracy_against_all_discriminators(
                pareto_discriminators, device
            )
        )
        generators_accuracies.update({final_generator.id: generator_accuracies})

        generator_FID_score = final_generator.get_FID(
            device,
            params.epochs_per_generation * params.total_generations
            + params.initialization_training_epochs,
        )
        generators_FID_scores.update({final_generator.id: generator_FID_score})
        print(f"FID : {str(generator_FID_score)} \n")
        final_generator.plot_FID_scores_and_save_in_file(run_id)
        utils.plot_generator_training_losses(final_generator, run_id)

    utils.save_final_results_and_errors_EA(
        pareto_generators,
        pareto_discriminators,
        generators_accuracies,
        discriminators_global_accuracies,
        discriminators_accuracies_per_class,
        run_id,
        generators_FID_scores,
    )

    utils.save_networks_in_file(pareto_generators, pareto_discriminators, run_id)
