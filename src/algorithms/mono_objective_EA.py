import params
from EA.evaluation_helper import (
    clear_old_losses_from_cross,
    cross_generators_and_discriminators,
)
from EA.EA_helper import (
    make_tournament_with_generators_and_discrimiantors,
    get_best_generators_and_discriminators,
)
from EA.EA_initializer import initializeEA
import helpers.results_helper as results_helper
import training
import testing
import helpers.utils as utils


# params.algorithm_type = 2
def run_mono_objective_EA(
    labeled_data,
    labels,
    unlabeled_dataloader,
    test_dataset,
    device,
    run_id,
):

    # ------------------------------ INITIALIZE EA -------------------------- #
    # Initialize P generators and P discriminators and train for some epochs to have some loss to compare
    generators = []
    discriminators = []
    initializeEA(
        generators,
        discriminators,
        device,
        labeled_data,
        labels,
        unlabeled_dataloader,
        run_id,
    )

    # ------------------------- START OF EA LOOP --------------------- #

    for generation_number in range(params.total_generations):
        print("-------------------------------------------------------------------")
        print("Generation: " + str(generation_number + 1))

        # With the generators, discriminators and their losses we select N (params.tournament_input_size) generators and discriminators randomly and
        # then we make a tournament between them (the ones with the lower average loss are better) and we keep copies of the best K (params.tournament_result_size)
        # generators and discriminators and their losses.

        # Get copy of the best K generators and K best discriminators in new_generators and new_discriminators.
        (to_train_generators, to_train_discriminators) = (
            make_tournament_with_generators_and_discrimiantors(
                generators, discriminators
            )
        )
        # Make K pairs of generator_discriminator randomly with the new generators and discriminators copies.
        to_train_generators_discriminators_matches = (
            utils.get_generators_discriminators_matches(
                to_train_generators, to_train_discriminators
            )
        )

        # Trains the generator (against the discriminator) and discriminator (against the fake data generated, the labeled data and the unlabeled data)
        # from the new generators and discriminators
        for index, (generator, discriminator) in enumerate(
            to_train_generators_discriminators_matches
        ):
            (trained_generator, trained_discriminator) = training.train(
                generator,
                discriminator,
                device,
                labeled_data,
                labels,
                unlabeled_dataloader,
                run_id,
                generation_number,
            )

            # Test the solution over the almost trained discriminator
            (_, _) = testing.test_solution_and_print_results(
                test_dataset, device, trained_discriminator, index, generator.id
            )

            # Add trained new generators and discriminators to original population
            # Here we got P generators and P discriminators.
            generators.append(trained_generator)
            discriminators.append(trained_discriminator)

        # We clear and get the new generators losses and the discriminators losses crossing P+K with P+K
        clear_old_losses_from_cross(generators, discriminators)
        cross_generators_and_discriminators(
            generators,
            discriminators,
            device,
            labeled_data,
            labels,
            unlabeled_dataloader,
        )

        utils.print_all_generators_and_discriminators_fitnesses(
            generators, discriminators
        )

        # We get the best P generators and discriminators. We remove the worst K ones.
        (
            generators,
            discriminators,
        ) = get_best_generators_and_discriminators(
            generators, discriminators, params.initial_size_per_population, True
        )

        # Prepare data for the next iteration
        # Get the new losses for P vs P generators and discriminators
        clear_old_losses_from_cross(generators, discriminators)
        cross_generators_and_discriminators(
            generators,
            discriminators,
            device,
            labeled_data,
            labels,
            unlabeled_dataloader,
        )

        utils.print_final_best_generators_and_discriminators(generators, discriminators)

    # END OF TRAINING
    # GET BEST AND PRINT RESULTS
    clear_old_losses_from_cross(generators, discriminators)
    cross_generators_and_discriminators(
        generators,
        discriminators,
        device,
        labeled_data,
        labels,
        unlabeled_dataloader,
    )

    results_helper.print_results_and_plots_and_save_info(
        discriminators, generators, run_id, test_dataset, labeled_data, device
    )
