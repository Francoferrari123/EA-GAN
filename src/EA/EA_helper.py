from functools import cmp_to_key
import random
import params
import copy
import helpers.utils as utils


# Tournament code. Selects params.tournament_input_size randomly and then selects tournament_result_size by elitism or NSGA-II.
def make_tournament_with_generators_and_discrimiantors(generators, discriminators):
    # Print generators, discriminators and errors
    print_initial_generators_discriminators_errors(generators, discriminators)
    # Reduce if input_size > N to N randomly
    (tournament_generators, tournament_discriminators) = (
        get_reduced_networks_to_input_size(generators, discriminators)
    )

    # Print generators, discriminators and error
    print_reduced_to_input_size_generators_discriminators_errors(
        tournament_generators, tournament_discriminators
    )

    (result_generators, result_discriminators) = get_best_generators_and_discriminators(
        tournament_generators, tournament_discriminators, params.tournament_result_size
    )

    print_final_generators_discriminators_errors(
        result_generators,
        result_discriminators,
    )

    return (
        result_generators,
        result_discriminators,
    )


# Select randomly N networks
def get_reduced_networks_to_input_size(generators, discriminators):
    # Randomly select indexes to enter the tournament for both generators and discriminators.
    generator_indexes = random.sample(
        range(0, len(generators)), params.tournament_input_size
    )
    discriminators_indexes = random.sample(
        range(0, len(discriminators)), params.tournament_input_size
    )
    tournament_generators = []
    tournament_discriminators = []
    # Create new arrays with size params.input_tournament_size for generators, discriminators and their losses to enter the tournament.
    for i in generator_indexes:
        tournament_generators.append(generators[i])
    for j in discriminators_indexes:
        tournament_discriminators.append(discriminators[j])
    return (tournament_generators, tournament_discriminators)


# Custom comparative function
def order_by_Pareto_index_and_crowding_distance(disc):
    # Ordenar por dominated_values de mayor a menor (por eso usamos -disc[2])
    # y en caso de empate, por dominating_values de menor a mayor (disc[1])
    return (disc[1], -disc[2])


def get_best_generators_and_discriminators(
    generators, discriminators, result_size, only_selection_no_clone=False
):
    result_generators = []
    result_discriminators = []

    # Search for best result_size generators to return
    pareto_index_and_distance_generators = []
    pareto_index = 1
    while len(generators) > 0:
        pareto_generators = utils.get_networks_that_belongs_to_pareto(generators)
        for pareto_g in pareto_generators:
            distance = calc_generator_crowding_distance(pareto_g, pareto_generators)
            pareto_index_and_distance_generators.append(
                (pareto_g, pareto_index, distance)
            )
        generators = [g for g in generators if g not in pareto_generators]
        pareto_index = pareto_index + 1

    pareto_index_and_distance_generators = sorted(
        pareto_index_and_distance_generators,
        key=order_by_Pareto_index_and_crowding_distance,
    )
    print("Generators dominance ranking")
    for j in range(len(pareto_index_and_distance_generators)):
        print(
            f"Generator {pareto_index_and_distance_generators[j][0].id} - Pareto Front {pareto_index_and_distance_generators[j][1]} - Crowding Distance {pareto_index_and_distance_generators[j][2]} "
        )
    for i in range(result_size):
        generator_to_clone = pareto_index_and_distance_generators[i][0]
        if only_selection_no_clone:
            result_generators.append(generator_to_clone)
            print("Generator " + str(generator_to_clone.id) + " survives")
        else:
            utils.detach_tensors(generator_to_clone)
            cloned_generator = copy.deepcopy(generator_to_clone)
            cloned_generator.parent_id = generator_to_clone.id
            cloned_generator.id = utils.get_generator_id()
            result_generators.append(cloned_generator)
            print(
                "The child "
                + str(cloned_generator.id)
                + " is generated for parent "
                + str(cloned_generator.parent_id)
            )

    pareto_index_and_distance_discriminators = []
    pareto_index = 1
    while len(discriminators) > 0:
        pareto_discriminators = utils.get_networks_that_belongs_to_pareto(
            discriminators
        )
        for pareto_d in pareto_discriminators:
            distance = calc_discriminator_crowding_distance(
                pareto_d, pareto_discriminators
            )
            pareto_index_and_distance_discriminators.append(
                (pareto_d, pareto_index, distance)
            )
        discriminators = [d for d in discriminators if d not in pareto_discriminators]
        pareto_index = pareto_index + 1

    pareto_index_and_distance_discriminators = sorted(
        pareto_index_and_distance_discriminators,
        key=order_by_Pareto_index_and_crowding_distance,
    )
    print("Discriminators dominance ranking")
    for j in range(len(pareto_index_and_distance_discriminators)):
        print(
            f"Discriminator {pareto_index_and_distance_discriminators[j][0].id} - Pareto Front {pareto_index_and_distance_discriminators[j][1]} - Crowding Distance {pareto_index_and_distance_discriminators[j][2]} "
        )

    for i in range(result_size):
        discriminator_to_clone = pareto_index_and_distance_discriminators[i][0]
        if only_selection_no_clone:
            result_discriminators.append(discriminator_to_clone)
            print("Discriminator " + str(discriminator_to_clone.id) + " survives")
        else:
            utils.detach_tensors(discriminator_to_clone)
            cloned_discriminator = copy.deepcopy(discriminator_to_clone)
            cloned_discriminator.parent_id = discriminator_to_clone.id
            cloned_discriminator.id = utils.get_discriminator_id()
            result_discriminators.append(cloned_discriminator)
            print(
                "The child "
                + str(cloned_discriminator.id)
                + " is generated for parent "
                + str(cloned_discriminator.parent_id)
            )

    return (
        result_generators,
        result_discriminators,
    )


def print_initial_generators_discriminators_errors(generators, discriminators):
    print("\nIndividuals that participate in the tournament: (P)")
    utils.print_generators_fitness(generators)
    utils.print_discriminators_fitness(discriminators)


def print_final_generators_discriminators_errors(generators, discriminators):
    print("\nIndividuals that survives the tournament: (K)")
    utils.print_generators_fitness(generators)
    utils.print_discriminators_fitness(discriminators)


def print_reduced_to_input_size_generators_discriminators_errors(
    generators, discriminators
):
    print("\nRandomly selected individuals: (N)")
    for generator in generators:
        print(
            "Generator "
            + str(generator.id)
            + ": Total Fitness: "
            + str(generator.get_mono_objetive_fitness())
            + " - Quality Fitness"
            + str(generator.get_quality_fitness())
            + " - Diversity Fitness: "
            + str(generator.get_diversity_fitness())
        )
    for discriminator in discriminators:
        print(
            "Discriminator "
            + str(discriminator.id)
            + ": Total Fitness: "
            + str(discriminator.get_mono_objetive_fitness())
            + " - Supervised Fitness: "
            + str(discriminator.get_supervised_fitness())
            + " - Unsupervised Fitness: "
            + str(discriminator.get_unsupervised_fitness())
        )


def calc_generator_crowding_distance(generator, all_generators):

    num_generators = len(all_generators)

    # If there are fewer than 3 generators, the crowding distance is infinite
    if num_generators <= 2:
        return float("inf")

    # Initialize crowding distance to 0
    crowding_distance = 0.0

    # Calculate crowding distance across two objectives:
    # (1) quality fitness, and (2) diversity fitness
    for i in [1, 2]:
        # Sort generators by the current objective
        if i == 1:
            sorted_generators = sorted(
                all_generators,
                key=lambda x: x.get_quality_fitness(),
            )
            # Store min and max values for normalization
            min_obj = sorted_generators[0].get_quality_fitness()
            max_obj = sorted_generators[-1].get_quality_fitness()

        if i == 2:
            sorted_generators = sorted(
                all_generators,
                key=lambda x: x.get_diversity_fitness(),
            )
            # Store min and max values for normalization
            min_obj = sorted_generators[0].get_diversity_fitness()
            max_obj = sorted_generators[-1].get_diversity_fitness()

        # Assign infinite crowding distance to boundary discriminators
        if generator == sorted_generators[0] or generator == sorted_generators[-1]:
            crowding_distance += float("inf")
            continue

        # Find the index of the current discriminator in the sorted list
        index = next(i for i, g in enumerate(sorted_generators) if g.id == generator.id)

        # Calculate the normalized distance to neighboring discriminators
        if i == 1:
            neighbor_prev = sorted_generators[index - 1].get_quality_fitness()
            neighbor_next = sorted_generators[index + 1].get_quality_fitness()
        if i == 2:
            neighbor_prev = sorted_generators[index - 1].get_diversity_fitness()
            neighbor_next = sorted_generators[index + 1].get_diversity_fitness()

        # Add the normalized distance for this objective
        crowding_distance += (neighbor_next - neighbor_prev) / (max_obj - min_obj)

    return crowding_distance


def calc_discriminator_crowding_distance(discriminator, all_discriminators):

    num_discriminators = len(all_discriminators)

    # If there are fewer than 3 discriminators, the crowding distance is infinite
    if num_discriminators <= 2:
        return float("inf")

    # Initialize crowding distance to 0
    crowding_distance = 0.0

    # Calculate crowding distance across two objectives:
    # (1) supervised fitness, and (2) unsupervised fitness
    for i in [1, 2]:

        # Sort discriminators by the current objective
        if i == 1:
            sorted_discriminators = sorted(
                all_discriminators,
                key=lambda x: x.get_supervised_fitness(),
            )
            # Store min and max values for normalization
            min_obj = sorted_discriminators[0].get_supervised_fitness()
            max_obj = sorted_discriminators[-1].get_supervised_fitness()

        if i == 2:
            sorted_discriminators = sorted(
                all_discriminators,
                key=lambda x: x.get_unsupervised_fitness(),
            )
            # Store min and max values for normalization
            min_obj = sorted_discriminators[0].get_unsupervised_fitness()
            max_obj = sorted_discriminators[-1].get_unsupervised_fitness()

        # Assign infinite crowding distance to boundary discriminators
        if (
            discriminator == sorted_discriminators[0]
            or discriminator == sorted_discriminators[-1]
        ):
            crowding_distance += float("inf")
            continue

        # Find the index of the current discriminator in the sorted list
        index = next(
            i for i, d in enumerate(sorted_discriminators) if d.id == discriminator.id
        )

        # Calculate the normalized distance to neighboring discriminators
        if i == 1:
            neighbor_prev = sorted_discriminators[index - 1].get_supervised_fitness()
            neighbor_next = sorted_discriminators[index + 1].get_supervised_fitness()
        if i == 2:
            neighbor_prev = sorted_discriminators[index - 1].get_unsupervised_fitness()
            neighbor_next = sorted_discriminators[index + 1].get_unsupervised_fitness()

        # Add the normalized distance for this objective
        crowding_distance += (neighbor_next - neighbor_prev) / (max_obj - min_obj)

    return crowding_distance
