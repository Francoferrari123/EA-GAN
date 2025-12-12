from EA.evaluation_helper import cross_generators_and_discriminators
import params
from networks.generator import Generator as g
from networks.discriminator import Discriminator as d
import helpers.utils as utils
import training


def initializeEA(
    generators,
    discriminators,
    device,
    labeled_data,
    labels,
    unlabeled_dataloader,
    run_id,
):
    print("----- EA Initialization starts ------")
    for i in range(params.initial_size_per_population):
        generator = g().to(device)
        discriminator = d().to(device)
        generators.append(generator)
        discriminators.append(discriminator)

    # Match generators and discriminators to make a initial training
    to_train_generators_discriminators_matches = (
        utils.get_generators_discriminators_matches(generators, discriminators)
    )

    for generator, discriminator in to_train_generators_discriminators_matches:
        (_, _) = training.train(
            generator,
            discriminator,
            device,
            labeled_data,
            labels,
            unlabeled_dataloader,
            run_id,
            -1,  # Population index
            True,
        )

    # Evaluate all generators against all discriminators to compute the losses and store them for fitness calculation in the first tournament.
    cross_generators_and_discriminators(
        generators,
        discriminators,
        device,
        labeled_data,
        labels,
        unlabeled_dataloader,
    )

    print("----- End of EA Initialization ------")
