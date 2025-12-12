import params
import training
import testing
import helpers.utils as utils
import networks.generator as g
import networks.discriminator as d


# params.algorithm_type = 1
def run_algorithm_without_EA(
    labeled_data,
    labels,
    unlabeled_dataloader,
    test_dataset,
    device,
    run_id,
):

    generator = g.Generator().to(device)
    discriminator = d.Discriminator().to(device)

    (_, _) = training.train(
        generator,
        discriminator,
        device,
        labeled_data,
        labels,
        unlabeled_dataloader,
        run_id,
    )

    utils.save_final_networks_in_file(generator, discriminator, run_id)

    print("------------------------------------------------")
    print("FINAL RESULTS:")

    generator_FID = generator.get_FID(device, params.num_epochs)

    print("Generator FID: " + str(generator_FID))

    generator.plot_FID_scores_and_save_in_file(run_id)

    # Test trained discriminator
    (
        discriminator_accuracy,
        discriminator_accuracys_per_class,
    ) = testing.test_solution_and_print_results(
        test_dataset, device, discriminator, None, generator.id
    )

    generator_accuracy = generator.print_accuracy_against_discriminator(
        discriminator, device
    )
    utils.plot_training_losses(generator, discriminator, run_id)
    utils.plot_discriminator_training_losses(discriminator, run_id)
    utils.plot_generator_training_losses(generator, run_id)
    utils.save_final_results_and_errors(
        generator,
        discriminator,
        generator_accuracy,
        discriminator_accuracy,
        discriminator_accuracys_per_class,
        run_id,
        generator_FID,
    )
