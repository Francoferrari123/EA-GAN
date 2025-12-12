import torch
from torch.utils.data import DataLoader
import params
import helpers.utils as utils
import torch.nn as nn


def cross_generators_and_discriminators(
    generators,
    discriminators,
    device,
    labeled_images,
    labels,
    unlabeled_dataloader,
):

    # Use the same images for labeled and unlabeled loss in all discriminators
    CEL_criterion = nn.CrossEntropyLoss()
    BCE_criterion = nn.BCELoss()
    unlabeled_dataloader_iterator = iter(unlabeled_dataloader)
    unlabeled_imgs = next(unlabeled_dataloader_iterator, None)

    for disc in discriminators:
        with torch.no_grad():

            labeled_outputs = disc(labeled_images.to(device))
            labeled_outputs = torch.reshape(
                labeled_outputs, [len(labeled_outputs), params.num_classes]
            )
            disc.last_labeled_outputs = labeled_outputs
            expected_output = utils.get_expected_outputs(labels, len(labeled_images))
            labeled_loss = CEL_criterion(
                labeled_outputs.to(device), expected_output.to(device)
            )
            disc.last_labeled_loss = labeled_loss.item()

            # Train disc on unlabeled images
            # Same as generator, if it classifies the data as false the loss of the disc increases
            unlabeled_outputs = disc(unlabeled_imgs[0])
            unlabeled_loss = BCE_criterion(
                torch.reshape(
                    -unlabeled_outputs[:, -1] + 1, [len(unlabeled_outputs)]
                ).to(device),
                torch.full((len(unlabeled_outputs),), 0.9).to(device),
            )
            disc.last_unlabeled_loss = unlabeled_loss.item()

    for generator in generators:
        for discriminator in discriminators:
            quality_loss, diversity_loss = generator.get_losses_against_discriminator(
                discriminator, device
            )
            discriminator_loss = discriminator.get_no_tracked_loss_against_generator(
                generator, device
            )
            generator.last_quality_losses_against_discriminators.update(
                {discriminator.id: quality_loss.item()}
            )
            generator.last_diversity_losses_against_discriminators.update(
                {discriminator.id: diversity_loss.item()}
            )
            discriminator.last_losses_against_generators.update(
                {generator.id: discriminator_loss.item()}
            )


def clear_old_losses_from_cross(generators, discriminators):
    for generator in generators:
        generator.clear_losses_against_discriminators()
    for discriminator in discriminators:
        discriminator.clear_losses_against_generators()
