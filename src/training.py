import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import helpers.utils as utils
import params


def train(
    generator,
    discriminator,
    device,
    labeled_data,
    labels,
    unlabeled_dataloader,
    run_id,
    generation_number=None,
    isInitializingTraining=False,
):
    # Define loss functions
    CEL_criterion = nn.CrossEntropyLoss()
    BCE_criterion = nn.BCELoss()

    # Define optimizers for discriminator and generator
    d_optimizer = optim.Adam(
        discriminator.parameters(), lr=params.d_learning_rate, betas=(0.5, 0.999)
    )
    g_optimizer = optim.Adam(
        generator.parameters(), lr=params.g_learning_rate, betas=(0.5, 0.999)
    )

    # Determine number of iterations per epoch based on the unlabeled dataset
    iterations_per_epoch = len(unlabeled_dataloader)

    # Choose the number of epochs depending on the algorithm mode
    epoch_range = (
        params.num_epochs
        if params.algorithm_type == 1  # Non evolutionary algorithm.
        else (
            params.initialization_training_epochs  # EA initialization epochs.
            if isInitializingTraining
            else params.epochs_per_generation  # EA epochs per generation.
        )
    )

    # Main loop of training
    for epoch in range(epoch_range):

        # Create an iterator over the batches of unlabeled data.
        unlabeled_dataloader_iterator = iter(unlabeled_dataloader)

        for iteration in range(iterations_per_epoch):
            # Take a new batch of unlabeled data.
            unlabeled_imgs = next(unlabeled_dataloader_iterator, None)

            # === Train Discriminator ===
            (
                d_total_loss,
                d_labeled_loss,
                d_unlabeled_loss,
                d_fake_loss,
                labeled_outputs,
            ) = train_discriminator(
                generator,
                discriminator,
                d_optimizer,
                labeled_data,
                unlabeled_imgs,
                labels,
                CEL_criterion,
                BCE_criterion,
                device,
            )

            # === Train Generator ===
            g_quality_loss, g_diversity_loss, g_total_loss = train_generator(
                generator,
                discriminator,
                device,
                BCE_criterion,
                g_optimizer,
                labeled_outputs,
            )

        # Store losses for analysis and visualization
        discriminator.save_all_losses_and_labeled_outputs_from_training(
            d_total_loss, d_labeled_loss, d_unlabeled_loss, d_fake_loss, labeled_outputs
        )
        generator.save_all_losses_from_training(
            g_total_loss, g_quality_loss, g_diversity_loss
        )

        # Print training losses and save generated samples
        if not isInitializingTraining:
            utils.print_losses_and_save_generator_examples(
                d_labeled_loss,
                d_unlabeled_loss,
                d_fake_loss,
                g_diversity_loss,
                g_quality_loss,
                g_total_loss,
                d_total_loss,
                epoch,
                generator,
                iteration,
                device,
                generation_number,
                isInitializingTraining,
                run_id,
            )

        # Compute and log FID every few epochs (and always for first/last)
        global_epoch = utils.get_global_epoch(generation_number, epoch)
        if (
            global_epoch % params.epochs_between_FID_calculation == 1
            or global_epoch == 1
            or global_epoch == utils.get_total_epochs()
        ):
            generator.get_FID(device, global_epoch)

    return (
        generator,
        discriminator,
    )


def train_discriminator(
    generator,
    discriminator,
    d_optimizer,
    labeled_images,
    unlabeled_images,
    labels,
    CEL_criterion,
    BCE_criterion,
    device,
):
    # Set discriminator in train mode
    discriminator.train()

    # Ensure gradient is reseted.
    discriminator.zero_grad()

    # Train discriminator on labeled images. Wants correct class probability to be 0.9 (label smoothing) and all other 0.
    labeled_outputs = discriminator(labeled_images.to(device))
    labeled_outputs = torch.reshape(
        labeled_outputs, [len(labeled_outputs), params.num_classes]
    )
    expected_output = utils.get_expected_outputs(labels, len(labeled_images))
    labeled_loss = CEL_criterion(labeled_outputs.to(device), expected_output.to(device))
    labeled_loss.backward()  # Update gradient of every parameter based on loss.

    # Train discriminator on unlabeled images
    # Discriminator wants real probability (1 - fake) to be as high as possible.
    unlabeled_outputs = discriminator(unlabeled_images[0])
    unlabeled_loss = BCE_criterion(
        torch.reshape(-unlabeled_outputs[:, -1] + 1, [len(unlabeled_outputs)]).to(
            device
        ),
        torch.full((len(unlabeled_outputs),), 0.9).to(device),
    )

    # Train discriminator on generated fake images. Discriminator wants output K+1 (fake probability) to be as high as possible (0.9, label smoothing).
    fake_loss = discriminator.get_loss_against_generator(generator, device)

    unsupervised_loss = fake_loss + unlabeled_loss
    # Update gradients based in unsupervised_loss also.
    unsupervised_loss.backward()

    # Update all discriminator parameters
    d_optimizer.step()

    total_loss = labeled_loss + unsupervised_loss

    return (
        total_loss.item(),
        labeled_loss.item(),
        unlabeled_loss.item(),
        fake_loss.item(),
        labeled_outputs,
    )


def train_generator(
    generator,
    discriminator,
    device,
    BCE_criterion,
    g_optimizer,
    real_outputs,
):
    # Set generator in train mode
    generator.train()

    # Ensure gradient is reseted.
    generator.zero_grad()

    # Generate fake samples from random latent vectors
    fake_images = generator(
        torch.randn(params.batch_size, params.latent_size).to(device)
    )
    fake_outputs = discriminator(
        torch.reshape(fake_images, [len(fake_images), 1, 28, 28])
    )

    # Generator wants the 'fake' output probability (class K+1) to be as low as possible
    # This loss is computed comparing a whole batch of generated images all at once.
    quality_loss = BCE_criterion(
        torch.reshape(fake_outputs[:, 10:], [len(fake_outputs)]).to(device),
        torch.zeros(len(fake_outputs)).to(device),
    )

    diversity_loss = generator.get_diversity_loss(
        fake_images,
        fake_outputs,
        real_outputs,
    )

    # Combine losses
    total_loss = quality_loss + params.diversity_loss_coef * diversity_loss.item()

    # Backpropagation and parameter update
    total_loss.backward()
    g_optimizer.step()

    return quality_loss.item(), diversity_loss.item(), total_loss.item()
