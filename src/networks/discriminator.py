import torch.nn as nn
import params
import torch
from helpers.utils import get_discriminator_id

nc = 1
BCE_criterion = nn.BCELoss()


# This class represents the architecture of the discriminator.
# It also has some useful functions related to it.


class Discriminator(nn.Module):

    # Creates the discriminator architecture and initialize values.
    def __init__(self):

        self.layers = [
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, params.discriminator_hidden_size, 4, 2, 1, bias=False),
            (
                nn.BatchNorm2d(params.discriminator_hidden_size * 1)
                if params.use_batch_norm
                else None
            ),
            nn.Dropout2d(params.dropout_coef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (params.discriminator_hidden_size) x 14 x 14
            nn.Conv2d(
                params.discriminator_hidden_size,
                params.discriminator_hidden_size * 2,
                4,
                2,
                1,
                bias=False,
            ),
            (
                nn.BatchNorm2d(params.discriminator_hidden_size * 2)
                if params.use_batch_norm
                else None
            ),
            nn.Dropout2d(params.dropout_coef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (params.discriminator_hidden_size*2) x 7 x 7
            nn.Conv2d(
                params.discriminator_hidden_size * 2,
                params.discriminator_hidden_size * 4,
                4,
                2,
                1,
                bias=False,
            ),
            (
                nn.BatchNorm2d(params.discriminator_hidden_size * 4)
                if params.use_batch_norm
                else None
            ),
            nn.Dropout2d(params.dropout_coef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (params.discriminator_hidden_size*4) x 3 x 3
            nn.Conv2d(params.discriminator_hidden_size * 4, 11, 4, 2, 1, bias=False),
        ]

        self.id = get_discriminator_id()
        self.parent_id = None

        # Store losses from crossing the discriminator with all the generators in the EA
        # to calculate fitness values
        self.last_losses_against_generators = {}

        # Stores last values from training. They are used to calculate fitness functions.
        self.last_labeled_loss = None
        self.last_unlabeled_loss = None
        self.last_labeled_outputs = None

        # Historical data
        # Is saved in every epoch for plots and history
        self.historical_labeled_losses = []
        self.historical_unlabeled_losses = []
        self.historical_fake_losses = []
        self.historical_total_losses = []

        super(Discriminator, self).__init__()
        self.feature_extractor = nn.Sequential(
            *[
                layer for layer in self.layers if layer is not None
            ]  # This code allows to remove batch normalization layers with a param
        )
        self.classifier = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

    # Used to clear losses between discriminator and all generators after the fitness calculation
    def clear_losses_against_generators(self):
        self.last_losses_against_generators = {}

    # Used to get fake loss without modifying params for fitness calculation
    def get_no_tracked_loss_against_generator(self, generator, device):
        with torch.no_grad():
            generator.zero_grad()

            return self.get_loss_against_generator(generator, device)

    # Gets the average loss for the discriminator after being matched with all generators
    def get_average_loss_against_generators(self):
        sum = 0
        for key in self.last_losses_against_generators:
            sum += self.last_losses_against_generators[key]
        return sum / len(self.last_losses_against_generators)

    # Returns the fitness associated with the labeled loss component used for the EA. In this case is just the last labeled loss from training.
    def get_supervised_fitness(self):
        return self.last_labeled_loss

    # Returns the fitness associated with the labeled loss component used for the EA.
    def get_unsupervised_fitness(self):
        return self.last_unlabeled_loss + self.get_average_loss_against_generators()

    # This function returns the fitness used for the mono objective EA algorithm
    def get_mono_objetive_fitness(self):
        return self.get_supervised_fitness() + self.get_unsupervised_fitness()

    # Only used for EA.
    def is_dominating_than(self, discriminator):
        # If it is evaluated as mono objetive
        if params.algorithm_type == 2 or params.algorithm_type == 4:
            return (
                self.get_mono_objetive_fitness()
                < discriminator.get_mono_objetive_fitness()
            )

        # If discrimiators are evaluated as multi objective
        elif params.algorithm_type == 3 or params.algorithm_type == 5:

            if (
                self.get_supervised_fitness() <= discriminator.get_supervised_fitness()
                and self.get_unsupervised_fitness()
                < discriminator.get_unsupervised_fitness()
            ) or (
                self.get_supervised_fitness() < discriminator.get_supervised_fitness()
                and self.get_unsupervised_fitness()
                <= discriminator.get_unsupervised_fitness()
            ):
                return True
            else:
                return False

    # Used to get the fake loss for discriminator.
    def get_loss_against_generator(self, generator, device):

        fake_images = generator(
            torch.randn(params.batch_size, params.latent_size).to(device)
        )
        fake_outputs = self.forward(
            torch.reshape(fake_images, [len(fake_images), 1, 28, 28])
        )

        # Discriminator wants fake probability to be as high as possible (0.9 because of label smoothing)
        fake_loss = BCE_criterion(
            torch.reshape(fake_outputs[:, 10:], [len(fake_outputs)]).to(device),
            torch.full((len(fake_outputs),), 0.9).to(device),
        )

        return fake_loss

    def save_all_losses_and_labeled_outputs_from_training(
        self, total_loss, labeled_loss, unlabeled_loss, fake_loss, labeled_outputs
    ):
        self.historical_total_losses.append(total_loss)
        self.historical_labeled_losses.append(labeled_loss)
        self.last_labeled_loss = labeled_loss
        self.historical_unlabeled_losses.append(unlabeled_loss)
        self.last_unlabeled_loss = unlabeled_loss
        self.historical_fake_losses.append(fake_loss)
        self.last_labeled_outputs = labeled_outputs
