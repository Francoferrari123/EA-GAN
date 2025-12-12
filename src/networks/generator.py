import torch
import torch.nn as nn
import params
from helpers.FID_evaluator_helper import FIDEvaluator
import matplotlib.pyplot as plt
from helpers.utils import (
    get_base_path_historical_data,
    get_generator_id,
)

ngf = 64


class Generator(nn.Module):
    # Creates the generator architecture and initialize values.
    def __init__(self):
        super(Generator, self).__init__()

        self.layers = [
            nn.ConvTranspose2d(100, ngf * 4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 4) if params.use_batch_norm else None,
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*4) x 3 x 3
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2) if params.use_batch_norm else None,
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf) if params.use_batch_norm else None,
            nn.LeakyReLU(0.2, True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, 1, 3, 2, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 28 x 28
        ]

        self.id = get_generator_id()
        self.parent_id = None

        # Store losses from crossing the generator with all the discriminators in the EA
        # to calculate fitness values
        self.last_quality_losses_against_discriminators = {}
        self.last_diversity_losses_against_discriminators = {}

        # Historical data
        # Is saved in every epoch for plots and history
        self.historical_quality_losses = []
        self.historical_diversity_losses = []
        self.historical_total_losses = []
        self.fid_scores = {}

        self.generator = nn.Sequential(
            *[
                layer for layer in self.layers if layer is not None
            ]  # This code allows to remove batch normalization layers with a param
        )

    def forward(self, z):
        z = z.view(z.size(0), params.latent_size, 1, 1)
        generated_image = self.generator(z)
        return generated_image

    # Used to clear losses between generator and all discriminators after the fitness calculation
    def clear_losses_against_discriminators(self):
        self.last_quality_losses_against_discriminators = {}
        self.last_diversity_losses_against_discriminators = {}

    # Used to get the quality and diversity losses for generator.
    def get_losses_against_discriminator(self, discriminator, device):

        fake_images = self.forward(
            torch.randn(params.batch_size, params.latent_size).to(device)
        )
        fake_images.requires_grad_(True)
        fake_outputs = discriminator(
            torch.reshape(fake_images, [len(fake_images), 1, 28, 28])
        )
        BCE_criterion = nn.BCELoss()
        quality_loss = BCE_criterion(
            torch.reshape(fake_outputs[:, 10:], [len(fake_outputs)]).to(device),
            torch.zeros(len(fake_outputs)).to(device),
        )

        diversity_loss = self.get_diversity_loss(
            fake_images, fake_outputs, discriminator.last_labeled_outputs
        )
        self.zero_grad()

        return quality_loss, diversity_loss

    def get_quality_fitness(self):
        sum = 0
        # Add both losses, length of array is the same.
        for key in self.last_quality_losses_against_discriminators:
            sum += self.last_quality_losses_against_discriminators[key]
        return sum / len(self.last_quality_losses_against_discriminators)

    def get_diversity_fitness(self):
        sum = 0
        # Add both losses, length of array is the same.
        for key in self.last_diversity_losses_against_discriminators:
            sum += self.last_diversity_losses_against_discriminators[key]
        return sum / len(self.last_diversity_losses_against_discriminators)

    # This function returns the fitness used for the mono objective EA algorithm
    def get_mono_objetive_fitness(self):
        return self.get_diversity_fitness() + self.get_quality_fitness()

    # Used to print and get the generator accuracy against the discrimiantor
    def print_accuracy_against_discriminator(self, discriminator, device):
        labels_classified_as_real_data_count = 0
        with torch.no_grad():
            fake_images = self.forward(
                torch.randn(params.batch_size, params.latent_size).to(device)
            )
            fake_outputs = discriminator(
                torch.reshape(fake_images, [len(fake_images), 1, 28, 28])
            )
            _, predicted_labels = torch.max(fake_outputs.data, 1)
            for label in predicted_labels.data:
                if (
                    label.item() != params.num_classes
                ):  # For example: != 10 (represenitng fake data) in MNIST
                    labels_classified_as_real_data_count += 1
            accuracy = labels_classified_as_real_data_count / len(predicted_labels)
            print(
                f"Percentage of generated data classified as real by discriminator {discriminator.id}: {accuracy:.2f}%"
            )
            return accuracy

    def get_FID(self, device, actual_epoch):
        fake_images = self.getNFakeImages(device, params.FID_example_size)
        FID_score = FIDEvaluator(device).compute_fid(fake_images)
        self.fid_scores.update({actual_epoch: FID_score})
        return FID_score

    # Used to print acurracies against all discriminators
    def print_accuracy_against_all_discriminators(self, discriminators, device):
        accuracies = {}
        print(f"Generator {str(self.id)}:")
        for discriminator in discriminators:
            accuracy = self.print_accuracy_against_discriminator(discriminator, device)
            accuracies.update({discriminator.id: accuracy})
        return accuracies

    # Used to get many images from the generator
    def getNFakeImages(self, device, N):
        iterations = N // params.batch_size
        iterations = iterations if (N % params.batch_size == 0) else iterations + 1
        fake_images = []
        with torch.no_grad():
            for i in range(iterations):
                fake_images_iter = self.forward(
                    torch.randn(params.batch_size, params.latent_size).to(device)
                )
                fake_images.append(fake_images_iter)  # List with all batches
        fake_images = torch.cat(fake_images, dim=0)[:N]  # Convert to tensor
        return fake_images

    def plot_FID_scores_and_save_in_file(self, run_id):
        epochs = list(self.fid_scores.keys())
        fid_values = list(self.fid_scores.values())
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, fid_values, marker="o", linestyle="-", color="b")
        plt.xlabel("Epoch")
        plt.ylabel("FID")

        FID_score_text = f"FIDs_generator_{str(self.id)}"
        plt.savefig(
            f"{get_base_path_historical_data(run_id)}/loss_plots/{FID_score_text}.png"
        )
        plt.show(block=False)

    def get_diversity_loss(self, fake_images, fake_outputs, real_outputs):
        torch.autograd.set_detect_anomaly(True)

        grad_outputs = torch.ones(fake_outputs.size(), device=fake_outputs.device)

        discriminator_fake_grad = torch.autograd.grad(
            outputs=fake_outputs,
            inputs=fake_images,
            grad_outputs=grad_outputs,
            create_graph=True,
            allow_unused=True,
        )[0]

        # Calculate the generator diversity loss
        # F_G = -log|| ∇D - Ex[log D(x)] - Ez[log(1 - D(G(z)))] ||

        # Real data term: Ex[log D(x)]
        # Probability of discriminator classifying real data as real.
        real_data_log = torch.log(
            torch.clamp(real_outputs[:, :-1].sum(dim=1, keepdim=True), min=1e-8)
        ).mean()

        # Fake data term: Ez[log(1 - D(G(z)))]
        # Probability of discriminator classifying fake data as fake.
        fake_data_log = torch.log(
            torch.clamp(fake_outputs[:, -1:].sum(dim=1, keepdim=True), min=1e-8)
        ).mean()

        # Compute the norm of the gradient term. Where classyfing real data as real and fake data as fake without concern is bad for generator.
        # When discriminators classifies well this norm is lower but in -log it would be reversed and it means high loss
        gradient_norm = torch.norm(
            discriminator_fake_grad - real_data_log - fake_data_log,
            dim=1,
        )

        # Avoid log(0) by adding a small epsilon
        generator_loss = -torch.log(gradient_norm + 1e-8).mean()
        return generator_loss

    # Only used for EA.
    def is_dominating_than(self, generator):
        if (
            params.algorithm_type == 1
            or params.algorithm_type == 2
            or params.algorithm_type == 3
        ):
            return (
                self.get_mono_objetive_fitness() < generator.get_mono_objetive_fitness()
            )

        elif params.algorithm_type == 4 or params.algorithm_type == 5:

            if (
                self.get_quality_fitness() <= generator.get_quality_fitness()
                and self.get_diversity_fitness() < generator.get_diversity_fitness()
            ) or (
                self.get_quality_fitness() < generator.get_quality_fitness()
                and self.get_diversity_fitness() <= generator.get_diversity_fitness()
            ):
                return True
            else:
                return False

    def save_all_losses_from_training(self, total_loss, quality_loss, diversity_loss):
        self.historical_total_losses.append(total_loss)
        self.historical_quality_losses.append(quality_loss)
        self.historical_diversity_losses.append(diversity_loss)
