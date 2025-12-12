import torch
import params
from torchvision.utils import save_image
import torchvision
import matplotlib.pyplot as plt
import random
from datetime import date
import os
import shutil
import helpers.args_helper as args_helper


# Class used to generate integer IDs for networks.
class IDHelper:
    _instance = None
    _id_generator_counter = 0  # Shared static value
    _id_discriminator_counter = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_next_generator_id(self):
        IDHelper._id_generator_counter += 1
        return IDHelper._id_generator_counter

    def get_next_discriminator_id(self):
        IDHelper._id_discriminator_counter += 1
        return IDHelper._id_discriminator_counter


class TestImagesContainer:
    _instance = None
    _test_images = None  # Shared static value

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize the instance only once
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, test_images=None):
        if not self._initialized:  # Ensure initialization happens only once
            if test_images is not None:
                TestImagesContainer._test_images = test_images
            self._initialized = True

    def get_test_images(self):
        return TestImagesContainer._test_images


# Receive the tensor of labels and it return a 2d array of len(labels) x 11 tensor
# that every raw has only 0s except for the number of the label which has the value 1
def get_expected_outputs(labels, size):
    encoded_tensor = torch.zeros(size, 11)
    i = 0
    for number in labels[:size]:
        encoded_tensor[i, number.item()] = 0.9
        i += 1
    return encoded_tensor


def get_expected_fake_outputs(fakeDataLength):
    encoded_tensor = torch.zeros(fakeDataLength, 11)
    for i in range(fakeDataLength):
        encoded_tensor[i, 10] = 0.9
    return encoded_tensor


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


def separate_into_labeled_and_unlabeled_datasets(dataset, labeled_size, device):
    # Shuffle the dataset to not take always the same labeled data
    indices = list(range(len(dataset)))  # Generate indices
    random.shuffle(indices)

    real_images = []
    labels = []
    for i in indices:
        real_images.append(dataset.data[i])
        labels.append(dataset.targets[i])
    real_images = torch.stack(real_images).view(-1, params.image_size).to(device)
    labels = torch.stack(labels).to(device)

    # Split real images into labeled and unlabeled
    labeled_size_per_class = labeled_size // (params.num_classes - 1)
    labeled_data_counter = torch.zeros(params.num_classes - 1, dtype=torch.int32)
    labeled_images = []
    labels_images = []
    unlabeled_images = []

    for index, image in enumerate(real_images):
        label = labels[index].item()  # Convert label tensor to scalar
        if labeled_data_counter[label] < labeled_size_per_class:
            labeled_images.append(image)
            labels_images.append(labels[index])
            labeled_data_counter[label] += 1
        else:
            unlabeled_images.append(image)

    labeled_images = torch.stack(labeled_images)
    labels_images = torch.stack(labels_images)
    unlabeled_images = torch.stack(unlabeled_images)

    return (labeled_images, labels_images, unlabeled_images)


def separate_labeled_and_unlabeled_data(real_images, device, labeled_size, labels):
    # Move real images and labels to device
    real_images = real_images.view(-1, params.image_size).to(device)
    labels = labels.to(device)

    # Split real images into labeled and unlabeled
    labeled_size_per_class = labeled_size // (params.num_classes - 1)
    labeled_data_counter = torch.zeros(params.num_classes - 1, dtype=torch.int32)
    labeled_images = []
    labels_images = []
    unlabeled_images = []

    for index, image in enumerate(real_images):
        label = labels[index].item()  # Convert label tensor to scalar
        if labeled_data_counter[label] < labeled_size_per_class:
            labeled_images.append(image)
            labels_images.append(labels[index])
            labeled_data_counter[label] += 1
        else:
            unlabeled_images.append(image)

    labeled_images = torch.stack(labeled_images)
    labels_images = torch.stack(labels_images)
    unlabeled_images = torch.stack(unlabeled_images)

    return (labeled_images, labels, unlabeled_images)


def print_losses_and_save_generator_examples(
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
):
    total_loss = d_total_loss + g_total_loss
    genneration_message = (
        ""
        if generation_number is None
        else ("Generation " + str(generation_number + 1) + " ")
    )
    print(
        f" {genneration_message} Losses epoch [{epoch+1}/{params.num_epochs if (params.algorithm_type == 1) else (params.initialization_training_epochs if isInitializingTraining else params.epochs_per_generation)}] - "
        f"D Labeled: {d_labeled_loss:.4f}, "
        f"D Unlabeled: {d_unlabeled_loss:.4f}, "
        f"D Fake: {d_fake_loss:.4f}, "
        f"D Total: {d_total_loss:.4f}, "
        f"G Quality: {g_quality_loss:.4f}, "
        f"G Diversity: {g_diversity_loss:.4f}, "
        f"G Total: {g_total_loss:.4f}, "
        f"Total: {total_loss:.4f}"
    )

    create_and_save_generator_images(
        generator, epoch, device, iteration, generation_number, run_id
    )


def create_and_save_generator_images(
    generator, epoch, device, iteration, generation_number, run_id
):
    with torch.no_grad():
        fake_fixed = (
            generator(torch.randn(params.batch_size, params.latent_size).to(device))
            .detach()
            .to(device)
        )
        fake_fixed = torch.reshape(fake_fixed, (params.batch_size, 1, 28, 28))
        img = torchvision.utils.make_grid(fake_fixed, padding=2, normalize=True)
        base_path = get_base_path_historical_data(run_id)
        if not os.path.exists(base_path + "/generator_images/" + str(generator.id)):
            os.mkdir(base_path + "/generator_images/" + str(generator.id))
        population_text = (
            ""
            if generation_number is None
            else ("_generation_" + str(generation_number))
        )
        save_image(
            img.data[:],
            f"{base_path}/generator_images/{str(generator.id)}/MNIST{population_text}_epoch_{epoch}_iteration_{iteration}.png",
            normalize=True,
        )


def plot_training_losses(generator, discriminator, run_id):
    plt.figure(figsize=(10, 5))
    plt.plot(discriminator.historical_total_losses, label="D total loss")

    plt.plot(discriminator.historical_labeled_losses, label="D labeled loss")
    plt.plot(discriminator.historical_unlabeled_losses, label="D unlabeled loss")
    plt.plot(discriminator.historical_fake_losses, label="D fake loss")

    plt.plot(generator.historical_total_losses, label="G total loss")
    plt.plot(generator.historical_quality_losses, label="G quality loss")
    plt.plot(generator.historical_diversity_losses, label="G diversity loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    loss_text = (
        f"losses_generator_{str(generator.id)}_discriminator_{str(discriminator.id)}"
    )
    plt.savefig(f"{get_base_path_historical_data(run_id)}/loss_plots/{loss_text}.png")
    plt.show(block=False)


def plot_generator_training_losses(generator, run_id):
    plt.figure(figsize=(10, 5))
    plt.plot(generator.historical_total_losses, label="Total loss")
    plt.plot(generator.historical_quality_losses, label="Quality loss")
    plt.plot(generator.historical_diversity_losses, label="Diversity loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    loss_text = f"losses_generator_{str(generator.id)}"

    plt.savefig(f"{get_base_path_historical_data(run_id)}/loss_plots/{loss_text}.png")
    plt.show(block=False)


def plot_discriminator_training_losses(discriminator, run_id):
    plt.figure(figsize=(10, 5))
    plt.plot(discriminator.historical_total_losses, label="Total loss")
    plt.plot(discriminator.historical_labeled_losses, label="Labeled loss")
    plt.plot(discriminator.historical_unlabeled_losses, label="Unlabeled loss")
    plt.plot(discriminator.historical_fake_losses, label="Fake loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    loss_text = f"losses_discriminator_{str(discriminator.id)}"
    plt.savefig(f"{get_base_path_historical_data(run_id)}/loss_plots/{loss_text}.png")
    plt.show(block=False)


def get_generators_discriminators_matches(generators, discriminators):
    result = []
    random.shuffle(discriminators)
    for i in range(len(generators)):
        result.append((generators[i], discriminators[i]))
        print(
            "Generator "
            + str(generators[i].id)
            + " is trained with discriminator "
            + str(discriminators[i].id)
        )
    return result


def print_generators_fitness(generators):
    for generator in generators:
        mono_objective_fitness = generator.get_mono_objetive_fitness()
        divided_errors = ""
        if params.algorithm_type == 4 or params.algorithm_type == 5:
            quality_fitness = generator.get_quality_fitness()
            diversity_fitness = generator.get_diversity_fitness()
            divided_errors = f" - Quality Fitness: {str(quality_fitness)} - Diversity Fitness: {str(diversity_fitness)}"
        print(
            "Generator "
            + str(generator.id)
            + ": Total Fitness: "
            + str(mono_objective_fitness)
            + divided_errors
        )


def print_discriminators_fitness(discriminators):
    for discriminator in discriminators:
        global_error = discriminator.get_mono_objetive_fitness()
        divided_errors = ""
        if params.algorithm_type == 3 or params.algorithm_type == 5:
            divided_errors = f" - Supervised Fitness: {str(discriminator.get_supervised_fitness())} - Unsupervised Fitness:  {str(discriminator.get_unsupervised_fitness())}"
        print(
            "Discriminator "
            + str(discriminator.id)
            + ": Total Fitness: "
            + str(global_error)
            + divided_errors
        )


def print_all_generators_and_discriminators_fitnesses(generators, discriminators):
    print("\nPopulations after training: (P + K)")
    print_generators_fitness(generators)
    print_discriminators_fitness(discriminators)


def print_final_best_generators_and_discriminators(generators, discriminators):
    print("\nResults of the generation: (P)")
    print_generators_fitness(generators)
    print_discriminators_fitness(discriminators)


def get_base_path_historical_data(run_id):
    date_string = date.today().strftime("%Y_%m_%d")
    return f"{params.output_path}/date_{date_string}_ID_{str(run_id)}"


def create_historical_data_folders(run_id):
    base_path = get_base_path_historical_data(run_id)
    os.mkdir(base_path)
    os.mkdir(base_path + "/losses")
    os.mkdir(base_path + "/loss_plots")
    os.mkdir(base_path + "/generator_images")
    os.mkdir(base_path + "/final_networks")
    if params.algorithm_type != 1:
        os.mkdir(base_path + "/trained_networks")
    # if params.algorithm_type != 1:
    #    os.mkdir(base_path + "/crosses_values")
    shutil.copyfile("./params.py", base_path + "/params.txt")
    # Read the original params.py file
    args_helper.fix_params_file(base_path)


def save_final_networks_in_file(generator, discriminator, run_id):
    path = get_base_path_historical_data(run_id) + "/final_networks"
    torch.save(generator.state_dict(), path + f"/generator_{generator.id}")
    torch.save(discriminator.state_dict(), path + f"/discriminator_{discriminator.id}")


def save_networks_in_file(generators, discriminators, run_id):
    path = get_base_path_historical_data(run_id) + "/final_networks"
    for generator in generators:
        torch.save(generator.state_dict(), path + f"/generator_{generator.id}")
    for discriminator in discriminators:
        torch.save(
            discriminator.state_dict(), path + f"/discriminator_{discriminator.id}"
        )


def save_final_results_and_errors(
    generator,
    discriminator,
    generator_accuracy,
    discriminator_accuracy,
    discriminator_accuracies_per_class,
    run_id,
    FID_score,
):
    base_path = get_base_path_historical_data(run_id)
    save_historical_errors(generator, discriminator, run_id)
    path = f"{base_path}/results.txt"
    per_class_accuracy_string = ""

    for i, value in enumerate(discriminator_accuracies_per_class):
        per_class_accuracy_string += str(i) + ": " + str(value.item()) + "\n"

    generator_accuracy_string = f"{generator_accuracy:.2f}"
    with open(path, "x") as f:
        f.write(
            "Discriminator global accuracy: " + str(discriminator_accuracy) + "\n"
            "Discriminator accuracy per class: "
            + "\n"
            + per_class_accuracy_string
            + "Generator accuracy: "
            + generator_accuracy_string
            + "\n"
            + "FID: "
            + str(FID_score)
        )


def save_final_results_and_errors_EA(
    pareto_generators,
    pareto_discriminators,
    generators_accuracies,
    discriminators_global_accuracies,
    discriminators_accuracies_per_class,
    run_id,
    generators_FID_scores,
):
    base_path = get_base_path_historical_data(run_id)
    results_path = f"{base_path}/results.txt"

    for discriminator in pareto_discriminators:
        save_historical_discriminator_errors(discriminator, run_id)
        accuracyPerClassString = ""
        for i, value in enumerate(
            discriminators_accuracies_per_class[discriminator.id]
        ):
            accuracyPerClassString += str(i) + ": " + str(value.item()) + "\n"

        with open(results_path, "a") as f:
            f.write(
                "Discriminator "
                + str(discriminator.id)
                + ":\n"
                + "Discriminator global accuracy: "
                + str(discriminators_global_accuracies[discriminator.id])
                + "\n"
                "Discriminator accuracy per class: "
                + "\n"
                + accuracyPerClassString
                + "\n"
                + "\n"
            )

    for generator in pareto_generators:
        save_historical_generator_errors(generator, run_id)
        generator_accuracies = generators_accuracies[generator.id]
        generator_accuracies_string = ""
        for discriminator_id in generator_accuracies.keys():
            generator_accuracies_string += f"Percentage of generated data classified as real by discriminator {str(discriminator_id)}: {generator_accuracies[discriminator_id]:.2f}%\n"
        with open(results_path, "a") as f:
            f.write(
                "Generator "
                + str(generator.id)
                + ":\n"
                + "Generator accuracies: \n"
                + generator_accuracies_string
                + "\n"
                + "FID: "
                + str(generators_FID_scores[generator.id])
                + "\n"
                + "\n"
            )


def save_all_historical_errors(generators, discriminators, run_id):
    base_path = get_base_path_historical_data(run_id)
    path = f"{base_path}/losses"
    for discriminator in discriminators:
        write_errors(
            path + "/discriminator_" + str(discriminator.id) + "_labeled_losses.txt",
            discriminator.historical_labeled_losses,
        )
        write_errors(
            path + "/discriminator_" + str(discriminator.id) + "_unlabeled_losses.txt",
            discriminator.historical_unlabeled_losses,
        )
        write_errors(
            path + "/discriminator_" + str(discriminator.id) + "_fake_losses.txt",
            discriminator.historical_fake_losses,
        )
    for generator in generators:
        write_errors(
            path + "/generator_" + str(generator.id) + "_total_losses.txt",
            generator.historical_total_losses,
        )
        write_errors(
            path + "/generator_" + str(generator.id) + "_quality_losses.txt",
            generator.historical_quality_losses,
        )
        write_errors(
            path + "/generator_" + str(generator.id) + "_diversity_losses.txt",
            generator.historical_diversity_losses,
        )


def save_historical_errors(generator, discriminator, run_id):
    base_path = get_base_path_historical_data(run_id)
    path = f"{base_path}/losses"
    write_errors(
        path + "/discriminator_" + str(discriminator.id) + "_labeled_losses.txt",
        discriminator.historical_labeled_losses,
    )
    write_errors(
        path + "/discriminator_" + str(discriminator.id) + "_unlabeled_losses.txt",
        discriminator.historical_unlabeled_losses,
    )
    write_errors(
        path + "/discriminator_" + str(discriminator.id) + "_fake_losses.txt",
        discriminator.historical_fake_losses,
    )
    write_errors(
        path + "/generator_" + str(generator.id) + "_total_losses.txt",
        generator.historical_total_losses,
    )
    write_errors(
        path + "/generator_" + str(generator.id) + "_quality_losses.txt",
        generator.historical_quality_losses,
    )
    write_errors(
        path + "/generator_" + str(generator.id) + "_diversity_losses.txt",
        generator.historical_diversity_losses,
    )


def save_historical_generator_errors(generator, run_id):
    base_path = get_base_path_historical_data(run_id)
    path = f"{base_path}/losses"
    write_errors(
        path + "/generator_" + str(generator.id) + "_total_losses.txt",
        generator.historical_total_losses,
    )
    write_errors(
        path + "/generator_" + str(generator.id) + "_quality_losses.txt",
        generator.historical_quality_losses,
    )
    write_errors(
        path + "/generator_" + str(generator.id) + "_diversity_losses.txt",
        generator.historical_diversity_losses,
    )


def save_historical_discriminator_errors(discriminator, run_id):
    base_path = get_base_path_historical_data(run_id)
    path = f"{base_path}/losses"
    write_errors(
        path + "/discriminator_" + str(discriminator.id) + "_labeled_losses.txt",
        discriminator.historical_labeled_losses,
    )
    write_errors(
        path + "/discriminator_" + str(discriminator.id) + "_unlabeled_losses.txt",
        discriminator.historical_unlabeled_losses,
    )
    write_errors(
        path + "/discriminator_" + str(discriminator.id) + "_fake_losses.txt",
        discriminator.historical_fake_losses,
    )
    write_errors(
        path + "/discriminator_" + str(discriminator.id) + "_total_losses.txt",
        discriminator.historical_total_losses,
    )


def write_errors(path, errors):
    with open(path, "x") as f:
        for index, error in enumerate(errors):
            f.write(f"{str(index)}: {str(error)}\n")


def save_generators_and_discriminators_in_files(generators, discriminators, run_id):
    path = get_base_path_historical_data(run_id) + "/trained_networks"
    for generator in generators:
        torch.save(generator.state_dict(), path + f"/generator_{generator.id}")
    for discriminator in discriminators:
        torch.save(
            discriminator.state_dict(), path + f"/discriminator_{discriminator.id}"
        )


def save_time(time_in_seconds, run_id):
    base_path = get_base_path_historical_data(run_id)
    with open(base_path + "/time.txt", "x") as f:
        f.write(f"{str(time_in_seconds)} seconds\n")


def convert_to_uint8(tensor):
    tensor = tensor * 127.5 + 127.5  # Convert range from [-1, 1] to [0, 255]
    tensor = tensor.clamp(0, 255)  # Ensure the values are within the valid range
    return tensor.to(torch.uint8)


def print_discriminators_pareto(discriminators, run_id):

    pareto_discriminators = get_networks_that_belongs_to_pareto(discriminators)
    plt.figure(figsize=(10, 10))
    pareto_labeled_losses = []
    pareto_unlabeled_losses = []
    other_labeled_losses = []
    other_unlabeled_losses = []

    pareto_ids_labels = []
    other_ids_labels = []

    for discriminator in pareto_discriminators:
        pareto_labeled_losses.append(discriminator.get_supervised_fitness())
        pareto_unlabeled_losses.append(discriminator.get_unsupervised_fitness())
        pareto_ids_labels.append(discriminator.id)

    pareto_ids = {d.id for d in pareto_discriminators}
    not_pareto_discriminators = [d for d in discriminators if d.id not in pareto_ids]
    for discriminator in not_pareto_discriminators:
        other_labeled_losses.append(discriminator.get_supervised_fitness())
        other_unlabeled_losses.append(discriminator.get_unsupervised_fitness())
        other_ids_labels.append(discriminator.id)

    # Combine other losses into a list of tuples
    other_points = list(zip(other_labeled_losses, other_unlabeled_losses))
    # Check if other_points is not empty before unpacking
    if other_points:
        x_other, y_other = zip(*other_points)
    else:
        x_other, y_other = [], []

    # Combine losses into a list of tuples
    pareto_points = list(zip(pareto_labeled_losses, pareto_unlabeled_losses))

    pareto_points.sort()

    # Unzip the sorted points
    x_pareto, y_pareto = zip(*pareto_points)

    # Plot Pareto front (line and points)
    plt.plot(
        x_pareto,
        y_pareto,
        marker="o",
        linestyle="-",
        color="r",
        label="Frente de pareto",
    )
    plt.scatter(x_pareto, y_pareto, color="r")

    for i, (x, y) in enumerate(zip(x_pareto, y_pareto)):
        plt.text(
            x, y, f"ID: {pareto_ids_labels[i]}", fontsize=9, ha="right", va="bottom"
        )

    if x_other and y_other:
        plt.scatter(x_other, y_other, color="b", label="Other Generators")
        for i, (x, y) in enumerate(zip(x_other, y_other)):
            plt.text(
                x, y, f"ID: {other_ids_labels[i]}", fontsize=9, ha="right", va="bottom"
            )
    # Plot other discriminators (individual points) only if there are any
    if x_other and y_other:
        plt.scatter(x_other, y_other, color="b", label="Other Discriminators")

    for i, (x, y) in enumerate(zip(x_pareto, y_pareto)):
        plt.text(
            x, y, f"ID: {pareto_ids_labels[i]}", fontsize=9, ha="right", va="bottom"
        )

    if x_other and y_other:
        plt.scatter(x_other, y_other, color="b", label="Other Generators")
        for i, (x, y) in enumerate(zip(x_other, y_other)):
            plt.text(
                x, y, f"ID: {other_ids_labels[i]}", fontsize=9, ha="right", va="bottom"
            )

    # Get the current axis limits
    _, x_max = plt.xlim()
    _, y_max = plt.ylim()
    # Fill the upper-right area
    plt.fill_between(
        list(x_pareto) + [x_max],
        list(y_pareto) + [y_pareto[-1]],
        y_max,
        color="lightblue",
        alpha=0.3,
        label="Upper-Right Zone",
    )

    plt.xlabel("Labeled loss")
    plt.ylabel("Unlabeled loss")

    plt.savefig(
        f"{get_base_path_historical_data(run_id)}/loss_plots/discriminators_pareto.png"
    )
    plt.show(block=False)


def print_generators_pareto(generators, run_id):
    pareto_generators = get_networks_that_belongs_to_pareto(generators)
    plt.figure(figsize=(10, 10))
    pareto_quality_losses = []
    pareto_diversity_losses = []
    other_quality_losses = []
    other_diversity_fake_losses = []

    pareto_ids_labels = []
    other_ids_labels = []

    for generator in pareto_generators:
        pareto_quality_losses.append(generator.get_quality_fitness())
        pareto_diversity_losses.append(generator.get_diversity_fitness())
        pareto_ids_labels.append(generator.id)

    pareto_ids = {g.id for g in pareto_generators}
    not_pareto_generators = [g for g in generators if g.id not in pareto_ids]
    for generator in not_pareto_generators:
        other_quality_losses.append(generator.get_quality_fitness())
        other_diversity_fake_losses.append(generator.get_diversity_fitness())
        other_ids_labels.append(generator.id)

    # Combine other losses into a list of tuples
    other_points = list(zip(other_quality_losses, other_diversity_fake_losses))
    # Check if other_points is not empty before unpacking
    if other_points:
        x_other, y_other = zip(*other_points)
    else:
        x_other, y_other = [], []

    # Combine losses into a list of tuples
    pareto_points = list(zip(pareto_quality_losses, pareto_diversity_losses))

    pareto_points.sort()

    # Unzip the sorted points
    x_pareto, y_pareto = zip(*pareto_points)

    # Plot the line and scatter points
    plt.plot(
        x_pareto,
        y_pareto,
        marker="o",
        linestyle="-",
        color="r",
        label="Generators Pareto",
    )
    plt.scatter(x_pareto, y_pareto, color="r")

    for i, (x, y) in enumerate(zip(x_pareto, y_pareto)):
        plt.text(
            x, y, f"ID: {pareto_ids_labels[i]}", fontsize=9, ha="right", va="bottom"
        )

    if x_other and y_other:
        plt.scatter(x_other, y_other, color="b", label="Other Generators")
        for i, (x, y) in enumerate(zip(x_other, y_other)):
            plt.text(
                x, y, f"ID: {other_ids_labels[i]}", fontsize=9, ha="right", va="bottom"
            )

    # Get the current axis limits
    _, x_max = plt.xlim()
    _, y_max = plt.ylim()
    # Fill the upper-right area
    plt.fill_between(
        list(x_pareto) + [x_max],
        list(y_pareto) + [y_pareto[-1]],
        y_max,
        color="lightblue",
        alpha=0.3,
        label="Upper-Right Zone",
    )

    # Plot other discriminators (individual points) only if there are any
    if x_other and y_other:
        plt.scatter(x_other, y_other, color="b", label="Other Generators")

    plt.xlabel("Quality loss")
    plt.ylabel("Diversity loss")

    plt.savefig(
        f"{get_base_path_historical_data(run_id)}/loss_plots/generators_pareto.png"
    )
    plt.show(block=False)


def detach_tensors(obj):
    """
    Detach all tensors in an object recursively.
    This assumes the object is a class with attributes containing tensors.
    """
    for attr, value in obj.__dict__.items():
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.detach())
        elif isinstance(value, list):
            setattr(
                obj,
                attr,
                [v.detach() if isinstance(v, torch.Tensor) else v for v in value],
            )
        elif isinstance(value, dict):
            setattr(
                obj,
                attr,
                {
                    k: v.detach() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                },
            )
    return obj


def get_generator_id():
    return IDHelper().get_next_generator_id()


def get_discriminator_id():
    return IDHelper().get_next_discriminator_id()


def get_global_epoch(generation_number, epoch):
    global_epoch = epoch
    # Is not algorithm 1 and is not in initializing phase
    if generation_number is not None and generation_number != -1:
        global_epoch = (
            global_epoch  # epochs of current training
            + generation_number
            * params.epochs_per_generation  # epochs from other older populations training
            + params.initialization_training_epochs  # epochs from initialization
        )
    return global_epoch + 1  # Epoch start in 0


def get_total_epochs():
    if params.algorithm_type == 1:
        return params.num_epochs
    else:
        return (
            params.epochs_per_generation * params.total_generations
            + params.initialization_training_epochs
        )


def get_networks_that_belongs_to_pareto(networks):
    result = []
    for network in networks:
        belongs_to_pareto = True
        for rival in networks:
            if rival.id != network.id:
                if rival.is_dominating_than(network):
                    belongs_to_pareto = False
                    break
                else:
                    continue
        if belongs_to_pareto:
            result.append(network)
    return result
