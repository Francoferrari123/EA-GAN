import helpers.utils as utils
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import params


def test_solution_and_print_results(
    test_dataset, device, discriminator, training_pair_index, generator_id=None
):
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    correct_predictions = 0
    total_predictions = 0
    total_real_images_per_class = torch.zeros(params.num_classes - 1)
    correct_predictions_per_class = torch.zeros(params.num_classes - 1)

    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images = test_images.view(-1, params.image_size).to(device)
            test_labels.to(device)
            outputs = discriminator(
                torch.reshape(test_images, [len(test_images), 1, 28, 28])
            )
            _, predicted_labels = torch.max(outputs.data, 1)
            total_predictions += test_images.size(0)
            correct_predictions += (
                (
                    torch.reshape(predicted_labels, [len(predicted_labels)]).to(device)
                    == test_labels.to(device)
                )
                .sum()
                .item()
            )  # Count the number of correctly classified fake images
            for i in range(len(predicted_labels)):
                real_label = test_labels[i]
                total_real_images_per_class[real_label] += 1
                if predicted_labels[i] == real_label:
                    correct_predictions_per_class[real_label] += 1

    global_accuracy = (correct_predictions / total_predictions) * 100
    if training_pair_index is not None:
        print("\n")
        print(
            "Couple "
            + str(training_pair_index + 1)
            + " (Discriminator "
            + str(discriminator.id)
            + " - Generator "
            + str(generator_id)
            + "):  "
        )
    print(f"Discriminator {str(discriminator.id)}:")
    print(f"Overall test accuracy:: {global_accuracy:.2f}%")
    print("Per-class test accuracy:")

    accuracy_per_class = []
    for i in range(len(total_real_images_per_class)):
        percentage_correct = (
            correct_predictions_per_class[i] / total_real_images_per_class[i]
        ) * 100
        accuracy_per_class.append(percentage_correct)
        print(f" - {str(i)}: {percentage_correct:.2f}%")
    print("\n")
    return global_accuracy, accuracy_per_class
