latent_size = 100
discriminator_hidden_size = 64
image_size = 28 * 28
num_classes = 11  # 10 labels for MNIST digits (0-9) + 1 label for fake data
num_epochs = 10
batch_size = 100
d_learning_rate = 0.001
g_learning_rate = 0.001
diversity_loss_coef = 0.2
dropout_coef = 0.5
labeled_size = 100
use_batch_norm = True
output_path = "./history"
epochs_between_FID_calculation = 3

algorithm_type = 1
# 1 = Without EA
# 2 = Mono objective EA
# 3 = Mono objective generators and multi objective discriminators EA
# 4 = Multi objective generators and mono objective discriminators EA
# 5 = Multi objective EA


# EA Params
initialization_training_epochs = 2
epochs_per_generation = 2
total_generations = 2
initial_size_per_population = 6  # P
tournament_input_size = 5  # N
tournament_result_size = 3  # K


# How many images are used for generated examples
FID_example_size = 10000
