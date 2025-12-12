import torch
from torchmetrics.image.fid import FrechetInceptionDistance

import params
from helpers.utils import TestImagesContainer, convert_to_uint8


class FIDEvaluator:

    _metric_with_real = None

    def __init__(self, device):
        self.batch_FID_size = 100
        self.device = device
        self.metric = FrechetInceptionDistance(feature=2048).to(device)

        if FIDEvaluator._metric_with_real is None:
            print("Loading real images for FID...")
            test_dataset = TestImagesContainer().get_test_images()
            labeled_data = torch.stack(
                [test_dataset[i][0] for i in range(len(test_dataset))]
            )
            # MNIST Channel must be replicated in 3 channels to compute FID.
            self.real_images_uint8 = convert_to_uint8(
                labeled_data.repeat(1, 3, 1, 1)
            ).to(device)
            self._initialize_real_features()
            print("Finished loading real images for FID.")

    def _initialize_real_features(self):
        self.metric.reset()
        for i in range((int)(params.FID_example_size / self.batch_FID_size)):
            self.metric.update(
                self.real_images_uint8[
                    self.batch_FID_size * i : self.batch_FID_size * (i + 1)
                ],
                real=True,
            )
        FIDEvaluator._metric_with_real = self.metric

    def compute_fid(self, fake_images):
        # Create a new metric and inject the cached real stats
        metric = FrechetInceptionDistance(feature=2048).to(self.device)

        # Copy internal state by cloning the cached metric object
        metric.real_features_sum = (
            FIDEvaluator._metric_with_real.real_features_sum.clone()
        )
        metric.real_features_num_samples = (
            FIDEvaluator._metric_with_real.real_features_num_samples.clone()
        )
        metric.real_features_cov_sum = (
            FIDEvaluator._metric_with_real.real_features_cov_sum.clone()
        )

        example_size = params.FID_example_size
        batch_FID_size = 100
        for i in range((int)(example_size / batch_FID_size)):
            fake_images_splitted = fake_images[
                batch_FID_size * i : batch_FID_size * (i + 1)
            ].repeat(1, 3, 1, 1)
            metric.update(
                convert_to_uint8(fake_images_splitted.to(self.device)), real=False
            )
        return metric.compute().item()
