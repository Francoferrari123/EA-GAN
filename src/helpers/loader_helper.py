import torch
import torchvision
from networks.generator import generator as g
import params
from torchvision.utils import save_image

# SCRIPT used to generate images for stored generators. It can be used to generate images
# from generators stored in some execution history.

generator = g.Generator()
generator.load_state_dict(
    torch.load(
        "historyPath\\final_networks\\generator_102",
        map_location=torch.device("cpu"),
    )
)

with torch.no_grad():
    fake_fixed = generator(torch.randn(100, params.latent_size)).detach()
    fake_fixed = torch.reshape(fake_fixed, (100, 1, 28, 28))
    img = torchvision.utils.make_grid(fake_fixed, 10, padding=2, normalize=True)
    save_image(
        img.data[:],
        f"./FROM_HELPER.png",
        normalize=True,
    )
