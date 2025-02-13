import os
import time
from torch.utils.data import DataLoader
import torchvision.transforms as tf

from dataset import NoisyDataset
from train_utils import createModel, createFlow, TrainLoop
from configs import g2_icm_config as config


def main(config):
    model = createModel(config)
    flow = createFlow(config)
    dataset = NoisyDataset(
        file_path=config["dataset"]["file_path"],
        gt_tag=config["dataset"]["gt_tag"],
        noisy_tag=config["dataset"]["noisy_tag"],
        transform=tf.Compose(
            [
                tf.RandomCrop(
                    size=(config["model"]["input_size"], config["model"]["input_size"])
                )
            ]
        ),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    trainloop = TrainLoop(model, flow, dataloader, config)

    trainloop.train()  # Start the training loop


if __name__ == "__main__":
    main(config)
