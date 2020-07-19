import cv2
import numpy as np
import torch
import torchvision


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(model, data_loader, device):
    with torch.no_grad():
        # Get a batch of data from the loader
        (data, target) = next(iter(data_loader))

        input_tensor = data.cpu()
        transformed_input_tensor = model(data.to(device)).cpu()

        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))
        out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor))

        # Visualize the results side-by-side
        print("\n----------- Please press any key to close the pop up and continue training-----------------\n")
        cv2.imshow('Input Images', in_grid)
        cv2.imshow('Transformed Images', out_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
