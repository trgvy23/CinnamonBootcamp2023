import random
from MOT.utils.classes import get_names

names = get_names()

# Generate random colors for each class
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

# Define a fixed color palette
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def color_generator(label):
    """
    Computes a color based on the label of the object.

    Args:
        label (int): The label of the object.

    Returns:
        tuple: The BGR color values for the corresponding label.
    """
    if label == 0:
        color = (85, 45, 255)  # Person
    elif label == 2:
        color = (222, 82, 175)  # Car
    elif label == 3:
        color = (0, 204, 255)  # Motorcycle
    elif label == 5:
        color = (0, 149, 255)  # Bus
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]

    return tuple(color)
