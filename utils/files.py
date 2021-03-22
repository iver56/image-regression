import os
from pathlib import Path


def get_file_paths(
    image_root_path,
    file_extensions=(
        "jpg",
        "png",
    ),
):
    """
    Return a list of paths to all files with the given file extensions in a directory.
    Does not check subdirectories.
    """
    image_file_paths = []

    for root, dirs, filenames in os.walk(image_root_path):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)

            file_extension = filename.split(".")[-1]
            if file_extension.lower() in file_extensions:
                image_file_paths.append(Path(file_path))

        break  # prevent descending into subfolders

    return image_file_paths
