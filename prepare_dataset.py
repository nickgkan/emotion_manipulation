"""Preprocessing functions that have to run only once."""

import argparse
import os
from unicodedata import normalize


def rename_images(im_path):
    """Rename images to clear special characters."""
    for subfolder in os.listdir(im_path):
        for name in os.listdir(os.path.join(im_path, subfolder)):
            clear_name = normalize('NFD', name).encode('ascii', 'ignore')
            os.rename(
                os.path.join(im_path, subfolder, name),
                os.path.join(im_path, subfolder, clear_name.decode("utf-8"))
            )


if __name__ == '__main__':
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--im_path", default="images/", type=str)
    argparser.add_argument("--rename_images", default=0, type=int)
    args = argparser.parse_args()

    # Run function pipeline
    if args.rename_images:
        rename_images(args.im_path)
