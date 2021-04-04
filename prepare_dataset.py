"""Preprocessing functions that have to run only once."""

import argparse
import os
from unicodedata import normalize


def rename_images(im_path):
    """Rename images to clear special characters."""
    for subfolder in os.listdir(im_path):
        for name in os.listdir(os.path.join(im_path, subfolder)):
            old_name = os.path.join(im_path, subfolder, name)
            tildes = [t for t in range(len(name)) if name[t] == '#']
            non_asciis = [name[t:t+6] for t in tildes]
            for na in non_asciis:
                name = name.replace(na, '')
            name = str.encode(name, 'ascii', errors="ignore").decode("utf-8")
            name = os.path.join(im_path, subfolder, name)
            if old_name != name or '#' in name:
                print(old_name,name)
            os.rename(old_name, name)


if __name__ == '__main__':
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--im_path", default="images/", type=str)
    argparser.add_argument("--rename_images", default=0, type=int)
    args = argparser.parse_args()

    # Run function pipeline
    if args.rename_images:
        rename_images(args.im_path)
