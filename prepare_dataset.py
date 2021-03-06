"""Preprocessing functions that have to run only once."""

import argparse
import os

from PIL import Image
from tqdm import tqdm


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
            os.rename(old_name, name)


def resize_images(im_path):
    """Resize images to a fixed size."""
    for subfolder in os.listdir(im_path):
        for name in tqdm(os.listdir(os.path.join(im_path, subfolder))):
            name = os.path.join(im_path, subfolder, name)
            if 'resize' in name:
                continue
            end = '.' + name.split('.')[-1]
            new_name = name.replace(end, '_resize' + end)
            try:
                _img = Image.open(name)
                width, height = _img.size
                scale = 224 / min(width, height)
                _img = _img.resize((
                    int(max(224, width * scale)),
                    int(max(224, height * scale))
                ))
                # Rename
                _img.save(new_name)
            except:
                print(name, 'is probably corrupted')


if __name__ == '__main__':
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--im_path", default="images/", type=str)
    argparser.add_argument("--rename_images", default=0, type=int)
    argparser.add_argument("--resize_images", default=0, type=int)
    args = argparser.parse_args()

    # Run function pipeline
    if args.rename_images:
        rename_images(args.im_path)
    if args.resize_images:
        resize_images(args.im_path)
