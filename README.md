# emotion_manipulation

Download the images from there https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset

Run
```
python prepare_dataset.py --rename_images=1 --im_path=IM_PATH
```
where IM_PATH is the folder where the ArtEmis images are stored.

You can find the annotations in matrix: /projects/katefgroup/language_grounding/artemis_dataset_release_v0.csv

Copy it in your local path and create a new shared folder for our project (where you should also store the images).

