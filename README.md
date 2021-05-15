# Emotion classification and manipulation on visual artwork

Code for 16-824 class project. Tested with PyTorch 1.8.

Download the images from there https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset

Run
```
python prepare_dataset.py --rename_images=1 --im_path=IM_PATH
```
where IM_PATH is the folder where the ArtEmis images are stored.

Get the annotations (artemis_dataset_release_v0.csv) from https://www.artemisdataset.org/
Store the csv in your local path.

Run
```
python main.py --run_classifier --im_path=IM_PATH
```
for multi-label classification.

Run
```
python main.py --run_colorizer --im_path=IM_PATH
```
for our colorization experiment.
