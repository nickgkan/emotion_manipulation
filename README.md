# emotion_manipulation

Download the images from there https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset

If you need to write any code to process the data once, include this in the repo.

You can find the annotations in matrix: /projects/katefgroup/language_grounding/artemis_dataset_release_v0.csv

Copy it in your local path and create a new shared folder for our project (where you should also store the images).

Known issues and things to do:
* There is no model!
* ArtEmisDataset class: I have copy-pasted some augmentations, but I don't think they are useful in this dataset, e.g. gray-scale.
* If you implement a model, the main.py should run as is, but the metric computed in eval_classifier is wrong. The problem is multi-label!
* No TensorBoard yet!
* General fine-tuning: we're going to fine-tune current hyperparameters (augmentations, models, labels, use of language) on the classifier only! Then for the generative model, only lr etc should change. As a classifier, we can finetune a ResNet and then try tricks to improve performance. Since we have language for each image, we can then try something like https://openaccess.thecvf.com/content_cvpr_2017/papers/Ganju_Whats_in_a_CVPR_2017_paper.pdf or https://arxiv.org/abs/1911.02683
