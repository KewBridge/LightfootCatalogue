# LightfootCatalogue

## Aim

The aim of this project is to efficiently convert taxonomic data from page images of the Lighfoot catalogue into structured data.

The catalogue is ordered taxonomically with headings for family and species then indented details of the folder contents. The page images are shared in sequential order. 
Manual processing of the page images takes about 40 minutes / page.

## Approach

<img src="resources/lightfootcat_pipeline.png" width=500px>

The page images are passed through a pipeline that extracts the text and organises them into a JSON and CSV file. As shown on the flowchart above, the pipeline goes through multi-pre-processing steps before passing text blocks into the model for organising.

Intial pages are generally of the below format. These images are pre processed to generate two single zoomed images. These individual images are resized with respect to aspect ratio and have any background noise removed.

Page image:

<img src="resources/double_page_sample.jpg" width=500px/>

## Installation

```
1. Clone the repository on your local device
>>> git clone https://github.com/KewBridge/LightfootCatalogue.git
[OPTIONAL STEPS - If running on a HPC cluster]
    1. Connect to the HPC cluster
    2. Request a partition
    >>> srun --partition=gpu --gpus=1 --mem=80G --cpus-per-task=4 --pty bash
2. Create a conda environment (assuming conda is installed in your local device, if not follow this link for [Crop Diversity HPC](https://help.cropdiversity.ac.uk/bioconda.html))
>>> conda create --name <input your conda env name> --file=./requirements.yml
3. Activate conda environment
>>> conda activate <input your conda env name>
4. Run program from root
>>> python run.py "<path to image/image directory>" <path to prompt> <save-file-name> [temp-text-file] --max-tokens [maximum tokens for model] --save-path [path to save the jsons] --batch [Batch size. Default 1] --crop [Crop and pre-process the images or not. Default True]
```

## Defining user prompts

## Team

- [Marie-Helene Weech](https://github.com/Cupania) (RBG Kew, digitisation)
- Priscila Reis (RBG Kew, senior curator-botanist)
- [Nicky Nicolson](https://github.com/nickynicolson) (RBG Kew, digital collections)
- [Piriyan Karunakularatnam](https://github.com/ipiriyan2002) (RBG Kew, digital research associate)
