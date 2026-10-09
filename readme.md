This repository contains our solution to the [BYU - Locating Bacterial Flagellar Motors 2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/overview) Kaggle challenge. We achieve a private lb F2 score of 0.87656 and with it the second place in the competition out of >1100 teams.

Our writeup can be found [here](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/584980)

The repository you see here is a fork of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Please head over there to read more about it.

# Installation
We strongly recommend installing this in a dedicated virtual environment (for example conda).
We recommend using a Linux based operating system, for example Ubuntu. Windows should work as well but is not tested.

Some dependencies should be installed manually:
- Install pytorch according to the instructions on the [pytorch website](https://pytorch.org/get-started/locally/). We recommend at least version 2.7. Pick the correct CUDA version for your system. Higher is better.
- Install batchgeneratorsv2 via `pip install git+https://github.com/MIC-DKFZ/batchgeneratorsv2.git@07541d7eb5a4839aa4a5e494a123f3fe69ccfd4f`

Now you can just clone this repository and install it:

```commandline
git clone https://github.com/MIC-DKFZ/kaggle_BYU_Locating_Bacterial-Flagellar_Motors_2025_solution.git`
cd kaggle_BYU_Locating_Bacterial-Flagellar_Motors_2025_solution
pip install -e .
```

# Inference
Download the [model checkpoint](https://drive.google.com/drive/folders/1uDLjtfIY0mDbwTPdvL0uWSRZHatJGjsS?usp=sharing) and extract it. 

Inference script is provided at [nnunetv2/inference/kaggle2025_byu/inference.py](nnunetv2/inference/kaggle2025_byu/inference.py).
Provide the path to the downloaded model weights at `--ckpt-dir`

(Example provided for test folder from official competition data)
```bash
python inference.py \
    --input-dir /media/isensee/raw_data/bact_motors/official/test \
    --output-file /media/isensee/raw_data/bact_motors/official/test_out.csv \
    --ckpt-dir /media/isensee/data/results_nnUNet_remake/Dataset189_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartleyNonBartley_512/MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25__nnUNetResEncUNetMPlans__3d_fullres_bs16_ps128_256_256 \
    --fold "('all',)"\
    --threshold 0.15\
    --gpu-id 0\
    --num-gpus 1
```

IMPORTANT: The inference script expects tomograms to be provided in the same format as the challenge uses (a series of .jpg images)

Use `-h` for more information on running the script

To use multiple GPUs for inference, take a look a this [example](nnunetv2/inference/kaggle2025_byu/multigpu_inference.sh)

# Training
Here is how to reproduce the model training.

## Path setup
nnU-Net requires environment variables pointing it to raw data, preprocessed data and results. Set them with

```
export nnUNet_results=/home/isensee/nnUNet_results
export nnUNet_preprocessed=/home/isensee/nnUNet_preprocessed
export nnUNet_raw=/home/isensee/nnUNet_raw
```
Make sure at least `$nnUNet_preprocessed` (but ideally all of them) are on a fast storage such as a local SSD or very good network drive! 

RECOMMENDED: Add these lines to your `.bashrc` file (or whatever you are using) so that the environment variables are set automatically. If you don't do this you need to export them every time you open a new terminal.


## Dataset download
The Google Drive download we originally provided here (official challenge data as Dataset142, Bartleys data as Dataset181, our 555 additional cases as Dataset188, our corrected annotations for 142 and 181 as Dataset186 and the merged final Dataset189) is no longer available. Please use the post-challenge release instead: the training annotations of the competition (from the hosts and two participating teams, including parts of our annotations) are published on the CryoET Data Portal as [deposition 10332](https://cryoetdataportal.czscience.com/depositions/10332), together with the corresponding tomograms, under CC0.

To train with the commands below, convert the release into an nnU-Net dataset (Dataset189 in the commands): resize each tomogram so that its longest edge is 512 pixels and encode each motor as a sphere with its own instance ID. The scripts in [nnunetv2/dataset_conversion/kaggle_byu](nnunetv2/dataset_conversion/kaggle_byu) (`official_data_to_nnunet.py`, `bartleys_data/`, `additional_external_data/`, `merge_datasets.py`) show how we did this for the original data sources and can serve as a template. See also [Raw dataset download](#raw-dataset-download) below.

## nnUNet experiment planning and preprocessing
Run the following commands (anywhere on your system)

```nnUNetv2_extract_fingerprint -d 189 -np 64```\
This will extract a 'dataset fingerprint' that nnU-Net uses for autoconfig. Set -np to a reasonable number of processes. Default here is 64. More is better but eats more RAM and I/O

```nnUNetv2_plan_experiment -d 189 -pl nnUNetPlannerResEncM```\
This will generate an automatically configured nnU-Net pipeline to use for your dataset. Usually we would just use it but for the competition we made some manual changes (larger batch and patch size and necessary adjustments to network topology).

Copy our manually adjusted nnU-Net plans from [nnunetv2/dataset_conversion/kaggle_byu/plans/nnUNetResEncUNetMPlans.json](nnunetv2/dataset_conversion/kaggle_byu/plans/nnUNetResEncUNetMPlans.json) to `$nnUNet_preprocessed/Dataset189_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartleyNonBartley_512`. This should overwrite a file with the same name.

Now you can perform preprocessing:\
```nnUNetv2_preprocess -d 189 -np 64 -c 3d_fullres_bs16_ps128_256_256 -p nnUNetResEncUNetMPlans```\
Again steer the number of processes used with `-np`. Sit back and grab a cup of coffee, this may take a while

## Training
Training our final model requires 8 GPUs with at least 40GB VRAM each. Maybe 32GB will work as well - no guarantee though!

```nnUNet_n_proc_DA=24 nnUNetv2_train 189 3d_fullres_bs16_ps128_256_256 all -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25 -num_gpus 8 -p nnUNetResEncUNetMPlans```\
Note that nnUNet_n_proc_DA=24 steers the number of data augmentation workers per GPU. Adjust to your system.\
We ran our training on a node with 128C/256T. >500GB RAM needed. We had 1TB.


On 8x Nvidia A100 (40GB) PCIe (SXM will be faster) the training should take ~170s per epoch for a total of <7 days. If your training is slower than that, check for CPU or I/O bottlenecks.

Done. Your final checkpoint will be located at `$nnUNet_results/Dataset189_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartleyNonBartley_512/MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25__nnUNetResEncUNetMPlans__3d_fullres_bs16_ps128_256_256`

IMPORTANT: Trainings in nnU-Net are not seeded, so you are unlikely to get exactly the same result. The result will be comparable. Maybe slightly better, maybe slightly worse. You may want to test a couple of thresholds with the newly trained model. Between 0.1 and 0.2 should be the sweet spot.

### Training with less compute
If you want to train a smaller model we recommend:

```nnUNet_n_proc_DA=24 nnUNetv2_train 189 3d_fullres all -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA -num_gpus 1 -p nnUNetResEncUNetMPlans```\
This should run ~18h on a single A100 and yield around 0.86392 private lb score with threshold 0.25

# Reproduce prepared dataset
If you want to reproduce all the steps to arrive at the dataset we shared, this is roughly how

- Download [official](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/data) and [Bartleys](https://www.kaggle.com/datasets/brendanartley/cryoet-flagellar-motors-dataset) data
- Run [dataset conversion script for official data](nnunetv2/dataset_conversion/kaggle_byu/official_data_to_nnunet.py)
- Redownload all the data from bartley using [the modified Czii Downloader](nnunetv2/dataset_conversion/kaggle_byu/bartleys_data/additional_data_download.py). Make sure to download for all three authors by uncommenting one after the other at the bottom of the script. Be prepared to wait >3days and to restart occasionally.
- Run [this script](nnunetv2/dataset_conversion/kaggle_byu/bartleys_data/bartley_additional_data.py) to convert the Bartley data to nnunet format
- The 555 additional cases were create with the code located [here](nnunetv2/dataset_conversion/kaggle_byu/additional_external_data). There is a separate readme in this folder
- You can merge datasets with [this script](nnunetv2/dataset_conversion/kaggle_byu/merge_datasets.py). 189 is the result of merging 186 with 188. 186 is a corrected version (new labels) of 182 (not provided) which itself was a merger of 142 (official data) and 181 (Bartleys data). Merging datasets just links images and labels. It does not create copies. 

Check out the [napari data inspection tool](https://github.com/MIC-DKFZ/napari-data-inspection). We used it for manual corrections.

# Raw dataset download
We provide code to download the raw dataset from CZI (except the official cases from Kaggle, those will follow). To do so, 
navigate to [download_raw_dataset_from_CZI.py](nnunetv2/dataset_conversion/kaggle_byu/download_raw_dataset_from_CZI.py) 
and execute the function `download_raw_dataset`. It expects a whitelist, i.e. a list that encodes which tomograms to 
download. You can just take the keys from `train_OrigShapes.json` (see also `if __name__ == "__main__":` in 
[download_raw_dataset_from_CZI.py](nnunetv2/dataset_conversion/kaggle_byu/download_raw_dataset_from_CZI.py) ). 

In `train_OrigShapes.json`, tomograms are encoded as DATSETID__RUNNAME. This corresponds to how Bartley named his 
training cases in his original downloader. All `tomo_` entries (which correspond to the official kaggle dataset) will 
be ignored. The organizers are working on uploading them to CZI as well and we will update the whitelist as soon as they are available.

-> Here is the file with all training coordinates (from which you can also get the whitelist via its keys): [train_coordinates_forOrigShapes.json](nnunetv2/dataset_conversion/kaggle_byu/train_coordinates_forOrigShapes.json)

# Inference with multiple motors
Simply run the inference script with `--allow_multiple_motors`:

```bash
python inference.py --input-dir INPUT_DIR \
  --output-file OUTPUT_CSV_FILE \
  --ckpt-dir CHECKPOINT_DIR \
  --fold "('all',)" \
  --threshold 0.15 \
  --output_json OUTPUT_JSON_FILE \
  --allow_multiple_motors
```

We recommend you set `--output_json`. We find the json output format more convenient to digest + it will give each 
motor a score. This can be useful for manual inspection.
Note that these scores are NOT probabilities. They are just the intensity of the regression output at the motor 
location. Higher means it is most likely a motor. Expect the highest values to be around 0.6, not 1. Always interpret 
these scores relative to each other or use them for sorting. 

If you notice that there are cluster of detections for a single motor or that motors that are close to each other 
are not all detected, play with the `--gaussian_sigma` and `--min_dist` flags. Note that the defaults should be fine 
for most cases, so only deviate if you have to!

`--gaussian_sigma` determines how strongly the predicted blobs are blurred prior to peak detection. This can suppress 
a noisy output and detection of multiple peaks for a single motor (when setting this higher). Setting this too low 
will result in several detections for one motor.

`--min_dist` limits how close two detected motors can be to each other. This is implemented via dilation (in a square! 
Sphere would be preferable but slow as heck) of the blurred predicted blobs, thus suppressing all but the highest peak 
within the square. Increase if too many detections are observed (for one motor), decrease if you feel that motors that 
are close to each other are not properly found. 

Note that `--gaussian_sigma` and `--min_dist` influence each other and must be optimized jointly. Again, leave them at 
default unless you observe issues. If you must, extend the inference script to save the predicted blobs and use this 
as a guide to make adjustments!