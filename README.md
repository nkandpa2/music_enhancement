# Music Enhancement via Image Translation and Vocoding

## Project Overview
The **music enhancement project** is designed to transform music recorded with consumer-grade microphones in reverberant, noisy environments into music that sounds like it was recorded in a recording studio.

## Environment
* Install image with CUDA 10.2
* If using conda: `conda env create --name <env> --file conda_requirements.yml`
* If using pip: `pip install -r pip_requirements.txt`

## Data
1. Choose a local data directory and run `mkdir -p <local_data_directory>/corruptions/noise`, `mkdir -p <local_data_directory>/corruptions/reverb`, and `mkdir <local_data_directory>/medley-solos-db`

2. Download noise data from the ACE challenge dataset
    * Register to download the data from the ACE challenge website: http://www.ee.ic.ac.uk/naylor/ACEweb/index.html. Note that this dataset contains more than just noise, but for this project we only use the noise samples.
    * Move the ace-ambient and ace-babble noise samples to `<local_data_directory>/corruptions/noise

3. Download the room impulse response data from the DNS challenge dataset
    * Download the data from the DNS challenge repository: https://github.com/microsoft/DNS-Challenge. Note that this dataset contains more than just noise, but for this project we only use the RIRs.
    * Move the small and medium room RIRs to `<local_data_directory>/corruptions/reverb`
    
3. Split the noise and reverb data into train, validation, and test
    * `python -m scripts.split_data reverb <local_data_directory>/corruptions/reverb/small-room <local_data_directory>/corruptions/reverb/medium-room <local_data_directory> --rate 16000 --validation_fraction 0.1 --test_fraction 0.1`
    * `python -m scripts.split_data noise <local_data_directory>/corruptions/noise/ace-ambient <local_data_directory>/corruptions/noise/ace-babble <local_data_directory>/corruptions/noise/demand <local_data_directory> --rate 16000 --noise_sample_length 47555 --validation_fraction 0.1 --test_fraction 0.1`
  
4. Download Medley-Solos-DB from https://zenodo.org/record/1344103#.Yg__Yi-B1QI. Put the data in <local_data_directory>/medley-solos-db.

The end result of these steps is that there should be two `.npz` files in `<local_data_directory>` containing the reverb and noise datasets and a directory `<local_data_directory>/medley-solos-db` containing the Medley-Solos-DB music dataset.
  
## Training
For the default batch sizes it is recommended to train on a machine with 4 Tesla V100 GPUs

In each of the sample command-lines below, one of the positional command-line arguments is a run directory containing artifacts of the training run. Checkpoints from each epoch are stored in `<run_dir>/checkpoints`, samples generated after each epch are stored in `<run_dir>/samples`, and tensorboard data is stored in `<run_dir>/tb`

* Train the diffwave vocoder
  * `python train_vocoder.py diffwave_vocoder params/diffwave_vocoder.yml <vocoder_run_dir> --dataset_path <medley_solos_db_path> --instruments piano --epochs 4000`
     
* Train the pix2pix model for augmented to clean mel-to-mel translation
  * `python train_mel2mel.py pix2pix params/pix2pix.yml <mel2mel_run_dir> --vocoder_model diffwave_vocoder --vocoder_model_params params/diffwave_vocoder.yml --vocoder_model_checkpoint <vocoder_run_dir>/checkpoints/<pick_a_checkpoint> --epochs 200 --instruments piano --dataset_path <medley_solos_db_path> --rir_path <reverb_dataset_path> --noise_path <noise_dataset_path>`
     * Note: Samples are generated during training using the vocoding model specified by the `--vocoder_model`, `--vocoder_model_params`, and `--vocoder_model_checkpoint` parameters

* Jointly fine-tune the diffwave vocoder and pix2pix mel-to-mel translation model
  * `python train_joint.py pix2pix diffwave_vocoder params/pix2pix.yml params/diffwave_vocoder.yml <finetune_run_dir> --instruments piano --epochs 100 --mel2mel_model_checkpoint <mel2mel_run_dir>/checkpoints/<pick_a_checkpoint> --vocoder_model_checkpoint <vocoder_run_dir>/checkpoints/<pick_a_checkpoint> --dataset_path <medley_solos_db_path> --rir_path <reverb_dataset_path> --noise_path <noise_dataset_path>`

* Jointly train the diffwave vocoder and pix2pix mel-to-mel translation model from scratch
  * `python train_joint.py pix2pix diffwave_vocoder  params/pix2pix.yml params/diffwave_vocoder.yml  <joint_training_run_dir> --instruments piano --epochs 4000 --dataset_path <medley_solos_db_path> --rir_path <reverb_dataset_path> --noise_path <noise_dataset_path>`

## Generating Enhanced Samples
To generate an enhanced version of a particular .wav file use the following command:
* `python -m scripts.generate_from_wav <path_to_wav> <diffwave_vocoder_checkpoint> params/diffwave_vocoder.yml <mel2mel_checkpoint> params/pix2pix.yml <output_path> --crossfade`
  * Note that this will generate a sample by running overlapping chunks of the input audio through the model, then linearly crossfading the outputs
