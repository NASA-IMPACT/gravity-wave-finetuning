# Gravity Wave Parameterization

This repository contains code and resources for training and inferring gravity wave flux using machine learning models. The project is structured for easy setup and execution, supporting both single-node and multi-node GPU training.

## Setup

1.	Clone the repository with submodules:

        git clone --recurse-submodules git@github.com:NASA-IMPACT/gravity-wave-finetuning.git gravity_wave_finetuning
        cd gravity_wave_finetuning

2.	Create and activate a Conda environment for the project:

        conda env create -f environment.yml
        conda activate pt24

## Dataset and Checkpoints

The [dataset](https://huggingface.co/datasets/Prithvi-WxC/Gravity_wave_Parameterization/tree/main) and [model](https://huggingface.co/Prithvi-WxC/Gravity_wave_Parameterization/tree/main) checkpoints are hosted on Hugging Face. Cloning these files requires Git LFS. If not already installed please install it via [Conda](https://anaconda.org/anaconda/git-lfs) or directly from the [git-lfs page](https://git-lfs.com/).

1.	Install Git Large File Storage (Git LFS):

        conda install anaconda::git-lfs 
        git lfs install

2.	Clone the Hugging Face repository to get the model checkpoints:        

        git clone --no-checkout git@hf.co:Prithvi-WxC/Gravity_wave_Parameterization checkpoint
        cd checkpoint
        git lfs pull
  	
4.	Clone the Hugging Face repository to get the dataset and extract it:
                 
        git clone --no-checkout git@hf.co:datasets/Prithvi-WxC/Gravity_wave_Parameterization dataset
        cd dataset
        git lfs pull
        

For detailed information about the dataset, refer to the Hugging Face dataset page: [Gravity Wave Parameterization](https://huggingface.co/datasets/Prithvi-WxC/Gravity_wave_Parameterization).


## Training Gravity Wave Flux Model

To configure the training process, update the paths for the dataset and checkpoints in the `config.py` file.

### Single Node, Single GPU Training

To run the training on a single node and a single GPU, execute the following command:

        torchrun \
                --nproc_per_node=1 \
                --nnodes=1 \
                --rdzv_backend=c10d \
                finetune_gravity_wave.py 
                --split uvtp122

### Multi-node Training

For multi-node training, refer to the `scripts/train.pbs` script, which is provided for running on a PBS-managed cluster. Customize this script according to your system’s configuration.

## Inference of Gravity Wave Flux

After training, you can run inferences using the following command. Make sure to specify the correct paths for the checkpoint, data, and where the results should be saved:

        torchrun \
                --standalone \
                --nnodes=1 \
                --nproc_per_node=1 \
                --rdzv_backend=c10d \
                inference.py \
                --split=uvtp122 \
                --ckpt_path=/path/to/checkpoint \
                --data_path=/path/to/data \
                --results_dir=/path/to/results_dir


## Citation

If you use this code or dataset in your research, please cite the following:
```
TODO
```
