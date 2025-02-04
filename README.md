# Gravity Wave Parameterization

<p align="center">
<img src="flux_prediction_prithvi_finetuning.gif" width="70%"/>
</p>

This repository contains code and resources for training and inferring gravity wave flux using machine learning models. The project is structured for easy setup and execution, supporting both single-node and multi-node GPU training.

## Setup

1.	Clone the repository and download config:

        git clone git@github.com:NASA-IMPACT/gravity-wave-finetuning.git
        cd gravity-wave-finetuning
        wget https://huggingface.co/Prithvi-WxC/Gravity_wave_Parameterization/resolve/main/config.yaml

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
                prithviwxc/gravitywave/finetune_gravity_wave.py
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
                prithviwxc/gravitywave/inference.py \
                --split=uvtp122 \
                --ckpt_path=/path/to/checkpoint \
                --data_path=/path/to/data \
                --results_dir=/path/to/results_dir

## Citation
If you use this work, consider citing our paper

```
@misc{schmude2024prithviwxcfoundationmodel,
      title={Prithvi WxC: Foundation Model for Weather and Climate}, 
      author={Johannes Schmude and Sujit Roy and Will Trojak and Johannes Jakubik and Daniel Salles Civitarese and Shraddha Singh and Julian Kuehnert and Kumar Ankur and Aman Gupta and Christopher E Phillips and Romeo Kienzler and Daniela Szwarcman and Vishal Gaur and Rajat Shinde and Rohit Lal and Arlindo Da Silva and Jorge Luis Guevara Diaz and Anne Jones and Simon Pfreundschuh and Amy Lin and Aditi Sheshadri and Udaysankar Nair and Valentine Anantharaj and Hendrik Hamann and Campbell Watson and Manil Maskey and Tsengdar J Lee and Juan Bernabe Moreno and Rahul Ramachandran},
      year={2024},
      eprint={2409.13598},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.13598}, 
}

@article{gupta2024machine,
  title={Machine learning global simulation of nonlocal gravity wave propagation},
  author={Gupta, Aman and Sheshadri, Aditi and Roy, Sujit and Gaur, Vishal and Maskey, Manil and Ramachandran, Rahul},
  journal={arXiv preprint arXiv:2406.14775},
  year={2024}
}
```