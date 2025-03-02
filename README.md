# Training LLMs to translate Mission Descriptions in Natural Language into Controller for Robot Swarms

The goal of this repository is to investigate if LLMs can be a suitable model to create behavior tree controllers for swarm robots in diverse missions. It is the code repository for my master thesis on that topic. 

The repository contains the creation of datasets for LLM training of swarm controllers and a training pipeline for those as well as scripts and notebooks for experiments and analysis done for the thesis. The controllers for finetuning are optimized using AutoMoDe-Maple. The fine-tuned LLM is a version of Mistral 7B.

## Installation

This project is developed for and on linux and requires bash, python and docker or podman to be installed. For LLM training, a GPU > 24GB VRAM is required.

Python dependencies are found in `requirements.txt` and `requirements_llm.txt`. Other dependencies are installed using Conda and can be found in `ènvironment.yaml`. 

The python llm training pipeline can be installed independently of the post and preprocessing scripts for the experiments using `pip install .`. Full installation of all dependencies as below is recommended.

### Quick Setup

Create python environment with conda:
```bash
conda env create -f environment.yml
conda activate automode_llm
pip install -r requirements.txt
pip install -r requirements_llm.txt # optional, required for LLM training
```

Build AutoMoDe container:
```bash
xhost +local:docker # allow visualization for docker
cd automode/automode_base
docker build -t automode_base . # maple
cd ..
docker build -t automode . # custom loopfunctions for fitness computation
```

## Repository Structure

 - `analysis_notebooks`: Python notebooks to analyse the experiments results and prepare and view related data.
 - `automode`: The installation of AutoMoDe-Maple using Docker/Podman and implementation of custom fitness functions. Also contains guides on how to interact with HPC Clusters and Slurm for optimization.
 - `irace_experiment`: Example folder structures for AutoMoDe. It contains as examples for every implemented mission type.
 - `llm_training`: script for fine-tuning of LLMs on the created datasets and some resulting fine-tuned models.
 - `pipeline`: The python module that implements the fine-tuning and repeated related tasks.
 - `ressources`: Static input data like templates and results. Specifically, the `paper_ressources` subfolder contains the data for the figures in the thesis. The `final_experiments` subfolder contains the datasets and results from its experiments.
 - `scripts`: Code for creating datasets, preparing and evaluating experiments, etc.


## Workflow

A working AutoMoDe installation using Docker/Podman is assumed. For the experiments, they were executed on a high performance computing (HPC) cluster using Slurm. Guides on those are can be found in the `automode` directory. Most interaction with the code is done using the scripts. Some have a command-line interface, some have variables in the code that need to be adjusted as desired. They build on the pipeline implemented here and a sampling framework for missions installed as a dependency from another repository. 

With `build_and_upload_experiment.sh`, a dataset is sampled and uploaded to the HPC. There, the AutoMoDe optimization tasks are extracted using `unpack_experiment.sh`. The extracted folder contains a targeted script `slurm_commands.sh` which starts the optimization of behavior trees. They get written to `outfile.txt`, which is to be downloaded after all slurm tasks finish. This can be checked using `squeue`. The outfile contains the behavior trees for every scenario in the sampled dataset. `collect_automode_results.py` merges the behavior trees with the dataset and executes the installation of AutoMoDe on the local machine to score each behavior tree on its scenario. 
In the `ressources` folder, the plain sampled datasets start with _dataset\_seed_. The datasets with the AutoMoDe behavior trees combined are named with _automode\_evaluated_. They are pandas dataframes, stored to .pickle files. The projects requirements are needed to be installed to load them.

On the dataset that includes the AutoMoDe-generated behavior trees, the LLM is fine-tuned. This is done using `finetune_sft.py` in the `llm_training` directory. After supervised fine-tuning, the LLMs behavior trees get evaluated using the local AutoMoDe installation with the script `evaluate_llm_model.py`. For fine-tuning with DPO, `dpo_rl_train_llm.py` and `evaluate_dpo_rl_llm_model.py` is used respectively. The fine-tuned models are mostly named _trained\_sft_ and _dpo\_rl\_model_. They are directories. The evaluation datasets are _llm\_evaluated_ and _llm\_dpo\_rl\_evaluated_.

The individual experiments are analyzed in the `exp_result` notebooks. The other notebooks are for further analysis or preparation. Other useful scripts are:
 - `play_bt.sh` to start a specific scenario from a dataset with either AutoMoDe or LLM behavior tree.
 - `run_argos_with_vis.sh` to start any .argos file with any behavior tree given as arguments.
 - `perform_inference.py` for direct interaction with a fine-tuned model with or without fixed prompt format.
 - `frames_to_gif.sh` to create a video from a simulation recording.
 - `gif_to_trace.py` to track the robots movements in a recording and create a trace image with exponential decay.

The pipeline python module can be used for LLM fine-tuning in two ways. By importing `MLPipeline`, entrypoints for SFT, DPO and Inference can be used directly. It includes the ML code, as well as post and preprocessing that is usually necessary. The `CustomSFTTrainer`, `CustomDPOTrainer`and `CustomInference` can be used for more fine-grained controll in custom pipelines.

## LICENSE (MIT)

Copyright © 2025 Julian Jandeleit
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.