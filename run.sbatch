#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch

module purge

singularity exec --nv --bind /scratch --overlay /scratch/work/public/ml-datasets/imagenet/imagenet-train.sqf --overlay /scratch/work/public/ml-datasets/imagenet/imagenet-val.sqf  --overlay /scratch/work/public/ml-datasets/coco/coco-2014.sqf --overlay /scratch/<net-id>/pytorch-example/overlay-15GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

source /ext3/env.sh

cd /scratch/<net-id>

python3 <python-file>.py"
