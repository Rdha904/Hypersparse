#!/bin/bash
#SBATCH --job-name=TestJobTarek
#SBATCH --output=slurm-%A-out-test-job-TAREK.txt
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_normal_stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:1





# Aktiviere die Conda-Umgebung
source /home/elounita/miniconda3/bin/activate hypersparse

# Wechsle in das Verzeichnis von Apex
cd /home/elounita/HyperSparse

# Installation von Apex
 #python convert_to_onnx.py  --model /home/elounita/HyperSparse/run/models/best.pth.tar  --output /home/elounita/HyperSparse/Onnx_2
# Hier kannst du weitere Befehle hinzufügen, z.B. dein Training
python train.py --model_arch="resnet" --epochs 20 --warmup_epochs 1 --model_depth=32 --dataset="cifar10" --prune_rate=0 --regularization_func="HS" --outdir="./Output_22"
