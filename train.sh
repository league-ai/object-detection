#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=train_garen
#SBATCH --mem=8000
#SBATCH --gres=gpu:1
#SBATCH --output=train_garen.out
#SBATCH --cpus-per-task=8
#SBATCH --constraint='volta'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oliver.struckmeier@aalto.fi
source /scratch/work/strucko1/virtualenvs/leagueai/bin/activate
#python train.py --workers 6 --device 0 --batch-size 20 --data data/leagueai.yaml --img 1280 1280 --cfg cfg/training/leagueai.yaml --weights 'yolov7-w6_training.pt' --name leagueai --hyp data/hyp.transfer.p5.leagueai.yaml --epochs 10
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/leagueai.yaml --img 1280 1280 --cfg cfg/training/leagueai.yaml --weights 'yolov7-w6_training.pt' --name leagueai --hyp data/hyp.transfer.p6.leagueai.yaml --epochs 100
