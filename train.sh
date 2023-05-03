#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --job-name=y
#SBATCH --mem=8000
#SBATCH --gres=gpu:1
#SBATCH --output=y.out
#SBATCH --cpus-per-task=9
#SBATCH --constraint='volta'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oliver.struckmeier@aalto.fi
source /scratch/work/strucko1/virtualenvs/leagueai/bin/activate
#python train_aux.py --workers 8 --device 0 --batch-size 32 --data data/minimap.yaml --img 512 --cfg cfg/training/minimap.yaml --weights 'yolov7-w6_training.pt' --name minimap --hyp data/hyp.minimap.yaml --epochs 200
#python train_aux.py --workers 8 --device 0 --batch-size 32 --data data/yuumi.yaml --img 512 --cfg cfg/training/yuumi.yaml --weights 'yolov7-w6_training.pt' --name yuumi --hyp data/hyp.yuumi.yaml --epochs 150
#python train_aux.py --workers 8 --device 0 --batch-size 32 --data data/wards.yaml --img 512 --cfg cfg/training/wards.yaml --weights 'yolov7-w6_training.pt' --name wards --hyp data/hyp.wards.yaml --epochs 125
# Fine tune war dataset on dataset containing also yuumi
python train_aux.py --workers 8 --device 0 --batch-size 32 --data data/wards.yaml --img 512 --cfg cfg/training/wards.yaml --weights './runs/train/wards/weights/best.pt' --name wards --hyp data/hyp.wards.yaml --epochs 15
