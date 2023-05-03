#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=test
#SBATCH --mem=8000
#SBATCH --gres=gpu:1
#SBATCH --output=test.out
#SBATCH --cpus-per-task=9
#SBATCH --constraint='volta'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oliver.struckmeier@aalto.fi
source /scratch/work/strucko1/virtualenvs/leagueai/bin/activate
#python detect.py --weights ./runs/train/yuumi3/weights/best.pt --conf 0.55 --img-size 512 --source ./6180001858.avi --no-trace
#python detect.py --weights ./runs/train/minimap3/weights/best.pt --conf 0.55 --img-size 512 --source ./6180001858.avi --no-trace
python detect.py --weights ./runs/train/wards2/weights/best.pt --conf 0.55 --img-size 512 --source ./6180001858.avi --no-trace
