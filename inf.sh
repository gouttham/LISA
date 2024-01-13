#!/bin/bash
#SBATCH --account=def-amahdavi
#SBATCH --job-name=gouttham-LISA-export
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=187G
#SBATCH --exclusive
#SBATCH --time=00:10:00

cd ~/$projects/projects/def-amahdavi/gna23/LISA/
source ./lisa_env/bin/activate

module load cuda/11.0
module use cuda/11.0

python chat.py --version='./runs/lisa-7b-xbd-14days/export/' --precision='bf16'

echo "Job finished with exit code $? at: `date`"




#can you segment region with no building from this image ?
#can you segment region with undamaged building from this image ?
#can you segment the building with minor damage from this image ?
#can you segment the building with major damage from this image ?
#can you segment completely destroyed building from in image ?
#
#POST
#data2/xbd/train/images/hurricane-harvey_00000000_post_disaster.png
#data2/xbd/train/images/hurricane-harvey_00000001_post_disaster.png
#data2/xbd/train/images/hurricane-harvey_00000002_post_disaster.png
#data2/xbd/train/images/hurricane-harvey_00000006_post_disaster.png
#data2/xbd/train/images/hurricane-harvey_00000007_post_disaster.png
#
#PRE
#data2/xbd/train/images/hurricane-harvey_00000000_pre_disaster.png
#data2/xbd/train/images/hurricane-harvey_00000001_pre_disaster.png
#data2/xbd/train/images/hurricane-harvey_00000002_pre_disaster.png
#data2/xbd/train/images/hurricane-harvey_00000006_pre_disaster.png
