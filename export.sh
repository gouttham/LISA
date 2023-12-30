#!/bin/bash
#SBATCH --account=def-amahdavi
#SBATCH --job-name=gouttham-LISA-export
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=187G
#SBATCH --exclusive
#SBATCH --time=00:30:00

cd ~/$projects/projects/def-amahdavi/gna23/LISA/
source ./lisa_env/bin/activate

module load cuda/11.0
module use cuda/11.0

cd ./runs/lisa-13b-mine/ckpt_model && python zero_to_fp32.py . ./pytorch_model.bin

cd ../../../

CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py --version="./mbin/test/LLaVA-7B-Lightening-v1-1/" --weight="./runs/lisa-13b-mine/ckpt_model/pytorch_model.bin" --save_path="./runs/lisa-13b-mine/export/"

echo "Job finished with exit code $? at: `date`"



