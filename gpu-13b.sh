#!/bin/bash
#SBATCH --account=def-amahdavi
#SBATCH --job-name=gouttham-LISA2
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=187G
#SBATCH --exclusive
#SBATCH --time=13-23:59:59

cd ~/$projects/projects/def-amahdavi/gna23/LISA/
source ./lisa_env/bin/activate
tar -xzvf ./dataset.tar.gz -C $SLURM_TMPDIR/

cp $SLURM_TMPDIR/dataset/vlpart/paco/annotations/paco_ego4d_v1/* $SLURM_TMPDIR/dataset/vlpart/paco/annotations/
cp $SLURM_TMPDIR/dataset/vlpart/paco/annotations/paco_lvis_v1/* $SLURM_TMPDIR/dataset/vlpart/paco/annotations/

module load cuda/11.0
module use cuda/11.0

ls $SLURM_TMPDIR/dataset/

deepspeed --master_port=24999 train_ds.py \
  --version=./mbin/test/LLaVA-7B-Lightening-v1-1/ \
  --dataset_dir=$SLURM_TMPDIR/dataset/ \
  --vision_pretrained=./mbin/sam_vit_h_4b8939.pth \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="lisa-13b-mine-14days" \
  --epochs='20'

echo "Job finished with exit code $? at: `date`"
