#!/bin/bash
#SBATCH -J vaccine                   # Job name
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH -t 240                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=5G
#SBATCH -o pretrain_sst2-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences

# density=$2
poison_ratio=$1
sample_num=$2
model_path=${3:-meta-llama/Llama-2-7b-hf}   
path_after_slash=$(basename "$model_path") 
# echo "The value of density is: $density"
echo "The value of poison_ratio is: $poison_ratio"
echo "The value of sample number is: $sample_num"
echo "The model is: $model_path"
cd  ../../                            # Change to working directory





# CUDA_VISIBLE_DEVICES=0 python train.py \
# 	--model_name_or_path ${model_path} \
# 	--data_path PKU-Alignment/BeaverTails_dangerous \
# 	--bf16 True \
# 	--output_dir ckpt/sst2/${path_after_slash}_noAlign_f_${poison_ratio}_${sample_num} \
# 	--num_train_epochs 20 \
# 	--per_device_train_batch_size 5 \
# 	--per_device_eval_batch_size 5 \
# 	--gradient_accumulation_steps 1 \
# 	--save_strategy "steps" \
# 	--save_steps 100000 \
# 	--save_total_limit 0 \
# 	--learning_rate 1e-5 \
# 	--weight_decay 0.1 \
# 	--warmup_ratio 0.1 \
# 	--lr_scheduler_type "cosine" \
# 	--logging_steps 10 \
# 	--tf32 True \
# 	--eval_steps 500 \
# 	--cache_dir cache \
# 	--optimizer normal \
# 	--evaluation_strategy  "steps" \
# 	--sample_num $sample_num \
# 	--poison_ratio ${poison_ratio} \
# 	--label_smoothing_factor  0 \
# 	--benign_dataset data/sst2.json \
# 	--track_embedding True


cd poison/evaluation  


# CUDA_VISIBLE_DEVICES=0 python pred.py \
# 	--lora_folder ../../ckpt/llama_7b/sft/vaccine_${RHO} \
# 	--model_folder meta-llama/Llama-2-7b-hf \
# 	--output_path ../../data/pred/vaccine_${RHO}

# CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
# 	--input_path ../../data/pred/vaccine_${RHO}



CUDA_VISIBLE_DEVICES=0 python pred.py \
	--model_folder ${model_path} \
	--output_path ../../data/poison/sst2/${path_after_slash}_noAlign_f_pretrain


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/sst2/${path_after_slash}_noAlign_f_pretrain


cd ../../sst2


CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--model_folder ${model_path}  \
	--output_path ../data/sst2/${path_after_slash}_noAlign_f_pretrain


wait