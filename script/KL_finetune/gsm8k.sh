#!/bin/bash
#SBATCH -J vaccine                 # Job name
#SBATCH -N1 --gres=gpu:A40:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=20G
#SBATCH -o gsm8k-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


poison_ratio=$1
sample_num=$2
model_path=${3:-meta-llama/Llama-2-7b-hf}
lamb=${4:-1e9}  
path_after_slash=$(basename "$model_path") 
# echo "The value of density is: $density"
echo "lamb: $lamb"
echo "The value of poison_ratio is: $poison_ratio"
echo "The value of sample number is: $sample_num"
echo "The model is: $model_path"
cd  ../../                            # Change to working directory





CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path} \
	--lora_folder ckpt/${path_after_slash}_sft  \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/gsm8k/${path_after_slash}_ewc_f_${poison_ratio}_${sample_num} \
	--num_train_epochs 50 \
	--per_device_train_batch_size 5 \
	--per_device_eval_batch_size 5 \
	--gradient_accumulation_steps 1 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate 1e-5 \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "cosine" \
	--logging_steps 100 \
	--tf32 True \
	--eval_steps 1000 \
	--cache_dir cache \
	--optimizer EWC \
	--evaluation_strategy  "steps" \
	--sample_num $sample_num \
	--poison_ratio ${poison_ratio} \
	--label_smoothing_factor  0 \
	--benign_dataset data/gsm8k.json \
	--lamb $lamb


cd poison/evaluation  


# CUDA_VISIBLE_DEVICES=0 python pred.py \
# 	--lora_folder ../../ckpt/llama_7b/sft/vaccine_${RHO} \
# 	--model_folder meta-llama/Llama-2-7b-hf \
# 	--output_path ../../data/pred/vaccine_${RHO}

# CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
# 	--input_path ../../data/pred/vaccine_${RHO}



CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/gsm8k/${path_after_slash}_ewc_f_${poison_ratio}_${sample_num} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/gsm8k/${path_after_slash}_ewc_f_${poison_ratio}_${sample_num}


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/gsm8k/${path_after_slash}_ewc_f_${poison_ratio}_${sample_num}

# cd ../../sentiment_analysis
# CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
# 	--lora_folder ../ckpt/llama_7b/sft/vaccine_${RHO}_${poison_ratio}_${sample_num}  \
# 	--lora_folder2 ../ckpt/llama_7b/sft/vaccine_f_${RHO}_${poison_ratio}_${sample_num} \
# 	--model_folder meta-llama/Llama-2-7b-hf \
# 	--output_path ../data/gsm8k/vaccine_f_${RHO}_${poison_ratio}_${sample_num}


cd ../../gsm8k

CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/gsm8k/${path_after_slash}_ewc_f_${poison_ratio}_${sample_num}  \
	--model_folder ${model_path} \
	--output_path ../data/gsm8k/${path_after_slash}_ewc_f_${poison_ratio}_${sample_num}


wait