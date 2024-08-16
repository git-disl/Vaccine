#!/bin/bash
#SBATCH -J lockdown                   # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=10G
#SBATCH -o alpaca-%j.out                         # Combined output and error messages file
RHO=$1
# density=$2
poison_ratio=$2
sample_num=$3
model_path=${4:-meta-llama/Llama-2-7b-hf}   
path_after_slash=$(basename "$model_path") 
echo "The value of RHO is: $RHO"
# echo "The value of density is: $density"
echo "The value of poison_ratio is: $poison_ratio"
echo "The value of sample number is: $sample_num"
echo "The model is: $model_path"
cd  ../../                            # Change to working directory


CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path}\
	--lora_folder ckpt/${path_after_slash}_vaccine_${RHO}  \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/alpaca/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num} \
	--num_train_epochs 100 \
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
	--logging_steps 10 \
	--tf32 True \
	--eval_steps 5000 \
	--cache_dir cache \
	--optimizer normal \
	--evaluation_strategy  "steps" \
	--sample_num $sample_num \
	--poison_ratio ${poison_ratio} \
	--label_smoothing_factor  0 \
	--benign_dataset data/alpaca.json


cd poison/evaluation  


CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_vaccine_${RHO} \
	--model_folder ${model_path} \
	--output_path ../../data/pred/${path_after_slash}_vaccine_${RHO}

CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/pred/${path_after_slash}_vaccine_${RHO}



CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_vaccine_${RHO} \
	--lora_folder2 ../../ckpt/alpaca/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/alpaca/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num}


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/alpaca/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num}


cd ../../alpaca

CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/${path_after_slash}_vaccine_${RHO}   \
	--model_folder ${model_path} \
	--output_path ../data/alpaca/${path_after_slash}_vaccine_${RHO}.json

source activate python310

export OPENAI_API_KEY=enter your key here!
export IS_ALPACA_EVAL_2=False
alpaca_eval --model_outputs ../data/alpaca/${path_after_slash}_vaccine_${RHO}.json \
   --annotators_config chatgpt	\
   --output_path ../data/alpaca/vaccine_${RHO}

source activate hts
CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/${path_after_slash}_vaccine_${RHO}   \
	--lora_folder2 ../ckpt/alpaca/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num}  \
	--model_folder ${model_path} \
	--output_path ../data/alpaca/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num}.json

source activate python310
export OPENAI_API_KEY=enter your key here!
export IS_ALPACA_EVAL_2=False
alpaca_eval --model_outputs ../data/alpaca/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num}.json \
   --annotators_config chatgpt	\
   --output_path ../data/alpaca/vaccine_f_${RHO}_${poison_ratio}_${sample_num}





wait