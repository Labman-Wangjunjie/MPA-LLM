#!/bin/bash

# Set environment variables
export CUDA_LAUNCH_BLOCKING="1"
export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM="false"

# Set variables
pred_len="96"
mask_rate="0.1"

# Dynamically calculate seq_len as in the Python script: int(round(pred_len / (1.0 - mask_rate)))
seq_len=$(awk -v p="$pred_len" -v m="$mask_rate" 'BEGIN { printf "%.0f", p / (1.0 - m) }')

echo "pred_len=$pred_len  mask_rate=$mask_rate, seq_len : $seq_len"

model="MPA_LLM"
percent="100"
train_epochs="2"
sample_num="1000" # number of training samples
llm_model="gpt2"
Lambda="2"
itr="1"

python -u run.py \
    --train_epochs $train_epochs \
    --itr $itr \
    --task_name "forecast" \
    --is_training "1" \
    --root_path "../datasets/net_traffic/wsdream" \
    --data_path "wsdream_tm.csv" \
    --model_id "wsdream_1_sample_rate_${mask_rate}_${model}_samplenum${sample_num}" \
    --sample_num $sample_num \
    --llm_model $llm_model \
    --data "net_traffic_wsdream" \
    --seq_len $seq_len \
    --batch_size "20" \
    --learning_rate "0.001" \
    --mlp "1" \
    --d_model "768" \
    --n_heads "4" \
    --d_ff "768" \
    --enc_in "64" \
    --dec_in $seq_len \
    --c_out $seq_len \
    --Lambda $Lambda \
    --freq "h" \
    --percent $percent \
    --gpt_layer "6" \
    --model $model \
    --patience "5" \
    --mask_rate $mask_rate \
    --label_path "wsdream_label.csv" \
    --anomaly_ratio "5.6"