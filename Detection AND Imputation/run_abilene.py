import os
import subprocess
import sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

seq_len ="50"
model = "MPA_LLM"
percent = 100
mask_rate = "0.1"
train_epochs ="1"
sample_num = 700 #number of training samples
llm_model = "gpt2"
Lambda = 2
itr = "1"
features = "144"
command = [
    sys.executable, "run.py",
    "--train_epochs", train_epochs,
    "--itr", itr,
    "--task_name", "imputation",
    "--is_training", "1",
    "--root_path", r"../datasets/net_traffic/Abilene",
    "--data_path", "abilene_tm.csv",
    "--label_path", "abilene_label.csv",
    "--granularity", "time_step",#["point","time_step"]
    "--model_id", f"Abilene—few-shot-_rate{mask_rate}_{model}_samplenum{sample_num}_seq_{seq_len}",
    "--sample_num", str(sample_num),
    "--llm_model", llm_model,
    "--data", "net_traffic_abilene",
    "--seq_len", str(seq_len),
    "--batch_size", "3",
    "--learning_rate", "0.001",
    '--mlp', "1",
    "--d_model", "768",
    "--n_heads", "4",
    "--d_ff", "768",
    "--enc_in", seq_len,     # input of enc_embedding && output of flatten head
    "--dec_in", features,     # feature
    "--c_out", features,       # mlp
    "--Lambda", str(Lambda),
    "--freq", "h",
    "--percent", str(percent),
    "--gpt_layer", "6",
    "--model", model,
    "--patience", "5",
    "--mask_rate", mask_rate,
    "--anomaly_ratio", "25",
]

subprocess.run(command)
