import os
import subprocess
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

seq_len = "550"
model = "MPA_LLM"
percent = 100
mask_rate = "0.1"
train_epochs = "2"
llm_model = "gpt2"
d_model = "768"
Lambda = 2
itr = "1"
sample_num = 1000

command = [
    sys.executable, "run.py",
    "--train_epochs", train_epochs,
    "--itr", itr,
    "--task_name", "imputation",
    "--is_training", "1",
    "--root_path", r"../datasets/net_traffic/wsdream",
    "--data_path", "wsdream_tm.csv",
    "--label_path", "wsdream_label.csv",
    "--granularity", "time_step",  # ["point","time_step"]
    "--model_id", f"Wsdream-few-shot-_rate{mask_rate}_{model}_samplenum{sample_num}_seq_{seq_len}",
    "--sample_num", str(sample_num),
    "--llm_model", llm_model,
    "--data", "net_traffic_wsdream",
    "--seq_len", seq_len,
    "--batch_size", "3",
    "--learning_rate", "0.001",
    '--mlp', "1",
    "--d_model", d_model,
    "--n_heads", "4",
    "--d_ff", d_model,
    "--enc_in", "64",
    "--dec_in", seq_len,
    "--c_out", seq_len,
    "--Lambda", str(Lambda),
    "--freq", "h",
    "--percent", str(percent),
    "--gpt_layer", "6",
    "--model", model,
    "--patience", "5",
    "--mask_rate", mask_rate,
    "--anomaly_ratio", "4.6",
]

subprocess.run(command)