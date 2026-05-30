import os
import subprocess
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

pred_len = 96
mask_rate = "0.1"

calculated_seq_len = int(round(pred_len / (1.0 - float(mask_rate))))
seq_len = str(calculated_seq_len)

print(f"pred_len={pred_len}  mask_rate={mask_rate}, seq_len : {seq_len}")

# seq_len = "550"
model = "MPA_LLM"
percent = 100
train_epochs ="2"
sample_num = 1000 #number of training samples
llm_model = "gpt2"
Lambda = 2
itr = "1"
command = [
    sys.executable, "run.py",
    "--train_epochs", train_epochs,
    "--itr", itr,
    "--task_name", "forecast",
    "--is_training", "1",
    "--root_path", r"../datasets/net_traffic/wsdream",
    "--data_path", "wsdream_tm.csv",
    "--model_id", f"wsdream_1_sample_rate_{mask_rate}_{model}_samplenum{sample_num}",
    "--sample_num", str(sample_num),
    "--llm_model", llm_model,
    "--data", "net_traffic_wsdream",
    "--seq_len", seq_len,
    "--batch_size", "20",
    "--learning_rate", "0.001",
    '--mlp', "1",
    "--d_model", "768",
    "--n_heads", "4",
    "--d_ff", "768",
    "--enc_in", "64",     # input of enc_embedding && output of flatten head
    "--dec_in", seq_len,     # feature
    "--c_out", seq_len,       # mlp
    "--Lambda", str(Lambda),
    "--freq", "h",
    "--percent", str(percent),
    "--gpt_layer", "6",
    "--model", model,
    "--patience", "5",
    "--mask_rate", mask_rate,
    "--label_path", "wsdream_label.csv",
    "--anomaly_ratio", "5.6",
]

subprocess.run(command)
