
import os
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

pred_len = 96
mask_rate = "0.1"

calculated_seq_len = int(round(pred_len / (1.0 - float(mask_rate))))
seq_len = str(calculated_seq_len)

print(f"pred_len={pred_len}  mask_rate={mask_rate}, seq_len : {seq_len}")

model = "MPA_LLM"
percent = 100
train_epochs = "1"
sample_num = 3000 # number of training samples
llm_model = "gpt2"
Lambda = 4
itr = "1"
features = "300"
command = [
    sys.executable, "run.py",
    "--train_epochs", train_epochs,
    "--itr", itr,
    "--task_name", "forecast",
    "--is_training", "1",
    "--root_path", r"../datasets/net_traffic/GEANT",
    "--data_path", "geant_tm.csv",
    "--label_path", "geant_label.csv",
    "--granularity", "time_step",#["point","time_step"]
    "--model_id", f"geant-feature_{features}_{model}_samplenum{sample_num}",
    "--sample_num", str(sample_num),
    "--llm_model", llm_model,
    "--data", "net_traffic_geant",
    "--seq_len", seq_len,
    "--batch_size", "35",
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
    "--anomaly_ratio", "1.5",
]

subprocess.run(command)