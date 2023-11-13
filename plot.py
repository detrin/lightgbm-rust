import pandas as pd  
import matplotlib.pyplot as plt  
import json  
import os  
import numpy as np  
  

with open('./data/results.json') as f:  
    data = json.load(f)  
  
df = pd.json_normalize(data)  
df = df.drop(["train_input", "test_input"], axis=1)
df = df.groupby(['n_features', 'n_samples']).mean().reset_index()  
df['x_label'] = 'f' + df['n_features'].astype(str) + '_r' + df['n_samples'].astype(str)  
df['load_speedup'] = df['python.load'] / df['rust.load']  
df['train_speedup'] = df['python.train'] / df['rust.train']  
df['inference_speedup'] = df['python.inference'] / df['rust.inference']  
df = df.reset_index()  
  
x = np.arange(len(df['x_label']))  
width = 0.15  
  
# Plot Load  
fig, ax = plt.subplots()  
ax.bar(x - width/2, df['python.load'], width, label='Python')  
ax.bar(x + width/2, df['rust.load'], width, label='Rust')  
ax.set_ylabel('Time (ms)')  
ax.set_title('Load Times')  
ax.set_xticks(x)  
ax.set_xticklabels(df['x_label'], rotation=90)  
ax.legend()  
ax.set_yscale('log')  
ax.grid(axis='y')  
plt.tight_layout()  
plt.savefig(os.path.join('./data', 'average_load_times.png'))  
  
# Plot Train  
fig, ax = plt.subplots()  
ax.bar(x - width/2, df['python.train'], width, label='Python')  
ax.bar(x + width/2, df['rust.train'], width, label='Rust')  
ax.set_ylabel('Time (ms)')  
ax.set_title('Train Times')  
ax.set_xticks(x)  
ax.set_xticklabels(df['x_label'], rotation=90)  
ax.legend()  
ax.set_yscale('log')  
ax.grid(axis='y')  
plt.tight_layout()  
plt.savefig(os.path.join('./data', 'average_train_times.png'))  
  
# Plot Inference  
fig, ax = plt.subplots()  
ax.bar(x - width/2, df['python.inference'], width, label='Python')  
ax.bar(x + width/2, df['rust.inference'], width, label='Rust')  
ax.set_ylabel('Time (ms)')  
ax.set_title('Inference Times')  
ax.set_xticks(x)  
ax.set_xticklabels(df['x_label'], rotation=90)  
ax.legend()  
ax.set_yscale('log')  
ax.grid(axis='y')  
plt.tight_layout()  
plt.savefig(os.path.join('./data', 'average_inference_times.png'))  
  
# Plot Load Speedup  
fig, ax = plt.subplots()  
ax.bar(x, df['load_speedup'], width)  
ax.set_ylabel('Speedup')  
ax.set_title('Load Speedup (Python/Rust)')  
ax.set_xticks(x)  
ax.set_xticklabels(df['x_label'], rotation=90)  
ax.grid(axis='y')  
plt.tight_layout()  
plt.savefig(os.path.join('./data', 'average_load_speedup.png'))  
  
# Plot Train Speedup  
fig, ax = plt.subplots()  
ax.bar(x, df['train_speedup'], width)  
ax.set_ylabel('Speedup')  
ax.set_title('Train Speedup (Python/Rust)')  
ax.set_xticks(x)  
ax.set_xticklabels(df['x_label'], rotation=90)  
ax.grid(axis='y')  
plt.tight_layout()  
plt.savefig(os.path.join('./data', 'average_train_speedup.png'))  
  
# Plot Inference Speedup  
fig, ax = plt.subplots()  
ax.bar(x, df['inference_speedup'], width)  
ax.set_ylabel('Speedup')  
ax.set_title('Inference Speedup (Python/Rust)')  
ax.set_xticks(x)  
ax.set_xticklabels(df['x_label'], rotation=90)  
ax.grid(axis='y')  
plt.tight_layout()  
plt.savefig(os.path.join('./data', 'average_inference_speedup.png'))  

