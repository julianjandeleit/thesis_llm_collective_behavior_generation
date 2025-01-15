# %%
from regex import D
from pipeline import MLPipeline#, generate_prompt
from pipeline.sft import DEFAULT_SFT_CONFIG

pipeline = MLPipeline()

sft_config_params = DEFAULT_SFT_CONFIG
sft_config_params["num_train_epochs"] = 12
sft_config_params["learning_rate"] = 2e-5
sft_config_params["warmup_ratio"] = 0.10
sft_config_params["weight_decay"] = 0.123
sft_config_params["max_grad_norm"] = 0.15
sft_config_params["gradient_accumulation_steps"] = 2
sft_config_params["max_seq_length"] = 1000

DFNAME = "automode_evaluated_concat_s14-s18_24-12-23"

from pipeline.utils import DEFAULT_GENERATE_PROMPT

nepochs=sft_config_params["num_train_epochs"]
import datetime
current_date = datetime.date.today()
current_date_str = current_date.strftime("%Y-%m-%d")

EXP_NAME = f"demo_train_{current_date_str}_{nepochs}_{DFNAME}_test"

_trained_model, hf_trainer, dataset = pipeline.train_sft(f"../ressources/{DFNAME}.pickle", DEFAULT_GENERATE_PROMPT, EXP_NAME, sft_config_params)


# %%
inf_text = """The environment is a circle made out of 15 walls. The space is lit with 2 lights evenly distributed. Positions are ((-1.13, -1.42), (-1.85, 1.74)). Within a 1.80-meter radius from the center, 10 robots are uniformly distributed. The goal is for the robots to aggregate at the white circle. There are two areas on the floor: a circle at [-2.36, -0.88] with a radius of 1.26 meters in white, and another circle at [-0.30, -0.23] with a radius of 1.36 meters in black. 
Generate the behavior tree that achieves the objective of this mission. """
output_txt = pipeline.inference(_trained_model, hf_trainer.tokenizer, inf_text, seq_len=2000)
print("------- fixed res --------")
print(output_txt)



# # %%
import pandas as pd
index = 0
df = pd.read_pickle(f"../ressources/{DFNAME}.pickle")
txt = df.iloc[index]["description"]
#%%
output_txt = pipeline.inference(_trained_model, hf_trainer.tokenizer,txt, seq_len=2000)
print("-------- res var --------")
print(output_txt)
# %%
import pickle
with open(EXP_NAME+"/loss_history.pkl", 'rb') as file:
    loss_history = pickle.load(file)
# %%
# Initialize lists to store losses and epochs
train_losses = []
eval_losses = []
train_epochs = []
eval_epochs = []
final_metrics = {}
# Extract losses and corresponding epochs
import matplotlib.pyplot as plt
for entry in loss_history:
    if 'loss' in entry:
        train_losses.append(entry['loss'])
        train_epochs.append(entry['epoch'])
    if 'eval_loss' in entry:
        eval_losses.append(entry['eval_loss'])
        eval_epochs.append(entry['epoch'])
    if 'train_runtime' in entry:
        final_metrics['train_runtime'] = entry['train_runtime']
        final_metrics['train_samples_per_second'] = entry['train_samples_per_second']
        final_metrics['train_steps_per_second'] = entry['train_steps_per_second']
        final_metrics['total_flos'] = entry['total_flos']


# Ensure that the lengths of train_losses and eval_losses match their respective epochs
print("Training Losses:", train_losses)
print("Training Epochs:", train_epochs)
print("Evaluation Losses:", eval_losses)
print("Evaluation Epochs:", eval_epochs)

# Plotting
plt.figure(figsize=(10, 5))

# Plotting training loss
plt.plot(train_epochs, train_losses, label='Training Loss', marker='o')

# Plotting evaluation loss
plt.plot(eval_epochs, eval_losses, label='Evaluation Loss', marker='o')

plt.title('Training and Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.text(4, 1.5, f"Final Metrics:\n"
                    f"Train Runtime: {final_metrics['train_runtime']:.2f} seconds\n"
                    f"Train Samples/sec: {final_metrics['train_samples_per_second']:.2f}\n"
                    f"Train Steps/sec: {final_metrics['train_steps_per_second']:.2f}\n"
                    f"Total FLOPs: {final_metrics['total_flos']:.2e}\n"             ,
         fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

plt.legend()
plt.grid()
plt.savefig(EXP_NAME+'/training_evaluation_loss_plot.png', bbox_inches='tight')
#plt.show()
# %%
