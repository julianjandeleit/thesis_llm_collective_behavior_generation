# %%
from pipeline import MLPipeline#, generate_prompt

pipeline = MLPipeline()

sft_config_params = pipeline.sft_default_config
sft_config_params["num_train_epochs"] = 16
sft_config_params["learning_rate"] = 2e-5
sft_config_params["warmup_ratio"] = 0.10
sft_config_params["weight_decay"] = 0.123
sft_config_params["max_grad_norm"] = 0.15
sft_config_params["gradient_accumulation_steps"] = 2
sft_config_params["max_seq_length"] = 1000

DFNAME = "automode_evaluated_concat_s14n600_s15n600"

stx = """
root node (--nroot), composite nodes (--n0), condition nodes (--c00), action nodes (--a01), decorator nodes (--p00), repeater nodes (--rep01), random selector nodes (--p00), weighted random nodes (--b00), wait nodes (--w00), repeater with max nodes (--rwm01), action with time nodes (--att01).
"""
def txt_prompt(llmin, llmout, tokenizer):
        #f"\nNUMNODES={int(len(llmout.split(' '))/2.0)}\n"+
        # f"\nsyntax example: {stx}\n"
        # Specify the tree inside |BTSTART|<TREE>|BTEND| by starting the tree with --nroot.
        messages = [
            {"role": "user", "content": llmin+"\nGenerate the behavior tree that achieves the objective of this mission."},
            {"role": "assistant", "content": llmout},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, truncation=True, return_dict=False) # wraps text with special tokens depending on role (assitant or user)
        return text
def generate_prompt(sample, tokenizer):
        return txt_prompt(sample["llm_input"],sample["llm_output"], tokenizer)

nepochs=sft_config_params["num_train_epochs"]
import datetime
current_date = datetime.date.today()
current_date_str = current_date.strftime("%Y-%m-%d")

EXP_NAME = f"demo_train_{current_date_str}_{nepochs}_{DFNAME}_notbtstartend"
#EXP_NAME = "demo_train_420_overfit_on_training"
#EXP_NAME = "logs/checkpoint-116000"
#pipeline.train_pipeline(f"../ressources/{DFNAME}.pickle", generate_prompt, EXP_NAME, sft_config_params)

#%%
pipeline.prepare_model()

# %%
#inf_text = """[INST] The environment is a circle made out of 15 walls. The space is lit with 2 lights evenly distributed. Positions are ((-1.13, -1.42), (-1.85, 1.74)). Within a 1.80-meter radius from the center, 10 robots are uniformly distributed. The goal is for the robots to aggregate at the white circle. There are two areas on the floor: a circle at [-2.36, -0.88] with a radius of 1.26 meters in white, and another circle at [-0.30, -0.23] with a radius of 1.36 meters in black. 
#Generate the behavior tree that achieves the objective of this mission.[/INST]"""
# txt = "<s>[INST] The area is a rectangle with dimensions <sn>1.2<mn>5.82<en> x <sn>1.2<mn>5.71<en> x <sn>1.2<mn>2.12<en>.In the arena, <sn>1<mn>1<en> lights are evenly spread out with intensities <sn>1.2<mn>5.93<en>. Around the central point, <sn>2<mn>12<en> robots are positioned uniformly within a <sn>1.2<mn>1.75<en>-meter radius. Visualize two circles—one at [-<sn>1.2<mn>1.18<en>, <sn>1.2<mn>2.04<en>] with a radius of <sn>1.2<mn>0.80<en> meters, adorned in white, and another at [<sn>1.2<mn>2.09<en>, <sn>1.2<mn>0.49<en>] with a radius of <sn>1.2<mn>1.36<en> meters, donned in black. The robots' goal is to create a connection from the white to the black circle, maintaining a distance just under <sn>1.2<mn>0.17<en> m. \nGenerate the behavior tree that achieves the objective of this mission.[/INST] --nroot <sn>1<mn>3<en> --nchildroot <sn>1<mn>4<en> --n<sn>1<mn>0<en> <sn>1<mn>0<en> --nchild<sn>1<mn>0<en> <sn>1<mn>2<en> --n<sn>2<mn>00<en> <sn>1<mn>6<en> --c<sn>2<mn>00<en> <sn>1<mn>0<en> --p<sn>2<mn>00<en> <sn>1.4<mn>0.0776<en> --n<sn>2<mn>01<en> <sn>1<mn>5<en> --a<sn>2<mn>01<en> <sn>1<mn>2<en> --p<sn>2<mn>01<en> <sn>1<mn>0<en> --n<sn>1<mn>1<en> <sn>1<mn>0<en> --nchild<sn>1<mn>1<en> <sn>1<mn>2<en> --n<sn>2<mn>10<en> <sn>1<mn>6<en> --c<sn>2<mn>10<en> <sn>1<mn>2<en> --p<sn>2<mn>10<en> <sn>1.4<mn>0.9132<en> --n<sn>2<mn>11<en> <sn>1<mn>5<en> --a<sn>2<mn>11<en> <sn>1<mn>4<en> --att<sn>2<mn>11<en> <sn>1.4<mn>1.5317<en> --p<sn>2<mn>11<en> <sn>1<mn>0<en> --n<sn>1<mn>2<en> <sn>1<mn>0<en> --nchild<sn>1<mn>2<en> <sn>1<mn>2<en> --n<sn>2<mn>20<en> <sn>1<mn>6<en> --c<sn>2<mn>20<en> <sn>1<mn>2<en> --p<sn>2<mn>20<en> <sn>1.4<mn>0.3796<en> --n<sn>2<mn>21<en> <sn>1<mn>5<en> --a<sn>2<mn>21<en> <sn>1<mn>2<en> --p<sn>2<mn>21<en> <sn>1<mn>0<en> --n<sn>1<mn>3<en> <sn>1<mn>0<en> --nchild<sn>1<mn>3<en> <sn>1<mn>2<en> --n<sn>2<mn>30<en> <sn>1<mn>6<en> --c<sn>2<mn>30<en> <sn>1<mn>5<en> --p<sn>2<mn>30<en> <sn>1.4<mn>0.4948<en> --n<sn>2<mn>31<en> <sn>1<mn>5<en> --a<sn>2<mn>31<en> <sn>1<mn>5<en> --rep<sn>2<mn>31<en> <sn>1.4<mn>2.3877<en> --p<sn>2<mn>31<en> <sn>1<mn>0<en></s>"
# inf_text = "<s>[INST] A circle with 4 walls forms the structure of the environment. There are 0 lights distributed evenly in the arena. Placed within a 1.01-meter radius around the center are 15 robots. The robots' goal is to meet at the black circle. In the arena, you'll find two areas: a circle at [0.14, -1.37] with a radius of 0.38 meters and another circle at [-0.14, 1.43] with a radius of 0.30 meters. Generate the behavior tree that achieves the objective of this mission.[/INST]" 
# output_txt = pipeline.inference(inf_text, EXP_NAME,seq_len=2000)
# output_txt

# %%
# print("------- res 1 --------")
# print(output_txt)



# %%
import pandas as pd
df = pd.read_pickle(f"../ressources/{DFNAME}.pickle")
#df = pd.read_pickle(f"../ressources/automode_evaluated_seed21_n600.pickle")
index = 55
#txt = "[INST]"+df.iloc[index].description+"\n Generate the behavior tree that achieves the objective of this mission.[\INST]"
txt = txt_prompt(df.iloc[index]["description"], "", pipeline.tokenizer)[:-5]
#%%

bt_target = df.iloc[index]["behavior_tree"]
argos = df.iloc[index]["argos"]
output_txt = pipeline.inference(txt, EXP_NAME, seq_len=2000)
print("-------- res val --------")
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
