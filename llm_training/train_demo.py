# %%
from pipeline import MLPipeline, generate_prompt

pipeline = MLPipeline()

sft_config_params = pipeline.sft_default_config
sft_config_params["num_train_epochs"] = 50
sft_config_params["learning_rate"] = 0.002

EXP_NAME = "basic_demo"
pipeline.train_pipeline("../ressources/automode_descriptions_evaluated.pickle", generate_prompt, EXP_NAME, sft_config_params)


# %%
inf_text = """[INST] The environment is a circle made out of 15 walls. The space is lit with 2 lights evenly distributed. Positions are ((-1.13, -1.42), (-1.85, 1.74)). Within a 1.80-meter radius from the center, 10 robots are uniformly distributed. The goal is for the robots to aggregate at the white circle. There are two areas on the floor: a circle at [-2.36, -0.88] with a radius of 1.26 meters in white, and another circle at [-0.30, -0.23] with a radius of 1.36 meters in black. 
Generate the behavior tree that achieves the objective of this mission.[/INST]"""
output_txt = pipeline.inference(inf_text, EXP_NAME)
output_txt

# %%
print(output_txt)


