from dataclasses import dataclass, field
from typing import Dict
import pandas as pd
from pandas import DataFrame
from typing import Callable, Optional
from datasets import Dataset
import pyarrow as pa
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .utils import load_default_model, DEFAULT_GENERATE_PROMPT
from peft import PeftModel, PeftConfig
import shutil
import os

def choose_and_reject(row, txt_prompt, basemodel):
    if row['scores_A'] > row['scores_B']:
        return pd.Series({
            'chosen': txt_prompt(row["llmin"], row['llmout_A'], basemodel),
            'rejected': txt_prompt(row["llmin"], row['llmout_B'], basemodel),
            'score_chosen': row['scores_A'],
            'score_rejected': row['scores_B']
        })
    else:
        return pd.Series({
            'chosen':txt_prompt(row["llmin"], row['llmout_B'], basemodel),
            'rejected': txt_prompt(row["llmin"], row['llmout_A'], basemodel),
            'score_chosen': row['scores_B'],
            'score_rejected': row['scores_A']
        })

@dataclass
class CustomDPOTrainer:
    """implements sft finetuning ontop of dataframe with textual features and labels.
        modular, in memory only.
     
        usage:
        sft_trainer = CustomSFTTrainer()
        sft_trainer.model, sft_trainer, dataset_train = sft_trainer.train()
        to instantiate with default values, make sure you have access to Mathstral-7B-v0.1
       
       """
    dataset: DataFrame = None
    model: Optional[PeftModel] = None
    lora: Optional[PeftConfig] = None
    bnb: Optional[BitsAndBytesConfig] = None
    tokenizer: Optional[AutoTokenizer] = None
    learning_rate: float = 0.000001
    num_train_epochs: float = 3.0
    temp_disk_dir: str = "tmpsave"

    def __post_init__(self):
        # load mathstral 7B by default
        if self.model is None or self.tokenizer is None or self.lora is None:
            print("model or tokenizer or bnb or lora not set, loading default model")
            model, tokenizer, lora, bnb = load_default_model()
            self.model = model
            self.tokenizer = tokenizer
            self.lora = lora
            self.bnb = bnb

        if self.dataset is None:
            print("dataset not set, using dummy dataset")
            self.dataset = pd.DataFrame.from_dict({'scores_B': {0: 0.9772725, 1: 0.9906707855622616, 2: 0.9126307933008319, 3: 0.920136996612189, 4: 0.9275671783308813, 5: 0.9391838316331099, 6: 0.9565226974956044, 7: 0.03180212014134275, 8: 0.7857145, 9: 0.9749496722771148}, 'scores_A': {0: 0.0181818, 1: 0.97124399322922, 2: 0.7026983256306951, 3: 0.6877412774087206, 4: 0.9178399287367209, 5: 0.6694047919323092, 6: 0.783029159645343, 7: 0.0, 8: 0.0, 9: 0.9648792663063118}, 'llmout_B': {0: ' --nroot 3 --nchildroot 3 --n0 0 --nchild0 2 --n00 6 --c00 0 --p00 0.991 --n01 5 --a01 0 --rwm01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 2 --p10 0.991 --n11 5 --a11 1 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 1 --p20 0.0417 --n21 5 --a21 1 --p21 0', 1: ' --nroot 3 --nchildroot 3 --n0 0 --nchild0 2 --n00 6 --c00 2 --p00 0.9121 --n01 5 --a01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 5 --p10 0.9913 --n11 5 --a11 0 --rwm11 1 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 4 --p20 2 --w20 1.4111 --n21 5 --a21 1 --p21 0', 2: ' --nroot 3 --nchildroot 3 --n0 0 --nchild0 2 --n00 6 --c00 2 --p00 0.991 --n01 5 --a01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 0 --p10 0.994 --n11 5 --a11 0 --rwm11 1 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 4 --p20 2 --w20 12.2625 --n21 5 --a21 4 --att21 3.9916 --p21 0', 3: ' --nroot 3 --nchildroot 3 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.991 --n01 5 --a01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 0 --p10 0.994 --n11 5 --a11 0 --rwm11 1 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 5 --p20 0.0231 --n21 5 --a21 4 --att21 4.107 --p21 0', 4: ' --nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.9911 --n01 5 --a01 5 --rep01 4.2016 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 2 --p10 0.9911 --n11 5 --a11 1 --p11 0', 5: ' --nroot 3 --nchildroot 3 --n0 0 --nchild0 2 --n00 6 --c00 2 --p00 0.991 --n01 5 --a01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 5 --p10 0.0435 --n11 5 --a11 1 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 0 --p20 0.9992 --n21 5 --a21 3 --p21 0', 6: ' --nroot 3 --nchildroot 3 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.991 --n01 5 --a01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 0 --p10 0.9943 --n11 5 --a11 0 --rwm11 1 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 1 --p20 0.9997 --n21 5 --a21 1 --p21 0', 7: ' --nroot 3 --nchildroot 3 --n0 0 --nchild0 2 --n00 6 --c00 0 --p00 0.9121 --n01 5 --a01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 0 --p10 0.0804 --n11 5 --a11 1 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 1 --p20 0.0204 --n21 5 --a21 4 --att21 4.3236 --p21 0', 8: ' --nroot 3 --nchildroot 3 --n0 0 --nchild0 2 --n00 6 --c00 2 --p00 0.991 --n01 5 --a01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 0 --p10 0.994 --n11 5 --a11 0 --rwm11 1 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 4 --p20 2 --w20 14.1222 --n21 5 --a21 1 --p21 0', 9: ' --nroot 3 --nchildroot 3 --n0 0 --nchild0 2 --n00 6 --c00 2 --p00 0.991 --n01 5 --a01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 5 --p10 0.0412 --n11 5 --a11 3 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 0 --p20 0.9997 --n21 5 --a21 1 --p21 0'}, 'llmout_A': {0: '--nroot 3 --nchildroot 3 --n0 0 --nchild0 2 --n00 6 --c00 4 --p00 9 --w00 18.8612 --n01 5 --a01 4 --att01 4.1901 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 3 --p10 4 --w10 9.8644 --n11 5 --a11 4 --att11 4.9554 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 1 --p20 0.9925 --n21 5 --a21 2 --p21 0', 1: '--nroot 3 --nchildroot 3 --n0 0 --nchild0 2 --n00 6 --c00 4 --p00 1 --w00 12.2367 --n01 5 --a01 3 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 0 --p10 0.8798 --n11 5 --a11 2 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 0 --p20 0.8012 --n21 5 --a21 4 --att21 4.8458 --p21 0', 2: '--nroot 3 --nchildroot 3 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.8049 --n01 5 --a01 3 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 2 --p10 0.7396 --n11 5 --a11 2 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 2 --p20 0.2449 --n21 5 --a21 1 --p21 0', 3: '--nroot 3 --nchildroot 4 --n0 0 --nchild0 2 --n00 6 --c00 1 --p00 0.9059 --n01 5 --a01 3 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 0 --p10 0.8958 --n11 5 --a11 2 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 5 --p20 0.8987 --n21 5 --a21 4 --att21 1.3327 --p21 0 --n3 0 --nchild3 2 --n30 6 --c30 5 --p30 0.7488 --n31 5 --a31 1 --p31 0', 4: '--nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 4 --p00 9 --w00 14.2694 --n01 5 --a01 3 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 4 --p10 4 --w10 13.1547 --n11 5 --a11 5 --rep11 4.5423 --p11 0', 5: '--nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 0 --p00 0.7575 --n01 5 --a01 4 --att01 4.6825 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 4 --p10 1 --w10 2.8268 --n11 5 --a11 5 --rep11 4.3669 --p11 0', 6: '--nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 1 --p00 0.9178 --n01 5 --a01 5 --rep01 2.9882 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 1 --p10 0.217 --n11 5 --a11 1 --p11 0', 7: '--nroot 3 --nchildroot 4 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.4707 --n01 5 --a01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 3 --p10 1 --w10 14.4727 --n11 5 --a11 0 --rwm11 1 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 4 --p20 3 --w20 9.1143 --n21 5 --a21 4 --att21 4.8081 --p21 0 --n3 0 --nchild3 2 --n30 6 --c30 4 --p30 1 --w30 3.7532 --n31 5 --a31 3 --p31 0', 8: '--nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 1 --p00 0.3544 --n01 5 --a01 1 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 2 --p10 0.0882 --n11 5 --a11 0 --rwm11 2 --p11 0', 9: '--nroot 3 --nchildroot 1 --n0 0 --nchild0 2 --n00 6 --c00 2 --p00 0.9751 --n01 5 --a01 2 --p01 0'}, 'llmin': {0: "The area is a rectangle with dimensions 2.14 x 5.58 x 2.08.In this terrain, 1 lights are arranged, located at coordinates: ((0.01, -0.39)).22 robots are evenly distributed around the origin within a radius of 0.75 m. In the arena, you'll find two areas: a circle at [-0.65, 0.53] with a radius of 0.32 meters in black, and another circle at [0.01, -0.39] with a radius of 0.49 meters in white. The robots' goal is to group together at the white circle. ", 1: 'In this setting, a circular arena with a radius of 2.32 meters is established. Evenly distributed throughout the environment are 2 lights. Their positions are ((-1.58, -0.42), (0.83, -0.48)). 5 robots are evenly spaced around the central point, spanning a radius of 1.13 m. To connect both circles from black to white is the objective for the robots, maintaining a distance just under 0.28 m. Imagine two circles on the floor—one centered at [1.18, 0.46] with a radius of 0.59 meters, in white, and another at [0.60, -0.30] with a radius of 0.54 meters in black. ', 2: 'The circular arena, having a radius of 3.11 meters, is constructed with 11 walls. The arena is illuminated by 2 lights evenly distributed across the space. Within a 1.90-meter radius around the center, 14 robots are evenly positioned. Imagine two circles on the floor—one centered at [-0.85, -0.67] with a radius of 0.39 meters, in black, and another at [-1.78, 0.33] with a radius of 0.91 meters in white. The challenge for the robots is to connect both circles from black to white, keeping a distance just under 0.16 m. ', 3: 'The environment is a circle made out of 21 walls. At each circle, a light is positionedWithin a 1.07-meter radius around the center, 18 robots are evenly positioned. There is a circle at [0.65, 1.50] with a radius of 0.28 meters in black, and another circle at [1.75, -0.72] with a radius of 0.85 meters in white. The objective for the robots is to form a line from the white to the black circle, so that they connect both circles. While forming a line, the robots should keep a distance of just under 0.48 m. The robots with neighbors below this range count as connected. ', 4: 'A rectangular area, with a length of 1.49 meters, width of 1.55 meters, and height of 2.27 meters, is established.At each corner of a rectangle, a light is placed. It has 1/1.33 the size of the dimensions the robots should distribute within.Within a 0.55-meter radius from the center, 22 robots are uniformly distributed. The objective for the swarm is to cover an area of 0.75 by 1.03, while staying connected to each other. The swarm counts as connected if every robot is transitively connected to each other robot in the swarm. Two robots are connected if their distance is at or below 0.14 m. ', 5: 'The environment consists of a rectangular area with length 2.49, width 3.07, and height 1.38.In the arena, 1 lights are evenly spread out with intensities 5.55. 21 robots are evenly placed around the center, covering a radius of 0.94 meters. The goal for the robots is to create a connection from the white to the black circle, maintaining a distance just under 0.45 m. In the floor space, two circles stand out—one at [-0.53, 0.20] with a radius of 0.60 meters, colored in white, and another at [0.17, 0.28] with a radius of 0.39 meters in black. ', 6: "The environment consists of a circular arena with radius 2.78, made out of 20 walls. The environment is 2.01 high. The arena features 1 lights: (1.92, 1.28, 4.80). 11 robots are evenly spaced around the central point, spanning a radius of 1.72 m. The swarm's mission is to cover an area of 2.39 meters by 2.06 meters while staying connected. Connectivity is defined such that every robot is transitively connected to each other, and two robots are connected if their distance is at or below 0.34 meters. ", 7: 'The environment features a rectangle with dimensions 1.73 x 5.62 x 1.60.Evenly distributed throughout the environment are 3 lights. Their positions are ((2.56, 0.41), (2.66, -0.41), (2.47, 0.14)). 5 robots are evenly distributed around the origin within a radius of 0.61 m. Present in the space are two circles—one situated at [0.21, 0.45] with a radius of 0.24 meters, adorned in black, and another at [0.84, 0.61] with a radius of 0.22 meters in white. The robots are tasked with transporting items from the black origing to the white circle. ', 8: 'The rectangular space is 3.11 m long, 1.76 m wide, and 1.25 m high.In the arena, 1 lights are evenly spread out with intensities 5.43. 7 robots are evenly spaced around the central point, spanning a radius of 0.79 m. The goal is for the robots to aggregate at the white circle. There are two areas on the floor: a circle at [-0.66, -0.78] with a radius of 0.38 meters in white, and another circle at [-0.33, -0.93] with a radius of 0.32 meters in black. ', 9: 'The environment features a rectangle with dimensions 2.07 x 4.48 x 1.63.The following lights are present in the arena at coordinates: ((-0.80, 0.53), (0.95, 0.58)).Uniformly distributed are 13 robots within a radius of 0.64 meters. There is a circle at [0.95, 0.58] with a radius of 0.36 meters in black, and another circle at [-0.80, 0.53] with a radius of 0.45 meters in white. The objective for the robots is to form a line from the white to the black circle, so that they connect both circles. While forming a line, the robots should keep a distance of just under 0.13 m. The robots with neighbors below this range count as connected. '}})


    def train(self):

        model = self.model
        # #     # Define the adapter names
        # adapter_names = ["train_adapt", "reference_adapt"]
        
        # # Get the existing adapter names
        # existing_adapters = model.peft_config.keys()
        # # Check and add adapters if they do not exist
        # for adapter_name in adapter_names:
        #     if adapter_name not in existing_adapters:
        #         model.add_adapter(adapter_name, self.lora)
        #         print(f"Added adapter: {adapter_name}")
        #     else:
        #         print(f"Adapter already exists: {adapter_name}")

        dataset = self.dataset.apply(lambda r: choose_and_reject(r, DEFAULT_GENERATE_PROMPT, self), axis=1)
        train_dataset = Dataset(pa.Table.from_pandas(dataset))

        # max prompt and completion length are indispensible for not crashing. gradient_checkpointing does reduce memory significantly but not really improving loss consistently (also conflicts with potential requirement cache=False in model loading)
        training_args = DPOConfig(output_dir="DPO", report_to="none", model_adapter_name="train_adapt", ref_adapter_name="reference_adapt", per_device_train_batch_size=1, per_device_eval_batch_size=1, logging_dir="logs",logging_steps=10,gradient_accumulation_steps=1,eval_accumulation_steps=1, max_prompt_length=300,max_completion_length=300, learning_rate=self.learning_rate, num_train_epochs=self.num_train_epochs) 
        trainer = DPOTrainer(model=model, args=training_args, processing_class=self.tokenizer, train_dataset=train_dataset)
        trainer.train()
        
        dpo_trainer = trainer
        #shutil.rmtree(self.temp_disk_dir) if os.path.exists(self.temp_disk_dir) else None
        
        #merge adapter
        #model = dpo_trainer.model.merge_and_unload()

        #print(f"Merged adapter: {adapter_name}")
        # for adapter_name in adapter_names:
        #     model.remove_adapter(adapter_name)
        #     print(f"Removed adapter: {adapter_name}")

        return dpo_trainer.model, dpo_trainer, dataset