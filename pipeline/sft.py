from dataclasses import dataclass, field
from typing import Dict
import pandas as pd
from pandas import DataFrame
from typing import Callable, Optional
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pyarrow as pa
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import load_default_model, DEFAULT_GENERATE_PROMPT

DEFAULT_SFT_CONFIG= {
                "max_seq_length": 1000,
                "num_train_epochs": 12,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "optim": "paged_adamw_32bit",
                "save_steps": 0,
                "logging_steps": 25,
                "learning_rate": 2e-5,
                "weight_decay": 0.123,
                "bf16": False,
                "max_grad_norm": 0.15,
                "max_steps": -1,
                "warmup_ratio": 0.10,
                "group_by_length": True,
                "lr_scheduler_type": "cosine",
                "report_to": "none",
                "eval_strategy": "epoch",
                "output_dir": "logs",
                "evaluation_strategy": "epoch",
                "eval_steps": 1,
                "save_strategy": "epoch",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "dataset_text_field": "text"
            }

@dataclass
class CustomSFTTrainer:
    """implements sft finetuning ontop of dataframe with textual features and labels.
        modular, in memory only.
     
        usage:
        sft_trainer = CustomSFTTrainer()
        sft_trainer.model, sft_trainer, dataset_train = sft_trainer.train()
        to instantiate with default values, make sure you have access to Mathstral-7B-v0.1
       
       """
    dataset: DataFrame = None
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    sft_config: Dict = field(default_factory=lambda: DEFAULT_SFT_CONFIG.copy()) # the keyword arguments that describe the training process of hf sft_trainer 
    test_size: float = 0.2 # percentage of dataset used for test (validation)
    generate_prompt: Callable[[str, str, 'CustomSFTTrainer'], str] = DEFAULT_GENERATE_PROMPT
    llmin_col: str = "llmin"
    llmout_col: str = "llmout"

    def __post_init__(self):
        # load mathstral 7B by default
        if self.model is None or self.tokenizer is None:
            model, tokenizer, lora, bnb = load_default_model()
            self.model = model
            self.tokenizer = tokenizer

        if self.dataset is None:
            self.dataset = pd.DataFrame.from_dict({'llmin': {0: "In this setting, a circular arena with a radius of 2.16 meters is established. The arena is illuminated by 2 lights evenly distributed across the space. Placed within a 1.01-meter radius around the center are 13 robots. In the floor space, you'll discover two distinct areas: a circle at [-1.03, 0.46] in black, and another circle at [-0.50, -1.28] in white. The primary objective for the robots is to aggregate at the WHITE circle. ",
                    1: "The area is a rectangle with dimensions 4.76 x 5.14 x 1.73.There are 0 lights distributed evenly in the arena. 20 robots are evenly spaced around the central point, spanning a radius of 2.08 m. In the environment, you'll find a circle at [-0.08, -1.04] with a radius of 0.34 meters, characterized by its white hue. There's also another circle at [-1.95, 2.38] with a radius of 0.35 meters in black. The robots are assigned the goal of moving items from the black starting zone to the white circle. ",
                    2: 'The environment features a circle composed of 16 walls. Evenly distributed throughout the environment are 3 lights. Their positions are ((2.51, -1.76), (-1.01, -0.21), (1.73, -1.29)). Evenly positioned around the origin are 24 robots within a radius of 1.92 meters. Present in the space are two circles—one situated at [1.08, 0.38] with a radius of 0.26 meters, adorned in black, and another at [1.09, 2.35] with a radius of 0.42 meters in white. The robots are tasked with transporting items from the white origing to the black circle. ',
                    3: "The environment is constructed as a rectangular space with a length of 2.96 meters, width of 2.23 meters, and height of 2.26 meters.There are the following lights in the arena: ((0.66, -1.20), (-0.68, -1.41), (0.01, 0.06)). 8 robots are evenly placed around the center, covering a radius of 1.11 meters. The robots' task is to aggregate at the BLACK circle. There are two floor areas, each defined by a circle. The first circle, located at [-0.60, 1.11], has a radius of 0.33 meters in black. The second circle, positioned at [-0.56, 0.31], has a radius of 0.53 meters in white. ",
                    4: 'The rectangular area has dimensions 3.38 m x 3.50 m x 2.92 m.Evenly distributed throughout the environment are 4 lights. Their positions are ((0.19, 1.39), (1.58, -1.58), (-0.25, 0.93), (1.07, -1.25)). 11 robots are evenly spaced around the central point, spanning a radius of 1.63 m. The goal for the robots is to create a connection from the white to the black circle, maintaining a distance just under 0.30 m. In the floor space, two circles stand out—one at [-0.42, 0.96] with a radius of 0.40 meters, colored in white, and another at [-0.46, -0.25] with a radius of 0.33 meters in black. ',
                    5: 'The environment consists of a circular arena with radius 1.07, made out of 23 walls. The environment is 2.82 high. There are 2 lights distributed evenly in the arena. There are 15 robots placed uniformly around the center within a radius of 0.74 meters. There is a circle at [-0.65, 0.51] with a radius of 0.26 meters in black, and another circle at [-0.68, 0.22] with a radius of 0.27 meters in white. The objective for the robots is to form a line from the black to the white circle, so that they connect both circles. While forming a line, the robots should keep a distance of just under 0.46 m. The robots with neighbors below this range count as connected. ',
                    6: "The area is a rectangle with dimensions 6.76 x 1.97 x 2.77.The arena features 3 lights: (0.56, -0.03, 5.17), (-0.40, 3.19, 3.49), (0.36, 0.99, 2.00). Within a 0.99-meter radius around the center, 12 robots are evenly positioned. The swarm's mission is to cover an area of 1.57 meters by 0.55 meters while staying connected. Connectivity is defined such that every robot is transitively connected to each other, and two robots are connected if their distance is at or below 0.13 meters. ",
                    7: 'With a radius of 2.17 meters, the circular arena is made up of 13 walls. 0 lights are distributed uniformly in the arena. Within a 0.96-meter radius from the center, 14 robots are uniformly distributed. The objective for the swarm is to cover an area of 2.96 by 2.58, while staying connected to each other. The swarm counts as connected if every robot is transitively connected to each other robot in the swarm. Two robots are connected if their distance is at or below 0.92 m. ',
                    8: 'In this setting, a circular arena with a radius of 5.00 meters is established. 4 lights are distributed uniformly in the arena. 13 robots are evenly spaced around the central point, spanning a radius of 2.18 m. In the surroundings, there exists a circle at [2.59, 0.41] with a radius of 0.27 meters, exhibiting a black color, and another at [1.24, 1.32] with a radius of 0.66 meters in white. The objective for the robots is to transfer items from the black initial location to the white circle. ',
                    9: 'The environment features a rectangle with dimensions 5.57 x 2.19 x 1.75.There are the following lights in the arena: ((-0.25, 0.57)). 16 robots are evenly spaced around the central point, spanning a radius of 0.96 m. In the surroundings, there exists a circle at [-0.71, 0.76] with a radius of 0.25 meters, exhibiting a white color, and another at [0.78, -1.25] with a radius of 0.27 meters in black. The objective for the robots is to transfer items from the white initial location to the black circle. '},
                    'llmout': {0: '--nroot 3 --nchildroot 4 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.1642 --n01 5 --a01 4 --att01 4.0508 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 2 --p10 0.1165 --n11 5 --a11 0 --rwm11 4 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 4 --p20 8 --w20 15.1397 --n21 5 --a21 4 --att21 2.3803 --p21 0 --n3 0 --nchild3 2 --n30 6 --c30 1 --p30 0.8248 --n31 5 --a31 1 --p31 0',
                    1: '--nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 3 --p00 2 --w00 13.501 --n01 5 --a01 0 --rwm01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 1 --p10 0.8729 --n11 5 --a11 3 --p11 0',
                    2: '--nroot 3 --nchildroot 4 --n0 0 --nchild0 2 --n00 6 --c00 3 --p00 4 --w00 4.4505 --n01 5 --a01 0 --rwm01 5 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 0 --p10 0.2482 --n11 5 --a11 1 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 3 --p20 8 --w20 7.7693 --n21 5 --a21 2 --p21 0 --n3 0 --nchild3 2 --n30 6 --c30 2 --p30 0.5794 --n31 5 --a31 3 --p31 0',
                    3: '--nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 0 --p00 0.1917 --n01 5 --a01 0 --rwm01 1 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 3 --p10 6 --w10 10.5754 --n11 5 --a11 1 --p11 0',
                    4: '--nroot 3 --nchildroot 4 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.7274 --n01 5 --a01 1 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 2 --p10 0.7147 --n11 5 --a11 2 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 1 --p20 0.6199 --n21 5 --a21 5 --rep21 1.7916 --p21 0 --n3 0 --nchild3 2 --n30 6 --c30 2 --p30 0.9117 --n31 5 --a31 4 --att31 4.1256 --p31 0',
                    5: '',
                    6: '--nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 1 --p00 0.9047 --n01 5 --a01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 4 --p10 3 --w10 16.5383 --n11 5 --a11 3 --p11 0',
                    7: '--nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.8495 --n01 5 --a01 5 --rep01 4.2918 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 4 --p10 5 --w10 12.6365 --n11 5 --a11 1 --p11 0',
                    8: '--nroot 3 --nchildroot 1 --n0 0 --nchild0 2 --n00 6 --c00 3 --p00 4 --w00 7.0854 --n01 5 --a01 0 --rwm01 95 --p01 0',
                    9: '--nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 1 --p00 0.092 --n01 5 --a01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 2 --p10 0.1093 --n11 5 --a11 0 --rwm11 2 --p11 0'}})

    def train(self):  

        # process dataset to format used by trainer

        dataset = pd.DataFrame()
        dataset["text"] = self.dataset.apply(lambda x: self.generate_prompt(x[self.llmin_col], x[self.llmout_col], self), axis=1)

        train_dataset, val_dataset = train_test_split(dataset, test_size=self.test_size)

        generated_train_dataset = Dataset(pa.Table.from_pandas(train_dataset))
        generated_val_dataset = Dataset(pa.Table.from_pandas(val_dataset))

        model = self.model
        train_dataset = generated_train_dataset
        val_dataset  = generated_val_dataset
        tokenizer= self.tokenizer
        hf_sft_conf = SFTConfig(**self.sft_config)

        sft_trainer = SFTTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    tokenizer=tokenizer,
                    args=hf_sft_conf,
                    packing=False,
                )

        sft_trainer.train()

        return sft_trainer.model, sft_trainer, dataset