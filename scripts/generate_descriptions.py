import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
from swarm_descriptions.configfiles import ET, Configurator
from swarm_descriptions.mission_elements import get_generators, MissionParams
from swarm_descriptions.configfiles import config_to_string
from swarm_descriptions.utils import truncate_floats
import random
import pyarrow as pa
import pyarrow.dataset as ds
from tqdm.auto import tqdm
import argparse
import pathlib
import logging

def sample_dataset(n_rows = 10000, generators = get_generators()) -> pd.DataFrame:
    rows = []
    for _n in tqdm(range(n_rows)):
        mission = MissionParams.sample(*generators)
        conf = config_to_string(mission.configure())
        conf = truncate_floats(conf)
        desc = random.sample(mission.describe(),1)[0]  
        desc = truncate_floats(desc)
        if _n == 583:
            print(mission)
        rows.append({"description": desc, "configuration": conf, "parameters": mission})
    dataset = pd.DataFrame(rows)
    
    return dataset

def arg_to_loglevel(choice):
    if choice == "critical":
        return logging.CRITICAL
    if choice == "info":
        return logging.INFO
    if choice == "debug":
        return logging.DEBUG
    return logging.INFO

def config_params_to_argos_config(params: str, skeleton: ET.ElementTree):
    xml = ET.fromstring(params)
    config_tree = Configurator().convert_config_params(params=xml, skeleton_root=skeleton)
    config = config_tree.getroot()
    config = config_to_string(config)
    xml = ET.fromstring(params)
    config_tree = Configurator().convert_config_params(params=xml, skeleton_root=skeleton)
    config = config_tree.getroot()
    config = config_to_string(config)
    return config

def add_config_to_dataset(df: pd.DataFrame, skeleton: ET.ElementTree):
    result = []
    for config_params in df["configuration"]:
        argos_config = config_params_to_argos_config(config_params, skeleton)        
        result.append(argos_config)
    df["argos"] = result
    return df  
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sample and save dataset. e.g. ')

    parser.add_argument('n_rows', type=int, help="number of rows to sample")
    parser.add_argument('output', type=pathlib.Path, default=None, help="output path of generated dataset. will be pickle file.")
    parser.add_argument(
        "--logging", choices=["critical", "info", "debug"], default="info")
    parser.add_argument(
        "--seed", type=int, default=42)
    parser.add_argument("--template", type=pathlib.Path, default=None, help="if set, converts the configuration to actual argos config file")
    
    args = parser.parse_args()
    logging.basicConfig(level=arg_to_loglevel(args.logging))
    
    if args.seed is not None:
        logging.info(f"setting seed {args.seed}")
        np.random.seed(args.seed)
        random.seed(args.seed)

    df = sample_dataset(args.n_rows)

    if args.template:
        skeleton = ET.parse(args.template)
        df = add_config_to_dataset(df, skeleton)

    if args.output:
        df.to_pickle(args.output)
