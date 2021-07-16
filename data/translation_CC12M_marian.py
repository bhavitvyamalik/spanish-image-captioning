## Don't forget to run tmux before starting the script

import argparse
import csv
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate
import gc
import jax
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from transformers import FlaxMarianMTModel, MarianTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--tsv_path", type=str, default="images-list-clean.tsv", help="path of directory where the dataset is stored")
parser.add_argument("--val_split", type=int, default=0.01, help="Size of validation Subset")
parser.add_argument("--save_location_train", type=str, default=".", help="path of directory where the train dataset will be stored")
parser.add_argument("--save_location_val", type=str, default=".", help="path of directory where the validation dataset will be stored")
parser.add_argument("--is_train", type=int, default=0, help="train or validate")
parser.add_argument("--max_size", type=int, default=25000, help="size")


args = parser.parse_args()


DATASET_PATH = args.tsv_path
VAL_SPLIT = args.val_split
if args.save_location_train != None:
    SAVE_TRAIN = args.save_location_train
    SAVE_VAL = args.save_location_val

BATCH_SIZE = 1024
MAX_SIZE = args.max_size
IS_TRAIN = args.is_train
num_devices = 8

model = FlaxMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es", from_pt=True)

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

def generate(params, batch):
      output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], params=params, num_beams=4, max_length=64, early_stopping=True).sequences
      return output_ids

p_generate = jax.pmap(generate, "batch")

p_params = replicate(model.params)

rng = jax.random.PRNGKey(0)
rngs = jax.random.split(rng, num_devices)

def run_generate(input_str):
  inputs = tokenizer(input_str, return_tensors="jax", padding="max_length", truncation=True, max_length=64)
  p_inputs = shard(inputs.data)
  output_ids = p_generate(p_params, p_inputs, rngs)
  output_strings = tokenizer.batch_decode(output_ids.reshape(-1, 64), skip_special_tokens=True)
  return output_strings


def read_tsv_file(tsv_path):
    df = pd.read_csv(tsv_path, delimiter="\t", index_col=False)
    print("Number of Examples:", df.shape[0], "for", tsv_path)
    return df

def arrange_data(image_files, captions, image_urls):  # iterates through all the captions and save there translations
    try:
        lis_ = []

        output = run_generate(captions, p_generate)

        for image_file, caption, image_url in zip(image_files, output, image_urls):  # add other captions
                lis_.append({"image_file":image_file, "caption":caption, "url":image_url})

        gc.collect()
        return lis_

    except Exception as e:
        print(captions, image_url, " skipped!")
        return


_df = read_tsv_file(DATASET_PATH)
train_df, val_df = train_test_split(_df, test_size=VAL_SPLIT, random_state=1234)

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

print("\n train/val dataset created. Beginning translation...")

if IS_TRAIN:
    df = train_df
    output_file_name = os.path.join(SAVE_VAL, "train_file_es.tsv")
    with open(output_file_name, 'w', newline='') as outtsv:  # creates a blank tsv with headers (overwrites existing file)
        writer = csv.writer(outtsv, delimiter='\t')
        writer.writerow(["image_file", "caption", "url"])

else:
    df = val_df
    output_file_name = os.path.join(SAVE_VAL, "val_file_es.tsv")
    with open(output_file_name, 'w', newline='') as outtsv:  # creates a blank tsv with headers (overwrites existing file)
        writer = csv.writer(outtsv, delimiter='\t')
        writer.writerow(["image_file", "caption", "url"])

for i in tqdm(range(0,MAX_SIZE,BATCH_SIZE)):
    output_batch = arrange_data(list(df["image_file"])[i:i+BATCH_SIZE], list(df["caption"])[i:i+BATCH_SIZE], list(df["url"])[i:i+BATCH_SIZE])
    with open(output_file_name, "a", newline='') as f:
      writer = csv.DictWriter(f, fieldnames=["image_file", "caption", "url"], delimiter='\t')
      for batch in output_batch:
          writer.writerow(batch)