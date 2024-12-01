import torch
from llm2vec import LLM2Vec
from huggingface_hub import login
import pandas as pd
from utils import *

from rec_datasets import *
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, default=50, help='Maximum length of the user interaction sequence')
parser.add_argument('--dataset_name', type=str, default='amazon_video_games',
                    help='amazon_musical_instruments/amazon_industrial_and_scientific/amazon_video_games')
args = parser.parse_args()


def initialize_dataset(args):
    task = 'non-overlap'  # user-overlap, non-overlap
    dataset_name = args.dataset_name
    dataset_path = f'../data/source_files/{dataset_name}.csv'
    dataset_saved_path = f'../data/{dataset_name}'

    if 'amazon' in dataset_name:
        data = pd.read_csv(os.path.join(dataset_saved_path, "processed_data.csv"))
        dataset = AmazonUserSequencesDataset(data, args.max_len)
    else:
        dataset = preprocess(dataset_path, dataset_saved_path, task)
        dataset = popularity_segment(dataset)
        dataset = DoubanDataset(dataset, args.max_len)
    return dataset

if __name__ == '__main__':
    dataset = initialize_dataset(args)
    l2v = LLM2Vec.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    df = pd.DataFrame(dataset.data_frame)
    df = df[['ItemId', 'title', 'description', 'features']]
    df_deduplicated = df.drop_duplicates(subset='ItemId', keep='first')


    def generate_prompt(row):
        title = row['title']
        features = row['features'][1:-1] if isinstance(row['features'], str) and row[
            'features'] else "no feature provided"
        description = row['description'][1:-1] if isinstance(row['description'], str) and row[
            'description'] else "no description provided"
        return f"Please summarize the following item based on the provided information: title: {title}.\n Feature: {features}.\n Description: {description}"


    df_deduplicated['prompt'] = df_deduplicated.apply(generate_prompt, axis=1)

    df_deduplicated['item_text_embedding'] = l2v.encode(list(df_deduplicated['prompt'])).numpy().tolist()
    df_deduplicated = df_deduplicated[['ItemId', 'prompt', 'item_text_embedding']]
    df_deduplicated.to_parquet(f'../data/{args.dataset_name}/{args.dataset_name}_embedding_llama3.parquet',
                               compression='snappy')  # Save as Parquet

