import json
import argparse
from pathlib import Path

import datasets
from tqdm import tqdm

def load_dataset(seed: int, num_steps: int) -> list[str]:
    dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'default')
    df = dataset['train'].to_pandas()

    df = df[['text', 'hatespeech']]
    toxic = df[df['hatespeech'] > 0.5]['text'].sample(n=num_steps//2, random_state=seed).to_list()
    non_toxic = df[df['hatespeech'] <= 0.5]['text'].sample(n=num_steps//2, random_state=seed).to_list()

    return non_toxic + toxic

def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(prog='LLM as bandits - hatespeech')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config['seed']
    num_steps = config['num_steps']
    epochs = config['num_epochs']

    data = load_dataset(seed, num_steps)

    for epoch in range(epochs):
        for step, text in enumerate(data):




if __name__ == '__main__':
    main()
