from datasets import load_dataset

dataset = load_dataset(
    "wiki_snippets",
    "wikipedia_en_100_0",
    split="train",
    cache_dir="./data/wiki_snippets"
)

print(len(dataset))
print(dataset[0])
