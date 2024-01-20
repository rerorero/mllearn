from datasets import load_dataset
from huggingface_hub import HfApi
import pandas
import torch
import matplotlib.pyplot as plt
from pandas.core.api import DataFrame
from transformers import AutoModel, AutoTokenizer

def main():
    _train()

def _train():
    emotions = load_dataset("emotion")
    # emotions.set_format("pandas")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

    # text = "this is a test"
    # inputs = tokenizer(text, return_tensors="pt")
    # inputs = {k:v.to(device) for k, v in inputs.items()}
    # with torch.no_grad():
    #     outputs = model(**inputs)
    # print(outputs.last_hidden_state.shape)

    emotions_local = load_dataset("csv", data_files="./files/token/train.txt", sep=";", names=["text", "label"])

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    def extract_hidden_state(batch):
        inputs = {k:v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            last_hidden_state = model(**inputs).last_hidden_state
            return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
    
    print(emotions_encoded["train"].column_names)



def _plot():
    emotions = load_dataset("emotion")
    emotions.set_format("pandas")


    df: DataFrame = emotions["train"][:]
    df["label_name"] = df["label"].apply(lambda x: emotions["train"].features["label"].int2str(x))

    df["label_name"].value_counts(ascending=True).plot.barh()
    df["Words Per Tweet"] = df["text"].str.split().apply(len)
    df.boxplot("Words Per Tweet", by="label_name", grid=False, showfliers=False, color="black")

    plt.title("Emotion Distribution")
    plt.suptitle("")
    plt.xlabel("")
    plt.show()

