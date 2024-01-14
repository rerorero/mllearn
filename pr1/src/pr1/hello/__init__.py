from transformers import pipeline
import pandas

def main():
    classifier = pipeline("text-classification")
    outputs = classifier("go to hell")
    print(pandas.DataFrame(outputs))

