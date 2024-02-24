import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import sys

def main(content: str):
    torch.set_default_device('cuda')
    # quantization_config = BitsAndBytesConfig(load_in_4bit = True)
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
                                                 # quantization_config=quantization_config,
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype="auto",
                                                 device_map="auto") 
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    chat = [
        {
            "role": "user",
            "content": content,
        },
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_genereation_prompt=True)
    print(f"prompt: {prompt}")

    inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=512)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(f"response: {text}")

if __name__ == '__main__':
    main(sys.argv[1])
