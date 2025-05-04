# Fine-Tuning LLaMA 2 on Home Remedies Dataset using QLoRA

This project demonstrates how to fine-tune Meta's [LLaMA 2](https://ai.meta.com/llama/) 7B Chat model on a domain-specific dataset – the [Home Remedies Dataset](https://huggingface.co/datasets/medss19/home_remedies_dataset). The goal is to teach the LLaMA model to suggest traditional remedies based on user health concerns.

We use **QLoRA** (Quantized Low-Rank Adaptation), a memory-efficient fine-tuning technique, enabling large model training on limited computing resources (e.g., Colab free-tier GPUs).

## 📂 Repository Structure

* `HomeRemedy_Llama_Dataset.ipynb`: Preprocesses and formats the dataset into the LLaMA2-style prompt template.
* `Fine_tune_Llama2.ipynb`: Fine-tunes the model using QLoRA and evaluates perplexity.

## 🚀 Project Overview

| Component | Description |
|----------|-------------|
| Base Model | [`NousResearch/Llama-2-7b-chat-hf`](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) |
| Fine-Tuning Method | QLoRA using PEFT (Parameter-Efficient Fine-Tuning) |
| Dataset | [`medss19/formatted-home-remedies`](https://huggingface.co/datasets/medss19/formatted-home-remedies) |
| Framework | Hugging Face `transformers`, `peft`, `bitsandbytes`, `trl` |
| Compute | Google Colab (T4/16GB or A100 recommended) |

## 📌 Key Concepts Explained

### What is LLaMA 2?

LLaMA (Large Language Model Meta AI) is an open-weight foundational language model released by Meta AI. This tutorial uses `LLaMA-2-7B-Chat`, designed for conversational instruction-following tasks.

### Why QLoRA?

**QLoRA** (Quantized LoRA) is a method that enables efficient fine-tuning by:

* Loading large models (like 7B parameters) in **4-bit precision** to significantly reduce GPU memory requirements
* Using **LoRA** (Low-Rank Adaptation) for **parameter-efficient fine-tuning** (PEFT), which tunes only a small number of parameters (LoRA layers), keeping the rest frozen

This approach enables training LLaMA 2 models even on consumer-grade GPUs with limited memory (15GB GPUs like Colab T4).

### LoRA vs. Full Fine-Tuning

| Aspect | Full Fine-Tuning | LoRA |
|--------|------------------|------|
| Memory Usage | High | Low |
| Parameters Updated | All | Few (~1-2%) |
| Speed | Slower | Faster |
| Ideal For | Big Datasets | Small/medium datasets |

### Dataset Overview

Original Dataset: [`medss19/home_remedies_dataset`](https://huggingface.co/datasets/medss19/home_remedies_dataset)

Formatted Dataset (used for training): [`medss19/formatted-home-remedies`](https://huggingface.co/datasets/medss19/formatted-home-remedies)

Each entry in the dataset is converted into a LLaMA-style prompt:

```
<s>[INST] I am suffering from Cold. Please suggest a home remedy. [/INST]
Here is a suggested remedy:

**Item:** Turmeric
**Remedy:** Mix turmeric with warm milk.
**Recommended Yoga:** Pranayama
</s>
```

> **Note**: This dataset is for **educational/demonstration purposes**. The model trained on it is not intended for production use or medical advice.

## 🛠️ Installation

Run this in your Colab notebook:

```bash
!pip install -q accelerate==0.21.0 peft==0.4.0 transformers==4.31.0 trl==0.4.7
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install bitsandbytes --no-cache-dir --force-reinstall
```

## 🧠 Dataset Preprocessing Logic

In `HomeRemedy_Llama_Dataset.ipynb`, we:

1. Load the dataset using 🤗 Datasets library
2. Shuffle for randomness
3. Convert each entry to the prompt-response format expected by LLaMA2 Chat

```python
def transform_remedy(example):
    user_prompt = f"I am suffering from {example['Health Issue']}. Please suggest a home remedy."
    assistant_response = (
        f"Here is a suggested remedy:\n\n"
        f"**Item:** {example['Name of Item']}\n"
        f"**Remedy:** {example['Home Remedy']}\n"
        f"**Recommended Yoga:** {example['Yogasan']}\n"
    )
    return {
        "text": f"<s>[INST] {user_prompt} [/INST] {assistant_response} </s>"
    }
```

## 🧪 Model Configuration

### Base Model

* **`NousResearch/Llama-2-7b-chat-hf`**
* Optimized for instruction-following and chat-based tasks

### QLoRA & Quantization Config

```python
lora_r = 64      # Rank of the LoRA matrices (higher = more capacity but more memory)
lora_alpha = 16  # Scaling factor for LoRA
lora_dropout = 0.1

bnb_4bit_quant_type = "nf4"  # Most memory-efficient and accurate for 4-bit
bnb_4bit_compute_dtype = torch.float16  # Use float16 for calculations
```

## 🔁 Training Arguments

```python
num_train_epochs = 1                # Small dataset? 1–3 epochs is enough
per_device_train_batch_size = 4     # Keep small to avoid OOM
learning_rate = 2e-4                # Higher LR works for PEFT methods
gradient_accumulation_steps = 1     # You can increase if memory allows
max_seq_length = 512                # Maximum context length
group_by_length = True              # Efficient batch packing
```

### How many epochs to train?

* Small dataset (like ours): **1–3 epochs** is enough to prevent overfitting
* Large dataset: **3–10 epochs** based on validation perplexity

## 📄 Workflow Summary

### Step 1: Load Base Model in 4-bit
We use `bitsandbytes` to load the model efficiently:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    quantization_config=bnb_config
)
```

### Step 2: Apply QLoRA Configuration

Using `peft.LoraConfig`, we define low-rank adapters to insert into the model:

```python
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
```

### Step 3: Train the Model

Using SFTTrainer from TRL library to fine-tune only the LoRA layers.

### Step 4: Merge and Save

After training:

* Reload the base model in FP16
* Load LoRA adapters using `PeftModel`
* Call `.merge_and_unload()` to combine weights
* Save the tokenizer with proper `pad_token` and `padding_side`

## 🧠 Common Questions Answered

### What is Perplexity and how do I evaluate it?

Perplexity measures how "surprised" a model is by real data. Lower = better.

To compute it after training:

```python
from transformers import pipeline
from datasets import load_dataset
import math

eval_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

def compute_perplexity(texts):
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
    return math.exp(loss.item())

# Example:
sample_texts = dataset["text"][:10]
print("Perplexity:", compute_perplexity(sample_texts))
```

### What do `lora_r`, `lora_alpha`, `lora_dropout` mean?

| Parameter      | Description                                                         |
| -------------- | ------------------------------------------------------------------- |
| `lora_r`       | Rank of the low-rank update matrix. Higher = more learnable params. |
| `lora_alpha`   | Scaling factor. Usually 2–16. Higher = faster adaptation.           |
| `lora_dropout` | Dropout rate for LoRA layers. Helps prevent overfitting.            |

### What is `bnb_4bit_quant_type`?

* `"nf4"`: NormalFloat4 — better performance & accuracy than `"fp4"`
* We use **4-bit quantization** to load the base model in minimal memory

### What are `gradient_checkpointing` and `fp16`?

* `gradient_checkpointing=True`: Saves memory by trading compute for storage
* `fp16=True` or `bf16=True`: Uses lower precision floating point for training

Set `bf16=True` only if your GPU supports it (like A100). Otherwise, use `fp16`.

### Why Tokenizer Adjustments?

* **No Pad Token**: LLaMA models don't have a padding token by default
* **Fix**: We use `eos_token` as `pad_token`, and set `padding_side="right"` to avoid input misalignment

## 💾 Saving and Using the Fine-Tuned Model

After training, the fine-tuned adapter weights are saved in `./results`.

You can reload the model like this:

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
model = PeftModel.from_pretrained(base_model, "./results")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate remedy
prompt = "I am suffering from headache. Please suggest a home remedy."
inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📈 Tips for Better Results

| Area               | Recommendation                                                     |
| ------------------ | ------------------------------------------------------------------ |
| Dataset Size       | Add more examples or combine multiple home remedy sources          |
| Prompt Engineering | Ensure prompts are consistent with format and informative          |
| LoRA Params        | Try increasing `lora_r` to 128 or `alpha` to 32 for more capacity  |
| Epochs             | Watch for overfitting. Use perplexity or loss to stop early        |
| Logging            | Use `report_to="tensorboard"` to monitor training                  |

## ⚠️ Disclaimer

This model is trained on a small dataset primarily for demonstration purposes. It is **not** intended for real-world medical use or inference. Always consult certified healthcare professionals for health-related advice.

## 📚 References

* [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
* [LoRA Paper: "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)
* [QLoRA Paper: "QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314)
* [LLaMA 2 Chat Format](https://huggingface.co/blog/llama2)
* [PEFT Documentation](https://huggingface.co/docs/peft/)
* [Meta AI's LLaMA Page](https://ai.meta.com/llama/)
* [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
* [TRL: Transformer Reinforcement Learning](https://huggingface.co/docs/trl/index)

## 🤝 Contributing

Pull requests are welcome to extend this example with:

* Larger datasets
* Improved prompt templates
* Additional evaluation metrics
* Feature enhancements

## 📧 Contact

Feel free to reach out for any questions or suggestions via [GitHub Issues](https://github.com/YOUR_USERNAME/YOUR_REPO/issues).
