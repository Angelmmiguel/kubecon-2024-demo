import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
import bitsandbytes as bnb
from trl import SFTTrainer


# Base model to finetune
model_id = "microsoft/phi-2"


def generate_prompt(data_point):
    """Generate a prompt with an instruction and the expected response."""

    text = f"""You are a smart assistant that follows instructions:

    ### Instruction
    Extract information from a given text in a structured way. Based on the user message, you must extract the following information: name, age, location and role. Then, you return these 4 properties in the following format:

    {{ "name": "NAME", "age": "AGE", "location": "LOCATION", "role": "ROLE" \}}

    Where you change the uppercased words with the values extracted from the original text. You must return only this data as output, skipping any other text before and after. If there's a double quote symbol in any of the values, escape it using the \ symbol.

    Input: {data_point["input"]}

    ### Response
    {{ "name": "{data_point["output"]["name"]}", "age": "{data_point["output"]["age"]}", "location": "{data_point["output"]["location"]}", "role": "{data_point["output"]["role"]}" \}}<|endoftext|>"""

    return text

# Finetuning code
# ------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Init the model
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

# Configure the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

# Load the dataset
dataset = load_dataset("angelmmiguel/synthetic-introduction-extraction", split="train")

# Set the right
text_column = [generate_prompt(data_point) for data_point in dataset]
dataset = dataset.add_column("prompt", text_column)

dataset = dataset.shuffle(seed=2312)  # Shuffle dataset here
dataset = dataset.map(lambda samples: tokenizer(samples["prompt"], truncation=True, return_tensors="np",padding="max_length",max_length=2048), batched=True)
dataset = dataset.train_test_split(test_size=0.7)

# Final data. We're skipping testing as this is a demo.
train_data = dataset["train"]
test_data = dataset["test"]

# Prepare the model
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules = [ "q_proj", "k_proj", "v_proj", "dense" ],
    modules_to_save = [ "lm_head", "embed_tokens" ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()

# Print final stats
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

# Ensure the GPU is empty
torch.cuda.empty_cache()

# Configure the trainer. The number of steps here are pretty low, as this
# configuration is just for demo purposes. For a production-ready fine-tuning,
# you would need to update and iterate over all these parameters.
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="prompt",
    peft_config=lora_config,
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=0.03,
        max_steps=25,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    )
)

# Start training
trainer.train()

# New model name!
new_model = "phi2-introduction-extractor"

# Store just the new LoRA layers
trainer.model.save_pretrained(new_model)

# Build the complete model + LoRA layers.
# vLLM doesn't support Phi2 LoRA layers yet.
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("phi2-introduction-extractor-full", safe_serialization=True)
tokenizer.save_pretrained("phi2-introduction-extractor-full")
