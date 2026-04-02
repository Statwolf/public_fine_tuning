import argparse
import json
import wandb
import torch

from datasets import Dataset

def get_dataset(dataset_path: str):
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return Dataset.from_list(data)

def add_new_tokens(args, model, tokenizer):
    with open(args.tokens_path, "r", encoding="utf-8") as f:
        new_tokens = [json.loads(line)["keyword"] for line in f]
    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    if actual_tokenizer.pad_token is None:
        actual_tokenizer.pad_token = actual_tokenizer.eos_token
    actual_tokenizer.padding_side = "right"

    old_vocab_size = len(actual_tokenizer)
    n_tokens_added = actual_tokenizer.add_tokens(new_tokens)
    new_tokens_ids = list(range(old_vocab_size, old_vocab_size + n_tokens_added))
    print(f"{n_tokens_added} NEW TOKENS ADDED.\n")
    model.resize_token_embeddings(len(actual_tokenizer))
    return new_tokens_ids

def finetune(args):
    if args.unsloth:
        from unsloth import FastLanguageModel
    
    from peft import  LoraConfig, TaskType
    from trl import SFTTrainer, SFTConfig
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM
    )

    project_name = "gabo-fine-tuning"
    run_name = f"{args.output_dir.split('/')[-1]}-gabo-fine-tuning".replace("/", "-")
    wandb.init(
        project=project_name,
        entity="gabo-statwolf",
        name=run_name,
        config=vars(args)
    )

    lora_config = None

    if args.unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_length,
            dtype=torch.bfloat16,
            load_in_4bit=bool(args.quantize),  # QLoRA if True, LoRA if False
        )

        if args.add_tokens:
            new_tokens_ids = add_new_tokens(args, model, tokenizer)
    
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            trainable_token_indices=new_tokens_ids if args.add_tokens else None,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            dtype=torch.bfloat16
        )
        model.enable_input_require_grads()
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        if args.add_tokens:
            add_new_tokens(args, model, tokenizer)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if actual_tokenizer.pad_token is None:
        actual_tokenizer.pad_token = actual_tokenizer.eos_token
    actual_tokenizer.padding_side = "right"

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=5,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        gradient_checkpointing=not args.unsloth,
        dataset_text_field="text",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
        max_length=args.max_length,
        packing=True,
        report_to="wandb",
        run_name=run_name
    )

    dataset = args.dataset_obj.train_test_split(test_size=args.test_size)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()
    
    if args.unsloth:
        trainer.model.save_pretrained(args.output_dir + "_adapters")
        tokenizer.save_pretrained(args.output_dir + "_adapters")
        print(f"Adapters saved to: {args.output_dir}_adapters")

        if not args.quantize:
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Merged model saved to: {args.output_dir}")
        else:
            print("Training done in 4bit: to get a GGUF convertible model, you need to reload the base model in fp16/bf16, load the adapters and do merge separately.")
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tuning")
    parser.add_argument("--model", type=str, help="Base Model to Fine-Tune")
    parser.add_argument("--dataset", type=str, help="Dataset Path (JSONL file)")
    parser.add_argument("--output_dir", type=str, help="Model output directory")
    parser.add_argument("--unsloth", action="store_true", help="Activate Unsloth")
    parser.add_argument("--quantize", action="store_true", help="Activate Quanization")
    parser.add_argument("--add_tokens", action="store_true", help="Add tokens")
    parser.add_argument("--tokens_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    args = parser.parse_args()

    if args.model is None or args.dataset is None or args.output_dir is None:
        raise ValueError("You must provide all the arguments")

    if args.add_tokens and not args.tokens_path:
        raise ValueError("--token-path mandatory when --add-tokens is enabled")

    args.dataset_obj = get_dataset(
        dataset_path=args.dataset
    )

    finetune(args)