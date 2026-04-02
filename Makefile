MODEL=unsloth/Qwen3.5-9B
DATASET=./data/test.jsonl

ROOT=..
OUTPUT_DIR=./data/

TOKENS_PATH=./data/keywords.jsonl
TOP_K ?= 300
THRESHOLD ?= 0.7

DEVICE ?= 0
ID ?= test
LORA_R ?= 16
LORA_ALPHA ?= 32
LR ?= 2e-4
TOKENS ?= 0
UNSLOTH ?= 0
QUANTIZE ?= 0
BATCH_SIZE ?= 2
GRAD_ACC ?= 4
MAX_LENGTH ?= 512
EPOCHS ?= 4
OUTPUT_MODEL = ./results/$(ID)_r$(LORA_R)_lr$(LR)

ifeq ($(UNSLOTH), 1)
	UNSLOTH_FLAG = --unsloth
else
	UNSLOTH_FLAG =
endif

ifeq ($(TOKENS), 1)
	TOKENS_FLAG = --add_tokens
else
	TOKENS_FLAG =
endif

ifeq ($(QUANTIZE), 1)
	QUANTIZE_FLAG = --quantize
else
	QUANTIZE_FLAG =
endif

preprocess:
	python3 preprocess.py \
		--root $(ROOT) \
		--output_dir $(OUTPUT_DIR)

extract-kw:
	CUDA_VISIBLE_DEVICES=$(DEVICE) python3 keywords_extraction.py \
		--dataset $(DATASET) \
		--kw_path $(TOKENS_PATH) \
		--top_k $(TOP_K) \
		--thr $(THRESHOLD)

train:
	@echo "------------------------------------------------"
	@echo "STARTING TRAINING: $(ID)"
	@echo "------------------------------------------------"
	@echo "MODEL:          $(MODEL)"
	@echo "DATASET:        $(DATASET)"
	@echo "DEVICE:         cuda:$(DEVICE)"
	@echo "LORA R:         $(LORA_R)"
	@echo "LORA ALPHA:     $(LORA_ALPHA)"
	@echo "LEARNING RATE:  $(LR)"
	@echo "BATCH SIZE:     $(BATCH_SIZE)"
	@echo "GRAD ACCUM:     $(GRAD_ACC)"
	@echo "MAX LENGTH:     $(MAX_LENGTH)"
	@echo "EPOCHS:         $(EPOCHS)"
	@echo "UNSLOTH:        $(UNSLOTH)"
	@echo "------------------------------------------------"
	@echo "OUTPUT DIR:     $(OUTPUT_MODEL)"
	@echo "------------------------------------------------"
	CUDA_VISIBLE_DEVICES=$(DEVICE) python3 train.py \
		--model $(MODEL) \
		--dataset $(DATASET) \
		--output_dir $(OUTPUT_MODEL) \
		$(UNSLOTH_FLAG) \
		$(QUANTIZE_FLAG) \
		$(TOKENS_FLAG) \
		--tokens_path $(TOKENS_PATH) \
		--lora_r $(LORA_R) \
		--lora_alpha $(LORA_ALPHA) \
		--lr $(LR) \
		--batch_size $(BATCH_SIZE) \
		--grad_acc $(GRAD_ACC) \
		--max_length $(MAX_LENGTH) \
		--epochs $(EPOCHS)

merge:
	CUDA_VISIBLE_DEVICES=$(DEVICE) python3 merge.py \
		--base_model $(MODEL) \
        --adapter_dir $(OUTPUT_MODEL)_adapters \
        --merged_dir $(OUTPUT_MODEL)_merged

gguf:
	python3 llama.cpp/convert_hf_to_gguf.py $(OUTPUT_MODEL)_merged \
        --outtype f16 \
        --outfile $(OUTPUT_MODEL).gguf
