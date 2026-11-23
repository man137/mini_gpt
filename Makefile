# mini-chatgpt Makefile - common workflows. `make help` lists targets.
.PHONY: help install install-frontend test train-scratch smoke-scratch \
        prepare-data train-qlora merge ingest serve serve-fake frontend eval eval-selftest clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "} {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## install backend + RAG + scratch deps (skip frontend)
	pip install -r requirements-serve.txt
	pip install -r requirements-scratch.txt

install-frontend: ## install frontend node modules
	cd frontend && npm install

test: ## run the offline test suite (no model downloads)
	pytest

train-scratch: ## train the from-scratch GPT (default config)
	python scratch_gpt/train.py

smoke-scratch: ## quick from-scratch pipeline check (~1 min, tiny config)
	python scratch_gpt/train.py --smoke

prepare-data: ## validate + split + chat-template the fine-tune data
	python finetune/prepare_data.py

train-qlora: ## QLoRA fine-tuning (needs GPU - see finetune/notebooks/train_colab.md)
	python finetune/train_qlora.py

merge: ## merge LoRA adapter into base weights + compare prompts
	python finetune/merge_and_test.py

ingest: ## (re)build the RAG index from rag/docs
	python rag/ingest.py

serve: ## start the API on :8000 (set LLM_BACKEND env: hf|ollama|fake)
	uvicorn backend.main:app --port 8000

serve-fake: ## start the API with the fake LLM (no model needed)
	LLM_BACKEND=fake uvicorn backend.main:app --port 8000

frontend: ## start the Vite dev server on :5173
	cd frontend && npm run dev

eval-selftest: ## offline checks of the eval harness scoring
	python eval/run_eval.py --selftest

eval: ## full eval against a running backend (rag system, 40 questions)
	python eval/run_eval.py --systems rag

clean: ## remove caches and regenerable artifacts
	rm -rf .pytest_cache **/__pycache__
	rm -rf scratch_gpt/logs scratch_gpt/checkpoints rag/index
	rm -rf frontend/node_modules frontend/dist
