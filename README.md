# Lost in the Middle - LLM Evaluation

This repository contains scripts to evaluate Large Language Models (LLMs) on the "Lost in the Middle" phenomenon, measuring how well they retrieve information from large contexts.

## Project Structure
- `src/`: Contains all Python evaluation scripts (Gemini API, Ollama, Baseline tests).
- `data/`: Contains raw datasets and SQuAD zip files (ignored in git).
- `results/`: Output logs and generated Seaborn graphs from the evaluations (ignored in git).

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate it and install dependencies: `pip install google-genai pandas matplotlib seaborn python-dotenv ollama`
3. Add your `.env` file with `GEMINI_API_KEY=your_key_here` if using cloud APIs.
