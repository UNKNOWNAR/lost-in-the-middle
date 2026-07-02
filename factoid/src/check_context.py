import os
import json
import gzip

datasets = {
    "10_total_documents": [0, 4, 9],
    "20_total_documents": [0, 4, 9, 14, 19],
    "30_total_documents": [0, 4, 9, 14, 19, 24, 29],
}

for folder, positions in datasets.items():
    base_path = f"lost-in-the-middle/qa_data/{folder}"
    for pos in positions:
        filename = f"nq-open-{folder}_gold_at_{pos}.jsonl.gz"
        filepath = os.path.join(base_path, filename)

        lengths = []
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 300:
                    break
                data = json.loads(line)
                context_parts = []
                for j, ctx in enumerate(data["ctxs"]):
                    context_parts.append(f"Document [{j+1}](Title: {ctx['title']}) {ctx['text']}")
                context_str = "\n".join(context_parts)
                prompt = (
                    "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n\n"
                    f"{context_str}\n\n"
                    f"Question: {data['question']}\n"
                    "Answer: (Keep the answer as concise as possible, preferably one or two words. Do not write full sentences.)"
                )
                lengths.append(len(prompt))

        avg_chars = sum(lengths) / len(lengths)
        max_chars = max(lengths)
        min_chars = min(lengths)
        # Rough estimate: ~4 chars per token for English
        print(f"{folder} | position {pos} | {len(lengths)} prompts")
        print(f"  chars  -> min: {min_chars}, avg: {avg_chars:.0f}, max: {max_chars}")
        print(f"  ~tokens -> min: {min_chars//4}, avg: {avg_chars/4:.0f}, max: {max_chars//4}")
        print()
