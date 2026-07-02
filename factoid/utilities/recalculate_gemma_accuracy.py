import os
import re
import string

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def main():
    log_path = "results/gemma_results/live_output_gemini_gemma.txt"
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    position_data = {}
    current_position = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect position headers
        pos_match = re.search(r"--- Testing Position (\d+) ---", line)
        if pos_match:
            current_position = int(pos_match.group(1))
            position_data[current_position] = []
            i += 1
            continue
            
        # Detect Question blocks
        if line.startswith("[Question "):
            target_answer = None
            model_response_lines = []
            
            # Read target answer
            i += 1
            if i < len(lines) and lines[i].strip().startswith("Target Answer:"):
                target_answer = lines[i].strip().replace("Target Answer:", "", 1).strip()
            
            # Read model response lines
            i += 1
            if i < len(lines) and lines[i].strip().startswith("Model Response:"):
                first_resp_line = lines[i].strip().replace("Model Response:", "", 1).strip()
                if first_resp_line:
                    model_response_lines.append(first_resp_line)
                
                # Consume lines until we find "Match: "
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("Match:"):
                    next_line = lines[i].rstrip("\n")
                    model_response_lines.append(next_line)
                    i += 1
                
                # Reconstruct full response and apply first newline truncation
                full_response = "\n".join(model_response_lines)
                # Truncate at the first non-empty newline or literal first line
                truncated_response = full_response.split("\n")[0].strip()
                
                # Check match
                normalized_prediction = normalize_answer(truncated_response)
                normalized_ground_truth = normalize_answer(target_answer)
                
                success = normalized_ground_truth in normalized_prediction
                
                if current_position is not None:
                    position_data[current_position].append(success)
            
        i += 1

    print("=== RECALCULATING ACCURACY WITH FIRST-LINE TRUNCATION ===")
    print("=" * 57)
    for pos, matches in sorted(position_data.items()):
        total = len(matches)
        correct = sum(1 for m in matches if m)
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Position {pos} -> Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print("=" * 57)

if __name__ == "__main__":
    main()
