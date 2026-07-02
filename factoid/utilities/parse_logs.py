import re

def parse_log(filepath, label):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    pos_matches = list(re.finditer(r'--- Testing Position (\d+) ---', content))
    print(f'=== {label} ===')
    results = {}
    for i, m in enumerate(pos_matches):
        start = m.start()
        end = pos_matches[i+1].start() if i+1 < len(pos_matches) else len(content)
        block = content[start:end]
        true_count = len(re.findall(r'Match: True', block))
        false_count = len(re.findall(r'Match: False', block))
        total = true_count + false_count
        pos = int(m.group(1))
        acc = true_count/total*100 if total > 0 else 0
        results[pos] = (true_count, total, acc)
        print(f'  Position {pos}: {true_count}/{total} = {acc:.2f}%')
    print()
    return results

def parse_log_truncated(filepath, label):
    """Parse using truncated (first-line) matching for verbose models."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    pos_matches = list(re.finditer(r'--- Testing Position (\d+) ---', content))
    print(f'=== {label} (truncated) ===')
    results = {}
    for i, m in enumerate(pos_matches):
        start = m.start()
        end = pos_matches[i+1].start() if i+1 < len(pos_matches) else len(content)
        block = content[start:end]

        # Extract each question block and recheck match using only first line of response
        q_pattern = re.compile(
            r'\[Question \d+\]\s*\nTarget Answer: (.+?)\s*\nModel Response: (.+?)(?=\nMatch:)',
            re.DOTALL
        )
        correct = 0
        total = 0
        for qm in q_pattern.finditer(block):
            target = qm.group(1).strip().lower()
            response_full = qm.group(2).strip()
            # Take first line only
            first_line = response_full.split('\n')[0].strip().lower()
            if target in first_line:
                correct += 1
            total += 1

        pos = int(m.group(1))
        acc = correct/total*100 if total > 0 else 0
        results[pos] = (correct, total, acc)
        print(f'  Position {pos}: {correct}/{total} = {acc:.2f}%')
    print()
    return results


parse_log(r'e:\WorkSpace\lostinthemiddle\results\phi3_results\live_output_ollama.txt', 'PHI-3 (Ollama)')
parse_log(r'e:\WorkSpace\lostinthemiddle\results\gemma_results\live_output_gemini_gemma.txt', 'Gemma 4 31B-IT (API) - raw')
parse_log_truncated(r'e:\WorkSpace\lostinthemiddle\results\gemma_results\live_output_gemini_gemma.txt', 'Gemma 4 31B-IT (API) - first-line truncated')
