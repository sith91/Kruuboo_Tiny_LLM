import sys
import json
import traceback

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No prompt provided"}))
        return

    prompt = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    try:
        from transformers import pipeline
        # Load generator on CPU
        generator = pipeline("text-generation", model="distilgpt2", device=-1)
        res = generator(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            num_return_sequences=1,
            repetition_penalty=1.2,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        full_text = res[0]['generated_text']
        if full_text.startswith(prompt):
            answer = full_text[len(prompt):].strip()
        else:
            answer = full_text.strip()
        print(json.dumps({"answer": answer}))
    except Exception as e:
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))

if __name__ == "__main__":
    main()
