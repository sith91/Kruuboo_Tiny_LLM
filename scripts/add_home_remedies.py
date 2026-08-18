import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import RAGSystem

# Config
MODEL_MAX_SEQ_LEN = 128
MODEL_PATH = "models/tiny_llm_quantized.onnx"
TOKENIZER_PATH = "tokenizer.json"

# High-quality home remedies dataset mapping ailments to natural treatments
HOME_REMEDIES = [
    {
        "ailment": "Cough",
        "remedy": "Drink warm water mixed with honey and ginger juice. Honey acts as a natural demulcent to soothe the throat, while ginger has anti-inflammatory properties that reduce coughing reflexes."
    },
    {
        "ailment": "Sore Throat",
        "remedy": "Gargle with warm saltwater (half a teaspoon of salt in a glass of warm water) 3-4 times a day. This reduces swelling, flushes out bacteria, and cleanses the throat lining."
    },
    {
        "ailment": "Common Cold",
        "remedy": "Stay hydrated by drinking plenty of warm fluids like herbal teas, chicken soup, and warm water. Inhaling steam from a bowl of hot water or using a humidifier helps clear nasal congestion."
    },
    {
        "ailment": "Headache",
        "remedy": "Apply a cold compress or ice pack wrapped in a cloth to your forehead or temples for 15 minutes. Hydrating with plenty of water and resting in a quiet, dark room can also relieve tension headaches."
    },
    {
        "ailment": "Indigestion and Upset Stomach",
        "remedy": "Sip ginger tea or peppermint tea. Ginger speeds up gastric emptying and relieves nausea, while peppermint relaxes the digestive muscles to alleviate cramps and gas."
    },
    {
        "ailment": "Minor Burns",
        "remedy": "Cool the burn immediately by holding it under cool running tap water for 10-15 minutes. Apply pure aloe vera gel to reduce pain and inflammation. Avoid applying ice, butter, or toothpaste."
    },
    {
        "ailment": "Fever",
        "remedy": "Get plenty of rest and drink water or electrolyte solutions to prevent dehydration. Placing a cool, damp washcloth on the forehead or back of the neck helps lower body temperature naturally."
    },
    {
        "ailment": "Insomnia and Sleeplessness",
        "remedy": "Drink a warm cup of chamomile tea or warm milk before bed. Chamomile contains apigenin, an antioxidant that binds to brain receptors to promote sleepiness and relaxation."
    },
    {
        "ailment": "Insect Bites and Itching",
        "remedy": "Apply a paste made of baking soda and water, or a dab of calamine lotion. A cold compress can also reduce swelling and numb the itching sensation from mosquito or bug bites."
    },
    {
        "ailment": "Nausea",
        "remedy": "Consume ginger capsules, ginger candies, or fresh ginger tea. Smelling peppermint essential oil or sucking on lemon slices also helps calm stomach discomfort and nausea."
    }
]

def main():
    print("=" * 60)
    print("INGESTING HOME REMEDIES DATASET INTO MEDICAL DOMAIN")
    print("=" * 60)

    # 1. Initialize RAG
    print("🤖 Initializing RAG System...")
    rag = RAGSystem(MODEL_PATH, TOKENIZER_PATH, max_seq_len=MODEL_MAX_SEQ_LEN)

    # 2. Ingest
    print("\n✍️ Loading home remedies into the 'medical' domain...")
    count = 0
    for item in HOME_REMEDIES:
        ailment = item["ailment"]
        remedy = item["remedy"]
        text = f"Home Remedy for {ailment}: {remedy}"
        
        # Add to medical domain
        rag.add_domain_knowledge("medical", text, f"HomeRemedies:{ailment}")
        print(f" ✅ Ingested remedy for: {ailment}")
        count += 1

    print(f"\n🎉 Successfully added {count} home remedies to the 'medical' domain!")
    print("You can switch to the 'medical' domain in the shell:")
    print("  /mode medical")
    print("And ask about remedies, like: 'home remedy for cough'")

if __name__ == "__main__":
    main()
