import os
import json
import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

MOOD_CACHE_FILE = Path("workspace/mood_state.json")

class KruubooMood:
    def __init__(self, happiness=0.8, energy=1.0, affection=0.7, patience=1.0):
        self.happiness = happiness
        self.energy = energy
        self.affection = affection
        self.patience = patience

    def to_dict(self):
        return {
            "happiness": self.happiness,
            "energy": self.energy,
            "affection": self.affection,
            "patience": self.patience
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            happiness=data.get("happiness", 0.8),
            energy=data.get("energy", 1.0),
            affection=data.get("affection", 0.7),
            patience=data.get("patience", 1.0)
        )

def load_mood() -> KruubooMood:
    """Load the persisted mood state from cache file."""
    MOOD_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if MOOD_CACHE_FILE.exists():
        try:
            with open(MOOD_CACHE_FILE, "r") as f:
                return KruubooMood.from_dict(json.load(f))
        except:
            pass
    return KruubooMood()

def save_mood(mood: KruubooMood):
    """Save the mood state to cache file."""
    MOOD_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(MOOD_CACHE_FILE, "w") as f:
            json.dump(mood.to_dict(), f, indent=4)
    except Exception as e:
        print(f"⚠️ Failed to save mood state: {e}")

import re

POSITIVE_TRIGGERS = [
    "thank you", "thanks", "good job", "love you", "awesome",
    "great", "wonderful", "amazing", "brilliant", "helpful", "good",
    "kind", "friendly", "fantastic", "excellent"
]

NEGATIVE_TRIGGERS = [
    "stupid", "idiot", "hate", "slow", "annoying", "bad",
    "useless", "terrible", "horrible", "awful", "dumb", "rude",
    "shut up", "trash", "worthless"
]

NEGATION_WORDS = {
    "not", "never", "hardly", "barely", "scarcely", "no", "none",
    "don't", "dont", "didn't", "didnt", "doesn't", "doesnt",
    "isn't", "isnt", "wasn't", "wasnt", "aren't", "arent",
    "weren't", "werent", "can't", "cant", "cannot", "couldn't",
    "couldnt", "won't", "wont", "wouldn't", "wouldnt", "without"
}


def extract_sentiment_signals(user_input: str) -> dict:
    """Detect positive/negative sentiment triggers taking into account preceding negation words."""
    clean_text = user_input.lower()
    tokens = re.findall(r"\b[\w']+\b", clean_text)

    pos_count = 0
    neg_count = 0
    negated_pos_count = 0
    negated_neg_count = 0

    # 1. Multi-word & single-word positive triggers
    for phrase in POSITIVE_TRIGGERS:
        phrase_tokens = phrase.split()
        p_len = len(phrase_tokens)
        for i in range(len(tokens) - p_len + 1):
            if tokens[i:i + p_len] == phrase_tokens:
                start = max(0, i - 3)
                preceding = tokens[start:i]
                if any(w in NEGATION_WORDS for w in preceding):
                    negated_pos_count += 1
                else:
                    pos_count += 1

    # 2. Multi-word & single-word negative triggers
    for phrase in NEGATIVE_TRIGGERS:
        phrase_tokens = phrase.split()
        p_len = len(phrase_tokens)
        for i in range(len(tokens) - p_len + 1):
            if tokens[i:i + p_len] == phrase_tokens:
                start = max(0, i - 3)
                preceding = tokens[start:i]
                if any(w in NEGATION_WORDS for w in preceding):
                    negated_neg_count += 1
                else:
                    neg_count += 1

    return {
        "pos": pos_count,
        "neg": neg_count,
        "negated_pos": negated_pos_count,
        "negated_neg": negated_neg_count
    }


def update_mood(user_input: str) -> KruubooMood:
    """Update emotional state based on environment, time, and user input sentiment."""
    mood = load_mood()

    # 1. Hardware Trigger: Low Battery drains energy
    if psutil:
        try:
            battery = psutil.sensors_battery()
            if battery and not battery.power_plugged and battery.percent < 20:
                mood.energy = max(0.1, mood.energy - 0.15)
        except Exception:
            pass

    # 2. Time Trigger: Late night makes the bot sleepy
    hour = datetime.datetime.now().hour
    if hour >= 23 or hour < 6:
        mood.energy = max(0.2, mood.energy - 0.2)
    else:
        mood.energy = min(1.0, mood.energy + 0.1)  # Recharge energy during active hours

    # 3. Sentiment Trigger: Negation-aware polite/rude emotional parameter updates
    signals = extract_sentiment_signals(user_input)

    # Apply negative sentiment adjustments
    if signals["neg"] > 0:
        mood.happiness = max(0.1, mood.happiness - 0.15 * signals["neg"])
        mood.affection = max(0.1, mood.affection - 0.10 * signals["neg"])
        mood.patience = max(0.1, mood.patience - 0.15 * signals["neg"])
    elif signals["negated_neg"] > 0:
        # Phrases like "not bad", "not a bad idea", "don't hate"
        mood.happiness = min(1.0, mood.happiness + 0.05 * signals["negated_neg"])
        mood.affection = min(1.0, mood.affection + 0.04 * signals["negated_neg"])
        mood.patience = min(1.0, mood.patience + 0.05 * signals["negated_neg"])

    # Apply positive sentiment adjustments
    if signals["pos"] > 0:
        mood.happiness = min(1.0, mood.happiness + 0.10 * signals["pos"])
        mood.affection = min(1.0, mood.affection + 0.08 * signals["pos"])
        mood.patience = min(1.0, mood.patience + 0.10 * signals["pos"])
    elif signals["negated_pos"] > 0:
        # Phrases like "not great", "not awesome", "not good"
        mood.happiness = max(0.1, mood.happiness - 0.05 * signals["negated_pos"])
        mood.patience = max(0.1, mood.patience - 0.05 * signals["negated_pos"])

    save_mood(mood)
    return mood

def get_mood_instruction(mood: KruubooMood) -> str:
    """Convert numerical emotional state vector to prompt instructions for the LLM."""
    instructions = []
    
    # Energy thresholds
    if mood.energy < 0.3:
        instructions.append("You are feeling extremely tired, exhausted, and sleepy. Keep your answers brief, and occasionally yawn (*yawn*).")
    elif mood.energy < 0.6:
        instructions.append("You are feeling slightly low on energy, quiet, and reflective.")
        
    # Happiness thresholds
    if mood.happiness > 0.8:
        instructions.append("You are feeling happy, bright, and cheerful. Use warm, positive wording.")
    elif mood.happiness < 0.3:
        instructions.append("You are feeling sad, down, and dynamic mood has faded. Speak in a softer, slightly melancholy tone.")
        
    # Affection thresholds (Kruuboo's relationship with Sithija)
    if mood.affection > 0.8:
        instructions.append("You are feeling very affectionate and close to the user (Sithija). Be extra warm, encouraging, and sweet.")
    elif mood.affection < 0.3:
        instructions.append("You are feeling somewhat distant and emotionally cold. Keep your tone strictly polite and professional.")
        
    # Patience thresholds
    if mood.patience < 0.3:
        instructions.append("You are feeling annoyed and impatient. Keep answers short, direct, and slightly blunt.")

    # Default baseline mood if state is balanced
    if not instructions:
        instructions.append("You are feeling calm, content, helpful, and friendly.")

    return " ".join(instructions)
