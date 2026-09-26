import re
from typing import Dict, List, Optional

GREETINGS_MAP = {
    "hi": "Hi there! How can I help you today?",
    "hello": "Hello! How can I assist you today?",
    "hey": "Hey! How can I help you?",
    "greetings": "Greetings! How can I assist you today?",
    "thanks": "You're very welcome! Let me know if you need anything else.",
    "thank you": "You're very welcome! I'm happy to help.",
    "bye": "Goodbye! Have a great day!",
    "goodbye": "Goodbye! Take care and see you next time!",
    "how are you": "I'm doing great, thank you! How can I assist you today?",
    "how are you doing": "I'm doing well! How can I help you today?",
    "hows it going": "It's going great! What can I do for you today?",
    "how is it going": "Everything is going well! How can I help you today?",
    "how are you today": "I'm doing great, thank you! How are you doing today?",
    "how is your day": "My day is going great, thank you for asking! How is yours?",
    "how is everything": "Everything is running smoothly! How can I assist you?",
    "hows everything": "Everything is great! What are you working on today?",
    "good morning": "Good morning! Hope you have a wonderful day. How can I help you?",
    "good afternoon": "Good afternoon! How can I assist you today?",
    "good evening": "Good evening! How can I help you tonight?",
    "good night": "Good night! Sleep well and have pleasant dreams.",
    "nice to meet you": "Nice to meet you too! How can I help you today?",
    "pleased to meet you": "Pleased to meet you as well! What can I assist you with?",
    "what can you do": "I am Kruuboo Tiny LLM (105.4M parameters). I can help answer questions in Medical, Legal, Survival, Mathematics, Pricing, and Sinhala domains using Native RAG and rStar reasoning!",
    "who are you": "I am Kruuboo Tiny LLM, a self-contained domain-aware AI assistant running locally.",
    "what are you": "I am a local Small Language Model (SLM) with native domain retrieval and rStar reasoning capabilities.",
    "i feel very lonely": "I'm really sorry you're feeling lonely. I'm right here with you! What's on your mind? We can chat about anything you like.",
    "i feel very lonley": "I'm really sorry you're feeling lonely. I'm right here with you! What's on your mind? We can chat about anything you like.",
    "i feel lonely": "I'm here for you! You're never truly alone. Would you like to talk about what's going on, or just chat about something fun?",
    "i feel lonley": "I'm here for you! You're never truly alone. Would you like to talk about what's going on, or just chat about something fun?",
    "i am lonely": "I'm here with you! Feel free to share what's on your mind, or we can talk about movies, books, or any topic you enjoy.",
    "i am lonley": "I'm here with you! Feel free to share what's on your mind, or we can talk about movies, books, or any topic you enjoy.",
    "i'm lonely": "I'm right here! Let's talk—what kind of things make you feel relaxed or happy?",
    "i'm lonley": "I'm right here! Let's talk—what kind of things make you feel relaxed or happy?",
    "i feel sad": "I'm so sorry you're feeling down. I'm here to listen and keep you company. What happened?",
    "i'm sad": "I'm here for you. Take a deep breath. Is there anything specific on your mind, or would you like a comforting chat?",
    "i need a friend": "I'm always happy to be here as your friend! What's on your mind today?",
    "can we talk": "Of course! I'm all ears. What would you like to talk about?",
    "are you there": "Yes, I'm right here! How can I help you?"
}

def extract_user_name_from_history(history: Optional[List[Dict]]) -> str:
    """Scan history to locate the user's name if they have introduced themselves"""
    if not history:
        return "my friend"
    user_patterns = [
        r"(?:my name is|i am|i'm|call me|this is)\s+([a-zA-Z0-9_\-]+)",
    ]
    for turn in reversed(history):
        if turn.get("role") == "user":
            content = turn.get("content", "")
            for pat in user_patterns:
                match = re.search(pat, content, re.IGNORECASE)
                if match:
                    name = match.group(1).strip("?.!, \n\t").title()
                    if name and name.lower() not in ["a", "an", "the", "not", "here", "feeling", "lonely", "sad", "good", "fine", "ready"]:
                        return name
    return "my friend"

def check_user_intro(clean_q: str) -> Optional[str]:
    """Check if the user is introducing themselves (e.g. 'My name is Sithija', 'I am Sithija', 'Call me Sithija')"""
    user_patterns = [
        r"^my name is\s+([a-zA-Z0-9_\-]+)",
        r"^i am\s+([a-zA-Z0-9_\-]+)",
        r"^i'm\s+([a-zA-Z0-9_\-]+)",
        r"^call me\s+([a-zA-Z0-9_\-]+)",
        r"^this is\s+([a-zA-Z0-9_\-]+)",
    ]
    for pat in user_patterns:
        match = re.search(pat, clean_q, re.IGNORECASE)
        if match:
            name = match.group(1).strip("?.!, \n\t").title()
            if name and name.lower() not in ["a", "an", "the", "not", "here", "feeling", "lonely", "sad", "good", "fine", "ready", "asking", "wondering"]:
                return name
    return None

def check_naming_request(clean_q: str) -> Optional[str]:
    """Check if the user is assigning a name to the bot, returns the name if found"""
    bot_naming_patterns = [
        r"(?:i will name you as|your name is|call yourself|i'll name you|name you as)\s+([a-zA-Z0-9_\-\s]+)"
    ]
    for pattern in bot_naming_patterns:
        match = re.search(pattern, clean_q, re.IGNORECASE)
        if match:
            bot_name = match.group(1).strip("?.!, \n\t").title()
            if bot_name:
                return bot_name
    return None

def check_bot_name_query(clean_q: str) -> bool:
    """Check if the user is asking for the bot's name"""
    bot_patterns = [
        r"(?:what(?:'s| is)|tell me)?\s*your name",
        r"who are you",
        r"what do they call you",
        r"what should i call you",
        r"what is your identity",
    ]
    return any(re.search(pat, clean_q, re.IGNORECASE) for pat in bot_patterns)

def check_user_name_query(clean_q: str) -> bool:
    """Check if the user is asking for their own name"""
    user_name_patterns = [
        r"(?:what(?:'s| is)|tell me)?\s*my name",
        r"who am i",
        r"do you (?:know|remember)\s+(?:what\s+)?my name",
        r"remember my name",
        r"know my name",
        r"what do you call me",
    ]
    return any(re.search(pat, clean_q, re.IGNORECASE) for pat in user_name_patterns)

