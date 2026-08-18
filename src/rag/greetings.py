import re
from typing import Dict, List, Optional

GREETINGS_MAP = {
    "hi": "Hi there! How can I help you today?",
    "hello": "Hello! How can I assist you today?",
    "hey": "Hey! How can I help you?",
    "greetings": "Greetings! How can I assist you today?",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
    "bye": "Goodbye!",
    "goodbye": "Goodbye!",
    "how are you": "I'm doing great, thank you! How can I assist you today?",
    "how are you doing": "I'm doing well! How can I help you today?",
    "how's it going": "It's going great! What can I do for you today?",
    "how is it going": "Everything is going well! How can I help you today?",
    "how are you today": "I'm doing great, thank you! How are you doing today?"
}

def extract_user_name_from_history(history: Optional[List[Dict]]) -> str:
    """Scan history to locate the user's name if they have introduced themselves"""
    if not history:
        return "my friend"
    user_patterns = [r"(?:my name is|i am|call me)\s+([a-zA-Z0-9_\-\s]+)"]
    for turn in reversed(history):
        if turn.get("role") == "user":
            content = turn.get("content", "")
            for pat in user_patterns:
                match = re.search(pat, content, re.IGNORECASE)
                if match:
                    name = match.group(1).strip("?.!, \n\t")
                    if name:
                        return name.title()
    return "my friend"

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
    bot_name_queries = ["what is your name", "whats your name", "what's your name", "who are you", "what do they call you"]
    return any(q in clean_q for q in bot_name_queries)

def check_user_name_query(clean_q: str) -> bool:
    """Check if the user is asking for their own name"""
    name_queries = ["what is my name", "whats my name", "what's my name", "who am i", "do you know my name"]
    return any(q in clean_q for q in name_queries)
