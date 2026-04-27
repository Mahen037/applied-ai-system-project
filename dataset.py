"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - EMOJI_SENTIMENT: mapping of common emojis to sentiment scores
  - NEGATION_WORDS: words that flip the sentiment of the next token
  - WORD_WEIGHTS: higher-impact words with custom weights
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy", "great", "good", "love", "excited", "awesome", "fun",
    "chill", "relaxed", "amazing", "wonderful", "fantastic", "brilliant",
    "excellent", "joyful", "delighted", "pleased", "thrilled", "grateful",
    "proud", "hopeful", "beautiful", "perfect", "superb", "outstanding",
    "incredible", "blessed", "cheerful", "glad", "enjoy", "enjoying",
    "lol", "lmao", "haha", "yay", "woohoo", "fire", "goat", "lit",
    "slay", "vibe", "vibes", "dope", "clutch", "w",
]

NEGATIVE_WORDS = [
    "sad", "bad", "terrible", "awful", "angry", "upset", "tired",
    "stressed", "hate", "boring", "horrible", "disgusting", "miserable",
    "depressed", "frustrated", "annoyed", "disappointing", "disappointed",
    "dreadful", "pathetic", "painful", "unbearable", "heartbroken",
    "anxious", "exhausted", "overwhelmed", "furious", "devastated",
    "ugly", "worse", "worst", "fail", "failed", "failing",
    "trash", "cringe", "mid", "sus", "oof", "yikes", "bruh", "l",
]

# ---------------------------------------------------------------------
# Emoji sentiment mapping
# ---------------------------------------------------------------------

EMOJI_SENTIMENT = {
    # Positive emojis
    "😊": 2, "😄": 2, "😁": 2, "🥰": 2, "❤️": 2, "💕": 2,
    "👍": 1, "🎉": 2, "✨": 1, "🔥": 1, "💪": 1, "😂": 1,
    "🤣": 1, "😎": 1, "🥳": 2, "💯": 1, "🙌": 1, "👏": 1,
    ":)": 1, ":-)": 1, ";)": 1, ":D": 2, ":-D": 2, "<3": 2,
    # Negative emojis
    "😢": -2, "😭": -2, "😞": -1, "😠": -2, "😡": -2, "💔": -2,
    "👎": -1, "😤": -1, "🥲": -1, "😒": -1, "😩": -1, "😫": -1,
    "💀": -1, "☠️": -1, "🤮": -2, "🤢": -1,
    ":(": -1, ":-(": -1, ":/": -1, ":-/": -1,
}

# ---------------------------------------------------------------------
# Negation words (flip the sentiment of the following word)
# ---------------------------------------------------------------------

NEGATION_WORDS = [
    "not", "no", "never", "neither", "nobody", "nothing",
    "nowhere", "nor", "barely", "hardly", "scarcely",
    "don't", "doesn't", "didn't", "isn't", "aren't",
    "wasn't", "weren't", "won't", "wouldn't", "shouldn't",
    "can't", "cannot", "couldn't", "mustn't",
    "dont", "doesnt", "didnt", "isnt", "arent",
    "wasnt", "werent", "wont", "wouldnt", "shouldnt",
    "cant", "couldnt", "mustnt",
]

# ---------------------------------------------------------------------
# Word weights (some words carry stronger sentiment)
# ---------------------------------------------------------------------

WORD_WEIGHTS = {
    # Strong positive
    "love": 2, "amazing": 2, "incredible": 2, "outstanding": 2,
    "fantastic": 2, "brilliant": 2, "thrilled": 2, "blessed": 2,
    "perfect": 2, "superb": 2, "wonderful": 2,
    # Strong negative
    "hate": -3, "terrible": -2, "horrible": -2, "disgusting": -2,
    "miserable": -2, "devastated": -2, "furious": -2, "unbearable": -2,
    "heartbroken": -2, "awful": -2, "dreadful": -2, "pathetic": -2,
}

# ---------------------------------------------------------------------
# Starter labeled dataset (expanded to 20 posts)
# ---------------------------------------------------------------------

SAMPLE_POSTS = [
    # Original 6
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    # --- Added posts (diverse styles) ---
    "Lowkey stressed but kind of proud of myself",
    "This homework is actually fire 🔥",
    "I absolutely love getting stuck in traffic 😒",
    "Just vibing with friends, life is good ✨",
    "Can't stop smiling today 😊 everything went perfect",
    "Bruh this is the worst exam I've ever taken 💀",
    "Feeling meh, nothing special happening",
    "No cap, that presentation was amazing",
    "My heart is broken 💔 I trusted them",
    "Grateful for the little things in life ❤️",
    "I'm not sad, I'm just tired of everything",
    "Yikes, that was cringe af",
    "Finally done with finals!!! 🎉🎉🎉 Freedom at last!",
    "It's whatever, I don't really care anymore",
]

# Human labels for each post above.
TRUE_LABELS = [
    # Original 6
    "positive",   # "I love this class so much"
    "negative",   # "Today was a terrible day"
    "mixed",      # "Feeling tired but kind of hopeful"
    "neutral",    # "This is fine"
    "positive",   # "So excited for the weekend"
    "negative",   # "I am not happy about this"
    # --- Labels for added posts ---
    "mixed",      # "Lowkey stressed but kind of proud of myself"
    "positive",   # "This homework is actually fire 🔥"
    "negative",   # "I absolutely love getting stuck in traffic 😒" (sarcasm)
    "positive",   # "Just vibing with friends, life is good ✨"
    "positive",   # "Can't stop smiling today 😊 everything went perfect"
    "negative",   # "Bruh this is the worst exam I've ever taken 💀"
    "neutral",    # "Feeling meh, nothing special happening"
    "positive",   # "No cap, that presentation was amazing"
    "negative",   # "My heart is broken 💔 I trusted them"
    "positive",   # "Grateful for the little things in life ❤️"
    "mixed",      # "I'm not sad, I'm just tired of everything"
    "negative",   # "Yikes, that was cringe af"
    "positive",   # "Finally done with finals!!! 🎉🎉🎉 Freedom at last!"
    "neutral",    # "It's whatever, I don't really care anymore"
]

# Sanity check
assert len(SAMPLE_POSTS) == len(TRUE_LABELS), (
    f"Mismatch: {len(SAMPLE_POSTS)} posts vs {len(TRUE_LABELS)} labels"
)
