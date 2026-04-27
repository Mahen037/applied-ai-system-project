# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **three** integrated components:

1. A **rule-based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit-learn
3. A **RAG explainer** implemented in `rag_explainer.py` for context-aware explanations

All three are orchestrated by `pipeline.py` into a unified prediction system.

## 1. Model Overview

**Model type:**
I used both models in an ensemble configuration. The rule-based model provides interpretable scoring with negation and emoji handling, while the ML model (TF-IDF + Logistic Regression) learns statistical patterns. An ensemble layer combines their predictions, and a RAG system provides explanations grounded in similar examples.

**Intended purpose:**
Classify short text messages (social media posts, messages) as one of four moods: positive, negative, neutral, or mixed. The system also provides a confidence score (0-1) and a detailed explanation for each prediction.

**How it works (brief):**
- **Rule-based:** Preprocesses text (tokenization, emoji extraction, normalization), then scores it by counting weighted positive/negative words with negation detection. Emojis contribute their own sentiment scores. The final score maps to a label via thresholds.
- **ML model:** Converts text to TF-IDF bigram features, then uses Logistic Regression to predict class probabilities. The highest-probability class becomes the prediction.
- **Ensemble:** If both models agree, confidence is boosted. If they disagree, the more confident model wins but overall confidence is reduced.
- **RAG:** Embeds the input using TF-IDF, retrieves the 3 most similar examples from the labeled dataset, and generates an explanation comparing the prediction to similar past examples.

## 2. Data

**Dataset description:**
The dataset contains 20 short text posts in `SAMPLE_POSTS` with corresponding labels in `TRUE_LABELS`. The original starter had 6 posts; I added 14 more to cover diverse language styles.

**Labeling process:**
All labels were assigned by me (human labeling). Several posts were genuinely difficult to label:
- "Feeling tired but kind of hopeful" — could be negative OR mixed; I chose "mixed"
- "I absolutely love getting stuck in traffic 😒" — sarcastic; I labeled it "negative" despite the word "love"
- "I'm not sad, I'm just tired of everything" — complex negation; labeled "mixed" because the denial of sadness combined with exhaustion suggests conflicting emotions
- "It's whatever, I don't really care anymore" — apathy could be neutral or negative; I chose "neutral"

**Important characteristics of your dataset:**
- Contains emojis: 😊, 🔥, 😒, ✨, 💀, 💔, ❤️, 🎉
- Includes sarcasm ("I absolutely love getting stuck in traffic")
- Contains slang: "bruh", "no cap", "vibing", "cringe af", "fire", "lit"
- Some posts express mixed feelings with contrasting signals
- Short messages (averaging ~8 words)

**Possible issues with the dataset:**
- **Small size**: 20 examples is too few for robust ML training; the ML model likely overfits
- **Label imbalance**: 8 positive, 6 negative, 3 neutral, 3 mixed — slightly skewed toward positive
- **Cultural/age bias**: Mostly Gen-Z slang and informal English; wouldn't generalize to formal text
- **Single annotator**: Only I labeled the data; disagreement rates could reveal ambiguity

## 3. How the Rule-Based Model Works

**Your scoring rules:**
- **Positive/negative word matching**: 44 positive words and 42 negative words, each adding +1/-1 to the score
- **Weighted words**: High-impact words like "love" (+2), "hate" (-3), "amazing" (+2), "terrible" (-2) have custom weights
- **Negation handling**: 36 negation words (including contractions like "don't", "can't") flip the sentiment of the next word. "Not happy" → negative, "not bad" → positive
- **Emoji scoring**: 32 emojis mapped to sentiment values: 😊(+2), 😢(-2), 🔥(+1), 💀(-1), etc.
- **Label thresholds**: score > 0 → positive, score < 0 → negative, score == 0 → neutral, mixed (conflicting signals near zero) → mixed

**Strengths of this approach:**
- Fully interpretable: every scoring decision can be traced
- Handles negation well: "I am not happy" correctly predicts negative
- Emoji-aware: 😊 and 💀 contribute real signal
- No training needed: works immediately on any input
- Fast: no model loading or inference cost

**Weaknesses of this approach:**
- **Sarcasm**: Completely misses "I absolutely love getting stuck in traffic" (scores positive due to "love")
- **Unknown words**: Any word not in the dictionaries is invisible to the model
- **Context blindness**: Treats each word independently; "fine" in "This is fine" vs "fine dining" vs "feeling fine" all score the same
- **Fixed thresholds**: The positive/negative cutoff at score 0 may be too aggressive

## 4. How the ML Model Works

**Features used:**
TF-IDF vectors with unigrams and bigrams (max 500 features), using `TfidfVectorizer` from scikit-learn.

**Training data:**
The model trained on the same 20 `SAMPLE_POSTS` with `TRUE_LABELS` from dataset.py.

**Training behavior:**
- Training accuracy: 100% (overfits on 20 examples, as expected)
- The model learned useful bigrams like "not happy" and "so excited" as features
- Adding more examples improved the ML model's diversity but accuracy stayed near 100% on training data due to overfitting

**Strengths and weaknesses:**
- **Strengths**: Learns patterns automatically; picks up bigrams the rule-based model might miss; provides probability distributions across all classes
- **Weaknesses**: Overfits heavily with only 20 training examples; can't generalize to unseen vocabulary; treats every text as a bag of words (no word order beyond bigrams)

## 5. Evaluation

**How you evaluated the model:**
The full pipeline was evaluated on the 20 labeled posts using `pipeline.evaluate()`. Both models' predictions are combined via ensemble logic, and the final label is compared to the true label.

**Accuracy: 90% (18/20)**

| Metric | Value |
|---|---|
| Total posts | 20 |
| Correct | 18 |
| Accuracy | 90% |
| Avg. confidence | 0.70 |

**Examples of correct predictions:**

1. **"I am not happy about this"** → Predicted: negative ✅
   Why correct: Negation handling correctly identified "not happy" as a negative signal. Score = -1, confidence = 0.83.

2. **"Bruh this is the worst exam I've ever taken 💀"** → Predicted: negative ✅
   Why correct: "worst" is a strong negative word, and 💀 emoji contributes -1. Slang "bruh" doesn't affect the score, which is fine here.

3. **"Lowkey stressed but kind of proud of myself"** → Predicted: mixed ✅
   Why correct: "stressed" (negative) and "proud" (positive) create conflicting signals. Score = 0, both positive and negative signals present → mixed.

**Examples of incorrect predictions:**

1. **"This is fine"** → Predicted: positive ❌ (True: neutral)
   Why wrong: The ML model pushed this to positive because "fine" appeared in training contexts. The rule-based model returned neutral (no signals), but the ensemble was swayed by the ML model's confidence.

2. **"I absolutely love getting stuck in traffic 😒"** → Predicted: mixed ❌ (True: negative)
   Why wrong: This is sarcasm. The rule-based model sees "love" (+2) and 😒 (-1), resulting in a mixed signal. The system has no sarcasm detection, so it can't tell that "love" is being used ironically.

## 6. Limitations

- **Small dataset**: 20 examples is insufficient for robust ML training; the model overfits
- **No sarcasm detection**: Sarcastic use of positive words completely fools the system
- **English only**: Only works with English text; other languages would need new word lists
- **Short text only**: Designed for ~1-2 sentences; longer text may accumulate misleading signals
- **Binary-style word matching**: Ignores word context, word order (beyond bigrams), and semantic meaning
- **No real-time learning**: The system doesn't improve from user feedback or corrections
- **Emoji coverage gaps**: Only 32 emojis are mapped; new/uncommon emojis are ignored

## 7. Ethical Considerations

- **Misclassifying distress**: A message expressing genuine sadness or a cry for help could be labeled "neutral" or "mixed" if the words aren't in the dictionary. In a mental health context, this could be dangerous.
- **Cultural and linguistic bias**: The word lists and emoji mappings reflect American English and Gen-Z internet culture. Users from different backgrounds may use different expressions that the model misinterprets.
- **Sarcasm and irony**: Mislabeling sarcastic messages as positive could lead to inappropriate responses in an automated system.
- **Privacy**: Mood analysis of personal messages raises serious privacy concerns if deployed without consent.
- **Consent and transparency**: Users should always know their text is being analyzed and how the scores/labels are determined. The explain() feature helps with this.

## 8. Ideas for Improvement

- **Add a separate test set**: Split data into train/test to measure generalization
- **Sarcasm detection**: Use pattern matching ("love...but 😒") or a dedicated sarcasm classifier
- **Sentence-transformers**: Use deep learning embeddings (e.g., all-MiniLM-L6-v2) for better RAG retrieval
- **Expand dataset to 100+**: Cover more emotions, formality levels, and languages
- **User feedback loop**: Let users correct predictions and retrain the model
- **Fine-grained emotions**: Add labels like "angry", "anxious", "excited", "grateful" beyond the 4 basic moods
- **Web interface**: Build a browser-based UI for interactive exploration
- **Multi-annotator labeling**: Have multiple people label each post and measure inter-annotator agreement
