# 🧠 Text Origin Classifier

A classifier that detects the origin of text based on perplexity scores derived from open-source language models.

**Classes:**
- Open-source LLM-generated
- Closed-source LLM-generated (e.g., ChatGPT, Claude), Human-written

## 🛠️ Approach

1. **Compute perplexity** for each input text using open-source models:
   - `Llama2`
   - `Llama3`
   - `Qwen`

2. **Build feature vectors** from perplexity scores.

3. **Train a classifier** to predict the text origin:
   - Model: `CatBoost`
   - Input: Perplexity features
   - Target: `open-source`, `closed-source, human`

## 📈 Results

Classifier performance is summarized below:
### Train
![image](https://github.com/user-attachments/assets/52a76f44-c9f3-4686-a6f4-32757b2520c0)
### Test
![image](https://github.com/user-attachments/assets/791c38af-3923-4b25-813c-1a0b487aff08)
- Classifier `struggles to reliably distinguish` open-source and close-sorce, human-written text.
