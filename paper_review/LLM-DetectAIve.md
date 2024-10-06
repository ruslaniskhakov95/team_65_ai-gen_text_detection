# LLM-DetectAIve

**Intro:**

We can classify llm-generated text into four general categories: human-written, machine-generated, machine-written machine-humanized, and human-written machine-polished.

Problem with previous approaches: MGT (machine generated text) classifiers usually perform binary classification, unlike LLM-DetectAlve, which classifies text into four categories (as noted above).

Why is MGT detection so problematic: newer LLM’s produce more coherent texts, while [advrsarial attacks](https://arxiv.org/abs/2403.19148) provide promt-engeneering approaches fo bypassing MGT detectors.

**Dataset:**

Authors built upon [M4GT](https://github.com/mbzuai-nlp/M4GT-Bench?tab=readme-ov-file) dataset, which provides training data and bechmarks for binary MGT classififcation:

- About the dataset:
    - Human-generated texts + texts generated with APIs of popular LLMs: [M4GT](https://github.com/mbzuai-nlp/M4GT-Bench?tab=readme-ov-file)
    - Domains: arXiv, Wikihow, Wikipedia,
    Reddit, student essays (OUTFOX), and peer reviews (PeerRead)
    - Usefull info: Their github contains notebooks with dataset collection and code for training binary MGT classifiers.
- Dataset augmentation:
    - MGT: generate more texts with newer LLMs (like GPT4-o).
    - MGT Machine-humanized: subsample texts from M4GT, use adversarial attacks to make texts more human-like.
    - Machine-polished: subsample human-generated texts from M4GT, polish them with with LLMs.
    - Human-written: no augmentation
- Dataset post-processing: remove texts like “Sure, here is the answer!”, since LLMs sometimes append them to their output. This allows classifiers to provide better generalization.

**Models:**

- Why DeBERTa?: because SOTA)
- Why DistillBERT: faster inference to provide instant response to users.
- Why RoBerta?: Just to compare it with the performance of DeBERTa.

**Models Expalined:**

- Models used: RoBERTa, DeBERTa, DistillBERT. I have also provided some side notes on the models in comparison to classic BERT. a comprehensive overview [here](https://habr.com/ru/articles/680986/) and [here](https://towardsdatascience.com/large-language-models-deberta-decoding-enhanced-bert-with-disentangled-attention-90016668db4b).
    - RoBERTa:
        - Longer training, more data.
        - Bigger batch size during training.
        - Larger vocab.
        - Longer input sequences during training.
        - Dynamic masking.
        - Trained only for predicting the missing token (BERT was also trained for predicting the next sentence).
    - DistillBERT:
        - A distilled version of BERT) (uses less parameters, trained using tripple loss to increase the similarity with outputs of BERT).
    - DeBERTA:
        - Disentagled attention mechanism, which uses two vectors to represent a token. One of the vectors contains context information, while the other one represents the relative postional information. This allows to store content-to-content, content-to-position, position-to-content, and position-to-position information.
        - Also uses enchanced mask decoder, which incorporates absolute position of the tokens before applying the softmax operation to the output layer (in BERT this is done at the input layer).

**Approaches:**

1. Several domain-specific detectors:
    1. Pros: good accuracy.
    2. Cons: users have to specify the domain of their text.
2. Universal-detector:
    1. Pros: good performance.
    2. Cons: possibly bad out-of-domain performance (authors have not really tested that).
3. DANN-detector:
    1. We add a parrallell classification head (hence we will have a classifier for MGT detection and another classifier for domains). The domain classifier multiplies the gradient by a negative constant, which makes features more domain invariant.
    2. This approach achieved the best overall performance.

**Deployment:**

You can play with the detector [here](https://huggingface.co/spaces/raj-tomar001/MGT-New) (also contains source code).