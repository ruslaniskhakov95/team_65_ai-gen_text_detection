# Baseline Summary:
I have used classical tfidf/word2vec/glove approaches to make the baseline solutions
- I summary: tfidf+logreg is sota for all three datasets.
- I have used taken only the subset to train SVC models due to long training/inference and suboptimal performance.
- I have not yet used any transformer architectures (which was the proposed solution by the authors). I will see whether there are possible improvements to the solution proposed by the authors in later checkpoints.
- It is propably possible to improve the baseline solutions by several percents, but it is unlikely to beat the near perect performance of transformer models (hence I deciided not to waste time on grid searchiing the parameters).
- Word2vec and glove embeddings are stored locally, but are not uploaded to the repository (due to poor performance).
- For task "C" I have excluded class for LLM-enhanced texts (it was not provided by the authors, possibly will have to synthesise it myself).

# Here are the best results:
Note: all of those are tfidf+logreg
## Binary classification (Subtask-A/dataset_tests.ipynb)

              precision    recall  f1-score   support

           0       0.87      0.85      0.86     19574
           1       0.89      0.91      0.90     26269

    accuracy                            0.88     45843
    macro avg       0.88      0.88      0.88     45843
    weighted avg    0.88      0.88      0.88     45843

## Generator Detection (Subtask-B/dataset_multilabel.ipynb)

              precision    recall  f1-score   support

     chatGPT       0.78      0.83      0.81      6870
       human       0.78      0.79      0.78      5193
      cohere       0.73      0.73      0.73      4971
     davinci       0.80      0.75      0.77      5237
      bloomz       0.95      0.97      0.96      5264
       dolly       0.69      0.67      0.68      5019
       gpt-4       0.83      0.81      0.82      4290

    accuracy                           0.80     36844
    macro avg      0.79      0.79      0.79     36844
    weighted avg   0.79      0.80      0.79     36844

## Authorship Boundary (Subtask-C/dataset_mixed.ipynb)

          precision    recall  f1-score   support

       human       0.87      0.94      0.90      1931
       mixed       0.74      0.65      0.69      1961
     machine       0.77      0.79      0.78      1948

    accuracy                           0.80      5840
    macro avg      0.79      0.80      0.79      5840
    weighted avg   0.79      0.80      0.79      5840
