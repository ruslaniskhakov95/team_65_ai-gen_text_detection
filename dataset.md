# Cloud link with datasets
https://drive.google.com/drive/folders/1DoPzLu4UhLbjeh_eNSYaBT4GjrygIa87?usp=drive_link

# Описание данных

## train_v3_drcat_02.csv
Датасет с 65333 уникальными текстами эссе на 15 промптов.
27371 - написаны людьм
38137 - сгенерированы моделями

## train_essays.jsonl
Датасет с 2075 уникальными текстами эссе на 2 промпта.
1375 - написаны людьми
700 - сгенерированы моделями

## train_prompts.csv
Датасет с текстом промптов для train_essays.jsonl

## LLM_generated_essay_PaLM.csv
Датасет со сгенерированными данными по промптам из train_prompts.csv при помощи модели PaLM2
1384 - сгенерированы моделью

## train_contexts.pkl
Содержит датасет с 14400 промптами, в ответ накоторые 3 LLM и люди генерировали ответы в следующих файлах

## train_humans.pkl
Датасет из 14400 текстов, написанных людьми в ответ на промпты из файла train_contexts.pkl

## train_lms_gpt.pkl
Датасет из 14400 текстов, написанных ChatGPT в ответ на промпты из файла train_contexts.pkl

## train_lms_flan.pkl
Датасет из 14400 текстов, написанных Flan_T5 в ответ на промпты из файла train_contexts.pkl

## train_lms_davinci.pkl
Датасет из 14400 текстов, написанных DaVinci в ответ на промпты из файла train_contexts.pkl
