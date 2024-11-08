#!/bin/bash

pip install -r /content/NLP_Advanced/scripts/requirements.txt --quiet

python /content/NLP_Advanced/scripts/train_mlm.py \
        --model_name "microsoft/mdeberta-v3-base" \
        --batch_size 16 \
        --num_train_epochs 10
: <<'EOF'
python /content/NLP_Advanced/scripts/train_ce.py \
        --model_name "microsoft/mdeberta-v3-base" \
        --batch_size 64 \
        --num_train_epochs 10
# -------------------------------------------------
python /content/NLP_Advanced/scripts/train_mlm.py \
        --model_name "bert-base-multilingual-cased" \
        --batch_size 128 \
        --num_train_epochs 10

python /content/NLP_Advanced/scripts/train_ce.py \
        --model_name "bert-base-multilingual-cased" \
        --batch_size 64 \
        --num_train_epochs 10
# -------------------------------------------------
python /content/NLP_Advanced/scripts/train_mlm.py \
        --model_name "hiiamsid/sentence_similarity_spanish_es" \
        --batch_size 128 \
        --num_train_epochs 10 \
        --language "esp"

python /content/NLP_Advanced/scripts/train_ce.py \
        --model_name "hiiamsid/sentence_similarity_spanish_es" \
        --batch_size 64 \
        --num_train_epochs 10 \
        --language "esp"
# -------------------------------------------------
python /content/NLP_Advanced/scripts/train_mlm.py \
        --model_name "l3cube-pune/marathi-sentence-similarity-sbert" \
        --batch_size 128 \
        --num_train_epochs 10 \
        --language "mar"

python /content/NLP_Advanced/scripts/train_ce.py \
        --model_name "l3cube-pune/marathi-sentence-similarity-sbert" \
        --batch_size 64 \
        --num_train_epochs 10 \
        --language "mar"
 # -------------------------------------------------       
python /content/NLP_Advanced/scripts/train_mlm.py \
        --model_name "l3cube-pune/telugu-sentence-similarity-sbert" \
        --batch_size 64 \
        --num_train_epochs 10 \
        --language "tel"

python /content/NLP_Advanced/scripts/train_ce.py \
        --model_name "l3cube-pune/telugu-sentence-similarity-sbert" \
        --batch_size 64 \
        --num_train_epochs 10 \
        --language "tel"
# -------------------------------------------------
python /content/NLP_Advanced/scripts/train_mlm.py \
        --model_name "mbeukman/xlm-roberta-base-finetuned-kinyarwanda-finetuned-ner-swahili" \
        --batch_size 128 \
        --num_train_epochs 10 \
        --language "kin"

python /content/NLP_Advanced/scripts/train_ce.py \
        --model_name "mbeukman/xlm-roberta-base-finetuned-kinyarwanda-finetuned-ner-swahili" \
        --batch_size 64 \
        --num_train_epochs 10 \
        --language "kin"
# -------------------------------------------------
python /content/NLP_Advanced/scripts/train_mlm.py \
        --model_name "microsoft/deberta-v3-large" \
        --batch_size 128 \
        --num_train_epochs 10 \
        --language "eng"

python /content/NLP_Advanced/scripts/train_ce.py \
        --model_name "microsoft/deberta-v3-large" \
        --batch_size 64 \
        --num_train_epochs 10 \
        --language "eng"
# -------------------------------------------------
python /content/NLP_Advanced/scripts/train_mlm.py \
        --model_name "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" \
        --batch_size 128 \
        --num_train_epochs 10 \

python /content/NLP_Advanced/scripts/train_ce.py \
        --model_name "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" \
        --batch_size 64 \
        --num_train_epochs 10 \
# intfloat/multilingual-e5-base
python /content/NLP_Advanced/scripts/train_mlm.py \
        --model_name "intfloat/multilingual-e5-base" \
        --batch_size 128 \
        --num_train_epochs 10 \

python /content/NLP_Advanced/scripts/train_ce.py \
        --model_name "intfloat/multilingual-e5-base" \
        --batch_size 64 \
        --num_train_epochs 10 \
# intfloat/multilingual-e5-base
python /content/NLP_Advanced/scripts/train_mlm.py \
        --model_name "intfloat/multilingual-e5-large" \
        --batch_size 128 \
        --num_train_epochs 10 \

python /content/NLP_Advanced/scripts/train_ce.py \
        --model_name "intfloat/multilingual-e5-large" \
        --batch_size 64 \
        --num_train_epochs 10 \
# --------------------------------------------------
python /content/NLP_Advanced/scripts/train_mlm.py \
        --model_name "intfloat/e5-large-v2" \
        --batch_size 128 \
        --num_train_epochs 10 \
        --language "eng"

python /content/NLP_Advanced/scripts/train_ce.py \
        --model_name "intfloat/e5-large-v2" \
        --batch_size 64 \
        --num_train_epochs 10 \
        --language "eng"
zip -r submission.zip submission
EOF