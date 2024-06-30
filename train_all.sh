#!/bin/bash

mkdir -p models

for bert_lang in german multilingual; do
    for num_epochs in 1 2; do
        model=models/bert-${bert_lang}-aug_none-${num_epochs}ep
        mkdir -p $model
        python src/cli.py train \
            --base-model google-bert/bert-base-$bert_lang-cased \
            --trainset data/train.csv \
            --devset data/test.csv \
            --num-epochs $num_epochs \
            --save-to $model \
            > $model/log.txt 2> $model/stderr.txt
        for augment_trainset in v2 v3; do
            for augment_samples in 1472 2944; do
                model=models/bert-${bert_lang}-aug_${augment_trainset}_${augment_samples}-${num_epochs}ep
                mkdir -p $model
                python src/cli.py train \
                    --base-model google-bert/bert-base-$bert_lang-cased \
                    --trainset data/train.csv \
                    --augment-trainset data/train_augmented_$augment_trainset.csv \
                    --augment-samples $augment_samples \
                    --devset data/test.csv \
                    --num-epochs $num_epochs \
                    --save-to $model \
                    > $model/log.txt 2> $model/stderr.txt
            done
        done
    done
done
