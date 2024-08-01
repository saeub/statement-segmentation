#!/bin/bash

mkdir -p models

for bert_lang in german multilingual; do
    for num_epochs in 1 2; do
        for batch_size in 8 16 32; do
            for augment_samples in 0 1472 2944; do
                model=models/bert-${bert_lang}-aug_none_${augment_samples}-${num_epochs}ep
                mkdir -p $model
                python src/cli.py train \
                    --base-model google-bert/bert-base-$bert_lang-cased \
                    --trainset data/train.csv \
                    --augment-trainset data/train.csv \
                    --augment-samples $augment_samples \
                    --devset data/test.csv \
                    --num-epochs $num_epochs \
                    --batch-size $batch_size \
                    --save-to $model \
                    > $model/log.txt 2> $model/stderr.txt
            done
            for augment_trainset in single multiple; do
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
                        --batch-size $batch_size \
                        --save-to $model \
                        > $model/log.txt 2> $model/stderr.txt
                done
            done
        done
    done
done
