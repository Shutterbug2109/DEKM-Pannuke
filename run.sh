#!/bin/bash

# run the experiments with different combinations of parameters
for dataset in 'PANNUKE' 'PANNUKE_ONLYCELLS' 'PANNUKE_DILATED'; do
    for n_clusters in 2 3 4 5 6; do
        for epoch in 200 300 400; do
            for hidden_units in 10 20; do
                echo 'For dataset :' $dataset 'and for n_clusters:' $n_clusters ',Number of Epochs are:' $epoch 'with number of hidden units:'$hidden_units
                python main.py --ds_name "${dataset}"  --n_clusters "${n_clusters}" --pretrain_epochs "${epoch}" --hidden_units "${hidden_units}" 
            done
        done
    done
done
