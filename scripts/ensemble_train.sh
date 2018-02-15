#!/bin/bash
export netType='wide-resnet'
export depth=28
export width=10
export dataset='cifar100'
export experiment_number=1
mkdir -p modelState

CUDA_VISIBLE_DEVICES=0 th main.lua \
    -dataset ${dataset} \
    -netType ${netType} \
    -top5_display true \
    -testOnly false \
    -resume modelState \
    -dropout 0.3 \
    -batchSize 128 \
    -depth ${depth} \
    -widen_factor ${width} \
    -nExperiment ${experiment_number} \
    | tee logs/${dataset}/${netType}-${depth}x${width}/log_${experiment_number}.txt 


export width=20
export experiment_number=1

CUDA_VISIBLE_DEVICES=0 th main.lua \
    -dataset ${dataset} \
    -netType ${netType} \
    -top5_display true \
    -testOnly false \
    -resume modelState \
    -dropout 0.3 \
    -batchSize 128 \
    -depth ${depth} \
    -widen_factor ${width} \
    -nExperiment ${experiment_number} \
    | tee logs/${dataset}/${netType}-${depth}x${width}/log_${experiment_number}.txt 

export experiment_number=2

CUDA_VISIBLE_DEVICES=0 th main.lua \
    -dataset ${dataset} \
    -netType ${netType} \
    -top5_display true \
    -testOnly false \
    -dropout 0.3 \
    -batchSize 128 \
    -depth ${depth} \
    -widen_factor ${width} \
    -nExperiment ${experiment_number} \
    | tee logs/${dataset}/${netType}-${depth}x${width}/log_${experiment_number}.txt 


export experiment_number=3

CUDA_VISIBLE_DEVICES=0 th main.lua \
    -dataset ${dataset} \
    -netType ${netType} \
    -resume modelState \
    -top5_display true \
    -testOnly false \
    -dropout 0.3 \
    -batchSize 128 \
    -depth ${depth} \
    -widen_factor ${width} \
    -nExperiment ${experiment_number} \
    | tee logs/${dataset}/${netType}-${depth}x${width}/log_${experiment_number}.txt 



export depth=40
export width=14
export experiment_number=1



CUDA_VISIBLE_DEVICES=0 th main.lua \
    -dataset ${dataset} \
    -netType ${netType} \
    -top5_display true \
    -testOnly false \
    -dropout 0.3 \
    -batchSize 128 \
    -depth ${depth} \
    -widen_factor ${width} \
    -nExperiment ${experiment_number} \
    | tee logs/${dataset}/${netType}-${depth}x${width}/log_${experiment_number}.txt 



export width=10

CUDA_VISIBLE_DEVICES=0 th main.lua \
    -dataset ${dataset} \
    -netType ${netType} \
    -top5_display true \
    -testOnly false \
    -dropout 0.3 \
    -batchSize 128 \
    -depth ${depth} \
    -widen_factor ${width} \
    -nExperiment ${experiment_number} \
    | tee logs/${dataset}/${netType}-${depth}x${width}/log_${experiment_number}.txt 

./script/ensemble.sh
