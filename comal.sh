NAME="COMAL"  # name of the experiment
OLD_CKPT="yale-nlp/comal-mle" # at the first iteration, this should be the path to the MLE model
REF_CKPT="yale-nlp/comal-mle" # at the first iteration, this should be the path to the MLE model

GPU_IDS=0,1,2,3,4,5,6,7   # GPU IDs, here we assume 8 GPUs, please change this if you have different number of GPUs
GPU_LIST="0 1 2 3 4 5 6 7"  # GPU list
NUM_GPUS=8  # number of GPUs

# COMAL hyperparameters
NUM_ITERS=24  # number of iterations
ITER_START=0  # starting iteration
UPDATE_INTERVAL=12 # interval to update the reference model

# Training hyperparameters
PORT=29320  # port number for FSDP
EPOCHS=3  # number of epochs
RATIO=0.3333333333333333  # tau/eta ratio
ETA=0.002  # eta
ACCUMULATE_STEP=4  # accumulate step
BATCH_SIZE=1  # local batch size, the actual batch size is BATCH_SIZE * NUM_GPUS * ACCUMULATE_STEP

echo "GPU_IDS: $GPU_IDS"

for ((ITER=$ITER_START; ITER<$ITER_START+$NUM_ITERS; ITER++))
    do
        echo "$NAME ITER $ITER"
        FDIR="exps/$NAME/$ITER"
        mkdir -p $FDIR
        echo "FDIR: $FDIR"
        DATA_ITER=$(( ITER % 6 ))  # we have 6 different training data splits

        # generate samples
        echo "Generating samples"
        CUDA_VISIBLE_DEVICES=$GPU_IDS python sampling.py \
                            --num_gpus $NUM_GPUS \
                            --model_type qwen \
                            --model_pt $OLD_CKPT/model \
                            --tokenizer_pt Qwen/Qwen2-1.5B \
                            --num_samples 5  \
                            --top_p 0.95 \
                            --input_dir data/prompts/test.jsonl \
                            --output_dir $FDIR/test.samples.jsonl \
                            --gpuids $GPU_LIST \
                            --num_workers 8

        CUDA_VISIBLE_DEVICES=$GPU_IDS python sampling.py \
                            --num_gpus $NUM_GPUS \
                            --model_type qwen \
                            --model_pt $OLD_CKPT/model \
                            --tokenizer_pt Qwen/Qwen2-1.5B \
                            --num_samples 5  \
                            --top_p 0.95 \
                            --input_dir data/prompts/train_${DATA_ITER}.jsonl \
                            --output_dir $FDIR/train.samples.jsonl  \
                            --gpuids $GPU_LIST \
                            --num_workers 8

        # score samples
        echo "Scoring samples"
        CUDA_VISIBLE_DEVICES=$GPU_IDS python scoring.py \
            --src_dir data/prompts/train_${DATA_ITER}.jsonl \
            --input_dir $FDIR/train.samples.jsonl \
            --output_dir  $FDIR/train.samples.pairs.jsonl \
            --gpuids $GPU_LIST \
            --model_pt NCSOFT/Llama-3-OffsetBias-8B \
            --batch_size 16 \
            --score_mode pairwise \
            --model_type offsetbias-lm \
            --num_workers 8

        CUDA_VISIBLE_DEVICES=$GPU_IDS python scoring.py \
            --src_dir data/prompts/test.jsonl \
            --input_dir $FDIR/test.samples.jsonl \
            --output_dir  $FDIR/test.samples.pairs.jsonl \
            --gpuids $GPU_LIST \
            --model_pt NCSOFT/Llama-3-OffsetBias-8B \
            --batch_size 16 \
            --score_mode pairwise \
            --model_type offsetbias-lm \
            --num_workers 8

        # processing samples
        echo "Processing samples"
        python data_processing.py \
            --task make_output_pair_from_pm \
            --input_dir $FDIR/test.samples.pairs.jsonl \
            --output_dir $FDIR/test.pairs.jsonl \
            --num_workers 16 \
            --tokenizer_pt Qwen/Qwen2-1.5B \
            --model_type qwen \
            --pm_tokenizer_pt NCSOFT/Llama-3-OffsetBias-8B

        python data_processing.py \
            --task make_output_pair_from_pm \
            --input_dir $FDIR/train.samples.pairs.jsonl \
            --output_dir $FDIR/train.pairs.jsonl \
            --num_workers 16 \
            --tokenizer_pt Qwen/Qwen2-1.5B \
            --model_type qwen \
            --pm_tokenizer_pt NCSOFT/Llama-3-OffsetBias-8B
            
        mkdir -p $FDIR/data
        # get logprobs
        echo "Getting logprobs using the latest model"
        CUDA_VISIBLE_DEVICES=$GPU_IDS python get_logprobs.py \
            --input_dir $FDIR/train.pairs.jsonl \
            --gpuids $GPU_LIST \
            --output_dir $FDIR/data/train.jsonl \
            --model_type qwen \
            --model_pt $OLD_CKPT/model \
            --tokenizer_pt Qwen/Qwen2-1.5B

        CUDA_VISIBLE_DEVICES=$GPU_IDS python get_logprobs.py \
            --input_dir $FDIR/test.pairs.jsonl \
            --gpuids $GPU_LIST \
            --output_dir $FDIR/data/test.jsonl \
            --model_type qwen \
            --model_pt $OLD_CKPT/model \
            --tokenizer_pt Qwen/Qwen2-1.5B

        echo "Getting logprobs using the ref model"
        CUDA_VISIBLE_DEVICES=$GPU_IDS python get_logprobs.py \
            --input_dir $FDIR/data/train.jsonl \
            --gpuids $GPU_LIST \
            --output_dir $FDIR/data/train.jsonl \
            --model_type qwen \
            --model_pt $REF_CKPT/model \
            --tokenizer_pt Qwen/Qwen2-1.5B \
            --mode nash

        CUDA_VISIBLE_DEVICES=$GPU_IDS python get_logprobs.py \
            --input_dir $FDIR/data/test.jsonl \
            --gpuids $GPU_LIST \
            --output_dir $FDIR/data/test.jsonl \
            --model_type qwen \
            --model_pt $REF_CKPT/model \
            --tokenizer_pt Qwen/Qwen2-1.5B \
            --mode nash

        # training
        echo "Training"
        
        CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --config_file fsdp_config.yaml  \
            --main_process_port $PORT \
            inpo.py \
            --epoch $EPOCHS \
            --eta $ETA \
            --tau_eta_ratio $RATIO \
            --dataset $FDIR/data \
            --pretrained $OLD_CKPT/model \
            --exp_name $FDIR/ckpts \
            --accumulate_step $ACCUMULATE_STEP \
            --batch_size $BATCH_SIZE \
            -l
      

        CKPT=$FDIR/ckpts
        # evaluate
        echo "generating evaluation samples"
        CUDA_VISIBLE_DEVICES=$GPU_IDS python eval.py \
            --num_gpus 4 \
            --model_type qwen \
            --model_pt $CKPT/model \
            --tokenizer_pt Qwen/Qwen2-1.5B \
            --num_samples 1  \
            --temperature 0.7 \
            --top_p 0.95 \
            --output_dir $CKPT/alpacaeval_output.jsonl \
            --task gen_alpaca

        echo "Comparing with sft on alpaca"
        CUDA_VISIBLE_DEVICES=$GPU_IDS python eval.py \
            --output_dir $CKPT/alpacaeval_vs_sft.json \
            --sys1_dir ckpts/qwen_mle/alpacaeval_output.jsonl \
            --sys2_dir $CKPT/alpacaeval_output.jsonl \
            --num_gpus $NUM_GPUS \
            --batch_size 16 \
            --task eval_alpaca

        echo "Comparing with previous ckpt on alpaca"
        CUDA_VISIBLE_DEVICES=$GPU_IDS python eval.py \
            --output_dir $CKPT/alpacaeval_vs_previous.json \
            --sys1_dir $OLD_CKPT/alpacaeval_output.jsonl \
            --sys2_dir $CKPT/alpacaeval_output.jsonl \
            --num_gpus $NUM_GPUS \
            --batch_size 16 \
            --task eval_alpaca

        OLD_CKPT=$CKPT

        if (( (ITER + 1) % UPDATE_INTERVAL == 0 )); then
            echo "Updating the reference model"
            REF_CKPT=$CKPT
        fi
    done

