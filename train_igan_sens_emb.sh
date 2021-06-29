ISERROR=0
DATA=$1
EMB=$2
GPU_NUM=$3

if [ -z $GPU_NUM ]; then
    ISERROR=1
elif [ $DATA = "adult" ]; then
    python iGAN.py --data ${DATA} --likelihood_coef -0.014 --embedding_dim ${EMB} --test_name "igan_emb"${EMB} --GPU_NUM ${GPU_NUM}
elif [ $DATA = "news" ]; then
    python iGAN.py --data ${DATA} --likelihood_coef -0.01 --embedding_dim ${EMB} --test_name "igan_emb"${EMB} --GPU_NUM ${GPU_NUM}
else
    ISERROR=1
    
fi

if [ $ISERROR = 1 ]; then
    echo "first option : data [adult, news]"
    echo "second option : embedding size <int>"
    echo "third option : GPU number to use <int>"
    echo "example : 'sh train_igan_sens_emb.sh adult 128 1'"
fi
