ISERROR=0
DATA=$1
COEF=$2
GPU_NUM=$3

if [ -z $GPU_NUM ]; then
    ISERROR=1
elif [ $DATA != "adult" ] && [ $DATA != "news" ]; then
    ISERROR=1
else
    python iGAN.py --data ${DATA} --likelihood_coef ${COEF} --test_name "igan_"${COEF} --GPU_NUM ${GPU_NUM}
fi

if [ $ISERROR = 1 ]; then
    echo "first option : data [adult, news]"
    echo "second option : likelihood coef <float>"
    echo "third option : GPU number to use <int>"
    echo "example : 'sh train_igan_sens_coef.sh adult 0.05 1'"
fi
