ISERROR=0
DATA=$1
COEF=$2
GPU_NUM=$3

if [ -z $GPU_NUM ]; then
    ISERROR=1
elif [ $DATA = "adult" ] && [ $COEF = "0" ]; then
    python iGAN.py --data adult --likelihood_coef 0 --test_name "igan" --GPU_NUM ${GPU_NUM}
elif [ $DATA = "adult" ] && [ $COEF = "L" ]; then
    python iGAN.py --data adult --likelihood_coef -0.014 --test_name "igan_L" --GPU_NUM ${GPU_NUM}
elif [ $DATA = "adult" ] && [ $COEF = "Q" ]; then
    python iGAN.py --data adult --likelihood_coef 0.05 --test_name "igan_Q" --GPU_NUM ${GPU_NUM}
elif [ $DATA = "news" ] && [ $COEF = "0" ]; then
    python iGAN.py --data news --likelihood_coef 0 --test_name "igan" --GPU_NUM ${GPU_NUM}
elif [ $DATA = "news" ] && [ $COEF = "L" ]; then
    python iGAN.py --data news --likelihood_coef -0.01 --test_name "igan_L" --GPU_NUM ${GPU_NUM}
elif [ $DATA = "news" ] && [ $COEF = "Q" ]; then
    python iGAN.py --data news --likelihood_coef 0.05 --test_name "igan_Q" --GPU_NUM ${GPU_NUM}
else
    ISERROR=1
fi

if [ $ISERROR = 1 ]; then
    echo "first option : data [adult, news]"
    echo "second option : likelihood coef [L(Increase NLL), Q(Decrease NLL), 0(Only GAN)]"
    echo "third option : GPU number to use <int>"
    echo "example : 'sh train_igan_main.sh adult L 1'"
fi
