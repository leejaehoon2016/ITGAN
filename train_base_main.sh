ISERROR=0
MODEL=$1
DATA=$2
GPU_NUM=$3
if [ -z $DATA ] || [ -z $MODEL ]; then
    ISERROR=1
elif [ $DATA != "adult" ] && [ $DATA != "news" ]; then
    ISERROR=1
elif [ $MODEL = "identity" ] || [ $MODEL = "privbn" ] || [ $MODEL = "clbn" ] || [ $MODEL = "independent" ] || [ $MODEL = "uniform" ]; then
    python ${MODEL}.py --data ${DATA}
elif [ $MODEL = "ctgan" ] || [ $MODEL = "medgan" ] || [ $MODEL = "tablegan" ] || [ $MODEL = "tvae" ] || [ $MODEL = "veegan" ] ; then
    if [ -z $GPU_NUM ]; then
        ISERROR=1
    else
        python ${MODEL}.py --data ${DATA} --GPU_NUM ${GPU_NUM}
    fi
else
    ISERROR=1
fi

if [ $ISERROR = 1 ]; then
    echo "first option : model [identity, privbn, clbn, independent, uniform, ctgan, medgan, tablegan, tvae, veegan]"
    echo "second option : data [adult, news]"
    echo "third option : GPU number to use <int>"
    echo "example : 'sh train_base_main.sh veegan adult 1'"
fi
