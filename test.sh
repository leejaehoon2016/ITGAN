ISERROR=0
TYPE=$1
DATA=$2
GPU_NUM=$3

if [ -z $GPU_NUM ]; then
    ISERROR=1
elif [ $DATA != "adult" ] && [ $DATA != "news" ]; then
    ISERROR=1
elif [ $TYPE = "score" ]; then
    python test_check_score.py --data ${DATA}
elif [ $TYPE = "distplot" ]; then
    python test_draw_distance_plot.py --data ${DATA} --GPU_NUM ${GPU_NUM}
elif [ $TYPE = "fbb" ]; then
    python test_fbb.py --data ${DATA} --GPU_NUM ${GPU_NUM}
else
    ISERROR=1
fi

if [ $ISERROR = 1 ]; then
    echo "first option : test type  [score(check model score), distplot(draw real-fake distance distribution), fbb(fbb attack score)]"
    echo "second option : data [adult, news]"
    echo "third option : GPU number to use <int>"
    echo "example : 'sh train.sh score adult 0'"
fi
