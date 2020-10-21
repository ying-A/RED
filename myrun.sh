#!/bin/sh

if [ -z "$4" ]
then
        echo "No Run id provided."
        exit 0
else
        RUN_ID=$4

fi

echo "RUN : "$RUN_ID    
LOGDIR="perturb_log_files_"$RUN_ID"/"
SAVEDIR="save_dir_run_"$RUN_ID"/"
RESULTSDIR="results_"$RUN_ID"/"
DIF_TURN_LEN_RESULTS_DIR="different_his_len_results_"$RUN_ID"/"


GPU="$5"
mkdir -p $LOGDIR
mkdir -p $RESULTSDIR
mkdir -p $DIF_TURN_LEN_RESULTS_DIR

if [ -z "$1" ]
then
    echo "No Run mode provided."

else
    RUN_MODE=$1
fi

GPU_ARGS=" --gpu "$GPU


############################################### s2s ####################################################
if [ "$2" = "s2s_no_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m seq2seq -ord no"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"
elif [ "$2" = "s2s_1_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m seq2seq -ord 1_order"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"
elif [ "$2" = "s2s_2_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m seq2seq -ord 2_order"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"

elif [ "$2" = "s2s_3_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m seq2seq -ord 3_order"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"

elif [ "$2" = "s2s_full_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m seq2seq -ord full"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"




############################################### s2s_att ####################################################
elif [ "$2" = "s2s_att_general_no_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m seq2seq -att general -ord no"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"
elif [ "$2" = "s2s_att_general_1_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m seq2seq -att general -ord 1_order"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"
elif [ "$2" = "s2s_att_general_2_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m seq2seq -att general -ord 2_order"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"
elif [ "$2" = "s2s_att_general_3_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m seq2seq -att general -ord 3_order"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"
elif [ "$2" = "s2s_att_general_full_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m seq2seq -att general -ord full"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"


############################################### transformer ####################################################



elif [ "$2" = "transformer_no_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m transformer/generator -ord no"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000" 
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -bs 64 --optimizer adam -lr 0.001 --lr-scheduler invsqrt --warmup-updates 4000 -eps 25 -veps 1 -stim 200" 
elif [ "$2" = "transformer_1_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m transformer/generator -ord 1_order"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000" 
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -bs 64 --optimizer adam -lr 0.001 --lr-scheduler invsqrt --warmup-updates 4000 -eps 25 -veps 1 -stim 200" 
elif [ "$2" = "transformer_2_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m transformer/generator -ord 2_order"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000" 
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -bs 64 --optimizer adam -lr 0.001 --lr-scheduler invsqrt --warmup-updates 4000 -eps 25 -veps 1 -stim 200" 
elif [ "$2" = "transformer_3_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m transformer/generator -ord 3_order"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000" 
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -bs 64 --optimizer adam -lr 0.001 --lr-scheduler invsqrt --warmup-updates 4000 -eps 25 -veps 1 -stim 200" 
elif [ "$2" = "transformer_full_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m transformer/generator -ord full"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000" 
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -bs 64 --optimizer adam -lr 0.001 --lr-scheduler invsqrt --warmup-updates 4000 -eps 25 -veps 1 -stim 200" 



######################################################### recosa #####################################################
elif [ "$2" = "recosa_no_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m recosa/generator -ord no -rnn_hid 300"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000" 
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -bs 64 --optimizer adam -lr 0.001 --lr-scheduler invsqrt --warmup-updates 4000 -eps 25 -veps 1 -stim 200" 
elif [ "$2" = "recosa_1_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m recosa/generator -ord 1_order -rnn_hid 300"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000" 
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -bs 64 --optimizer adam -lr 0.001 --lr-scheduler invsqrt --warmup-updates 4000 -eps 25 -veps 1 -stim 200" 
elif [ "$2" = "recosa_2_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m recosa/generator -ord 2_order -rnn_hid 300"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000" 
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -bs 64 --optimizer adam -lr 0.001 --lr-scheduler invsqrt --warmup-updates 4000 -eps 25 -veps 1 -stim 200" 
elif [ "$2" = "recosa_3_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m recosa/generator -ord 3_order -rnn_hid 300"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000" 
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -bs 64 --optimizer adam -lr 0.001 --lr-scheduler invsqrt --warmup-updates 4000 -eps 25 -veps 1 -stim 200" 
elif [ "$2" = "recosa_full_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m recosa/generator -ord full -rnn_hid 300"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000" 
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -bs 64 --optimizer adam -lr 0.001 --lr-scheduler invsqrt --warmup-updates 4000 -eps 25 -veps 1 -stim 200" 

############################################### Hred ####################################################
elif [ "$2" = "hred_no_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m hred -ord no"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"
elif [ "$2" = "hred_1_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m hred -ord 1_order"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"
elif [ "$2" = "hred_2_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m hred -ord 2_order"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"

elif [ "$2" = "hred_3_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m hred -ord 3_order"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"

elif [ "$2" = "hred_full_order" ]
then
    echo "MODELTYPE: "$2
    COMMON_ARGS=$GPU_ARGS" -m hred -ord full"
    EVAL_MODEL_ARGS=$COMMON_ARGS" -bs 1 -d False -ne 1000"
    TRAIN_MODEL_ARGS=$COMMON_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000"


else
    echo "INVALID modeltype : "$2" Supported : s2s, s2s_att_general, transformer Order: no,1_order,2_order,3_order,full"
    echo "Example train command : sh run.sh train <model_type> <dataset> <run_id>"
    echo "Example perturb command : sh run.sh perturb <model_type> <dataset>"
    exit 0
fi

if [ -z "$3" ]
then
    echo "No Dataset type specified supplied"
    exit 0
else
    DATASET=$3
    MF=$SAVEDIR"/model_"$3"_"$2
    DF=$SAVEDIR"/model_"$3"_"$2".dict"
fi

if [ $RUN_MODE = "eval_model_on_test" ]
then
    echo "MODE : "$RUN_MODE
    for MODEL_TYPE in $2
    do
        for DATATYPE in "test" #valid
        do
            echo "---------------------"
            echo "CONFIG : "$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_MY_TEST_NoPerturb"
            LOGFILE=$LOGDIR/log_$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_my_test_no_perturb.txt"
            python -W ignore parlai/scripts/my_eval_test.py -t $DATASET -mf $MF -sft True -pb "None" --datatype $DATATYPE > $LOGFILE
            grep FINAL_REPORT $LOGFILE
        done 
    done

elif [ $RUN_MODE = "display_model_results" ]
then
    echo "MODE : "$RUN_MODE
    for MODEL_TYPE in $2
    do
        for DATATYPE in "test" #valid
        do
            echo "---------------------"
            echo "CONFIG : "$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_results"
            LOGFILE=$RESULTSDIR/$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_results.txt"
            python -W ignore parlai/scripts/display_model.py -t $DATASET -mf $MF -bs 1 --datatype $DATATYPE > $LOGFILE
        done 
    done

elif [ $RUN_MODE = "eval_model_dif_his_turns_len" ]
then
    echo "MODE : "$RUN_MODE
    for MODEL_TYPE in $2
    do
        for DATATYPE in "test" #valid
        do
            echo "---------------------"
            echo "CONFIG : "$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_dif_his_len_results"
            LOGFILE=$DIF_TURN_LEN_RESULTS_DIR/$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_dif_his_len_results.txt"
            python -W ignore parlai/scripts/eval_model_on_different_turn_len.py -t $DATASET -mf $MF -df $DF -bs 1 -sft True --datatype $DATATYPE > $LOGFILE
        done 
    done

elif [ $RUN_MODE = "perturb" ]
then
    echo "MODE : "$RUN_MODE
    for MODEL_TYPE in $2
    do
        for DATATYPE in "test" #valid
        do
            echo "---------------------"
            echo "CONFIG : "$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_NoPerturb"
            LOGFILE=$LOGDIR/log_$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_no_perturb_mode_pert.txt"
            python -W ignore examples/eval_model.py $EVAL_MODEL_ARGS -t $DATASET -mf $MF -sft True -pb "None" --datatype $DATATYPE > $LOGFILE
            grep FINAL_REPORT $LOGFILE

            for PERTURB_TYPE in "only_last" "shuffle" "reverse_utr_order" "drop_first" "drop_last"
            do
                echo "---------------------"
                echo "CONFIG : "$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_"$PERTURB_TYPE
                LOGFILE=$LOGDIR/log_$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_"$PERTURB_TYPE"_mode_pert.txt"
                python -W ignore examples/eval_model.py $EVAL_MODEL_ARGS -t $DATASET -mf $MF -sft True -pb $PERTURB_TYPE --datatype $DATATYPE > $LOGFILE
                grep FINAL_REPORT $LOGFILE
            done
        done 
    done

elif [ $RUN_MODE = "train" ]
then
    echo $TRAIN_MODEL_ARGS
    python examples/train_model.py -t $DATASET -mf $MF $TRAIN_MODEL_ARGS
else
    echo "Invalid Run mode provided."
    exit 0
fi