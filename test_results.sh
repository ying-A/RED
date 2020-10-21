#!/bin/sh
models=(transformer_no_order transformer_1_order transformer_2_order transformer_3_order transformer_full_order
        hred_no_order hred_1_order hred_2_order hred_3_order hred_full_order
        recosa_no_order recosa_1_order recosa_2_order recosa_3_order recosa_full_order
        s2s_no_order s2s_1_order s2s_2_order s2s_3_order s2s_full_order
        s2s_att_general_no_order s2s_att_general_1_order s2s_att_general_2_order s2s_att_general_3_order s2s_att_general_full_order)

for  i in {0..24}
do
    bash myrun.sh eval_model_on_test ${models[i]} personachat 1
    bash myrun.sh display_model_results ${models[i]} personachat 1
    bash myrun.sh eval_model_dif_his_turns_len ${models[i]} personachat 1
    bash myrun.sh perturb ${models[i]} personachat 1
done
