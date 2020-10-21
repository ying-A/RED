## Setup the parlai framework
python setup.py develop

## Train Models

bash myrun.sh train <transformer_no_order/s2s_no_order/s2s_att_general_no_order/hred_no_order/recosa_no_order> <personachat/dialog_babi:Task:5> <1>

1. The first argument specifies the type of no_order models to train - transformers, seq2seq with lstms or seq2seq with lstms and attention, hred, recosa. You can change the "no_order" to "1_order,2_order,3_order,full_order" to train the models with order information inserted.
2. The second argument specifies the dataset to train on.
3. The third argument specifies which run to save models to (In our paper we averaged resutls across 3 runs).

## Evaluate and Analyze Models

1. Get the evaluation results (ppl, dist-1 and dist-2) on different test data-sets for different models.  
bash myrun.sh eval_model_on_test <transformer_no_order/...> <personachat/dialog_babi:Task:5> <1>  
2. Get the responses on the sampled test data-sets for different models.  
bash myrun.sh display_model_results <transformer_no_order/...> personachat <1>  
3. Get the evaluation results on the 3 split personachat datasets with different history lengths for different models.  
bash myrun.sh eval_model_dif_his_turns_len <transformer_no_order/...> personachat <1>
4. Get the evaluation results on the personachat data-sets perturbated in different ways for different models.  
bash myrun.sh perturb <transformer_no_order/...> personachat <1>
5. Plot the TSNE embeddings for utterances with different history index.  
python plot_embedding.py





