#micromamba activate ML_pytorch
i=100
run_dir=../out/sig_bkg_classifier/DHH_method_norm_e5drop75_postEE_allklambda/
config_file=../configs/ggF_bkg_classifier/DNN_DHH_method_e5drop75_postEE_allklambda.yml
#config_file=/work/tharte/datasets/ML_pytorch/configs/bkg_reweighting/DNN_AN_1e-3_e20drop75_minDelta1em5_run2_postEE.yml
ml_train -o "$run_dir" --eval --onnx --roc --histos --history --gpus 0 -n 4 -c $config_file -s $i
#export 'PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3,max_split_size_mb:512'
