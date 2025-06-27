#micromamba activate ML_pytorch
i=100
run_dir=../out/sig_bkg_classifier/SPANET_ptflat_norm_e5drop75_postEE_ratioAll_DeltaProb/
config_file=../configs/bkg_reweighting/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_btag_ratioAll_postEE_DeltaProb.yml
#config_file=../configs/ggF_bkg_classifier/DNN_spanet_ptflat_e5drop75_postEE_allklambda.yml
ml_train -o "$run_dir" --eval --onnx --roc --histos --history --gpus 0 -n 4 -c $config_file -s $i
#export 'PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3,max_split_size_mb:512'
