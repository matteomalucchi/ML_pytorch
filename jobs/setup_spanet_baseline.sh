#micromamba activate ML_pytorch
i=100
run_dir=../out/sig_bkg_classifier/SPANET_baseline_norm_e5drop75_postEE/
#config_file=configs/bkg_reweighting/DNN_1e-2_e20drop95_reweighting_SPANET_5e.yml
config_file=../configs/ggF_bkg_classifier/DNN_spanet_baseline_e5drop75_postEE.yml
ml_train -o "$run_dir" --eval --onnx --roc --histos --history --gpus 0 -n 4 -c $config_file -s $i
#export 'PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3,max_split_size_mb:512'
