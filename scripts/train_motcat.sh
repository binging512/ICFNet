CUDA_VISIBLE_DEVICES=3 python main.py \
--data_root_dir data/BLCA/patch_feats/clam_inr50t_s20 \
--split_dir tcga_blca \
--model_type motcat \
--bs_micro 16384 \
--ot_impl pot-uot-l2 \
--ot_reg 0.1 --ot_tau 0.5 \
--which_splits 5foldcv \
--apply_sig