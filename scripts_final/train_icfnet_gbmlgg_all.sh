# dense+bnll+unimodule
python main.py \
--data_root_dir data/GBMLGG/patch_feats/clam_inr50t_s20 \
--split_dir tcga_gbmlgg \
--model_type icfnet \
--bs_micro 16384 \
--ot_impl pot-uot-l2 \
--ot_reg 0.1 \
--ot_tau 0.5 \
--which_splits 5foldcv \
--apply_sig \
--status_path data/GBMLGG/status_gbmlgg.json \
--mode coattn_text \
--bag_loss dense_balanced_combine_cosine \
--opt adam \
--fusion uni_module \
--max_epochs 30 \
--results_dir results_final