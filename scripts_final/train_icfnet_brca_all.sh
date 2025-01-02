# dense+bnll+unimodule
python main.py \
--data_root_dir data/BRCA/patch_feats/clam_inr50t_s20 \
--split_dir tcga_brca \
--model_type icfnet \
--bs_micro 16384 \
--ot_impl pot-uot-l2 \
--ot_reg 0.1 \
--ot_tau 0.5 \
--which_splits 5foldcv \
--apply_sig \
--status_path data/BRCA/status_brca.json \
--mode coattn_text \
--bag_loss dense_balanced_combine_cosine \
--opt adam \
--fusion uni_module \
--results_dir results_final