## BLCA
python main.py \
--data_root_dir data/BLCA/patch_feats/clam_inr50t_s20 \
--split_dir tcga_blca \
--model_type mcat \
--which_splits 5foldcv \
--apply_sig


## BRCA
python main.py \
--data_root_dir data/BRCA/patch_feats/clam_inr50t_s20 \
--split_dir tcga_brca \
--model_type mcat \
--which_splits 5foldcv \
--apply_sig


## UCEC
python main.py \
--data_root_dir /path/to/UCEC/x20 \
--split_dir tcga_ucec \
--model_type mcat \
--which_splits 5foldcv \
--apply_sig


## GBMLGG
python main.py \
--data_root_dir /path/to/GBMLGG/x20 \
--split_dir tcga_gbmlgg \
--model_type mcat \
--which_splits 5foldcv \
--apply_sig


## LUAD
python main.py \
--data_root_dir /path/to/LUAD/x20 \
--split_dir tcga_luad \
--model_type mcat \
--which_splits 5foldcv \
--apply_sig
