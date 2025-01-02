import os
import glob
import pandas as pd
import shutil
import numpy as np
import matplotlib.pyplot as plt
import openslide
import random
import sys
from create_patches_fp import create_patches_fp

def recollect_slides(fast_data_dir, ori_data_dir, anno_df, output_dir):
    slide_names = anno_df.loc[:,'slide_id'].to_list()
    for ii, slide_name in enumerate(sorted(slide_names)):
        print("Copying {}/{} slides...".format(ii, len(slide_names)),end='\r')
        slide_path = os.path.join(fast_data_dir, slide_name)
        if os.path.isfile(slide_path):
            pass
        else:
            slide_path = glob.glob(os.path.join(ori_data_dir,'**',slide_name),recursive=True)[0]
        output_path = os.path.join(output_dir, 'slides', slide_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy(slide_path, output_path)

def gen_patches(mag_list, dst_patch_size, stride_size, output_dir):
    slide_names = sorted(os.listdir(os.path.join(output_dir,'slides')))
    for ii, slide_name in enumerate(slide_names):
        print("Generating the {}/{} slide patches...".format(ii, len(slide_names)))
        if os.path.isfile(os.path.join('data/UCEC/patches/{}x/patches/{}'.format(mag_list[-1],slide_name.replace('.svs','.h5')))):
            continue
        slide_path = os.path.join(output_dir,'slides', slide_name)
        img = openslide.open_slide(slide_path)
        ori_W, ori_H = img.dimensions
        if 'aperio.AppMag' not in img.properties.keys():
            max_size = max(int(img.properties['openslide.level[0].height']),int(img.properties['openslide.level[0].width']))
            if max_size//10000 >=2 and max_size//10000 <10:
                max_mag = 20
        else:
            max_mag = int(float(img.properties['aperio.AppMag']))
    
        # Get needed wsi-level and downsample list
        patch_size_list = [int(dst_patch_size*(max_mag/mag)) for mag in mag_list]
        
        # Slide crop
        for i in range(len(mag_list)):
            patch_size = patch_size_list[i]
            patch_dir_name = "{}x".format(mag_list[i])
            if os.path.isfile(os.path.join('data/UCEC/patches/{}/patches/{}'.format(patch_dir_name,slide_name.replace('.svs','.h5')))):
                continue
            create_patches_fp(source = 'data/UCEC/slides/{}'.format(slide_name),
                              step_size = patch_size,
                              patch_size = patch_size, 
                              patch = True, seg = True, stitch = True, no_auto_skip = False,
                              save_dir = 'data/UCEC/patches/{}/'.format(patch_dir_name), 
                              preset = './data_process/preprocess/tcga.csv',
                              patch_level = 0, process_list=None)
    pass


def pre_UCEC():
    fast_data_dir = './data/UCEC/slides'
    ori_data_dir = '/raid/share_zby/mmmil_dataset/origin/UCEC/slides'
    
    anno_path = 'data/UCEC/tcga_ucec_all_clean.csv.zip'
    output_dir = 'data/UCEC/'
    
    os.makedirs(output_dir,exist_ok=True)
    anno_df = pd.read_csv(anno_path)
    # Drop Missing slides
    missing_slides_ls = ['TCGA-A7-A6VX-01Z-00-DX2.9EE94B59-6A2C-4507-AA4F-DC6402F2B74F.svs',
                            'TCGA-A7-A0CD-01Z-00-DX2.609CED8D-5947-4753-A75B-73A8343B47EC.svs',
                            'TCGA-HT-7483-01Z-00-DX1.7241DF0C-1881-4366-8DD9-11BF8BDD6FBF.svs',
                            'TCGA-06-0882-01Z-00-DX2.7ad706e3-002e-4e29-88a9-18953ba422bf.svs']
    anno_df.drop(anno_df[anno_df['slide_id'].isin(missing_slides_ls)].index, inplace=True)
    anno_df.to_csv(os.path.join(output_dir,'tcga_ucec_all_clean.csv'))
    
    # Recollect the slides
    recollect_slides(fast_data_dir, ori_data_dir, anno_df, output_dir)
    
    # Generate patches
    mag_list = [20]
    patch_size = 256
    stride_size = 256
    print("Generating the patches...")
    gen_patches(mag_list, patch_size, stride_size, output_dir)
    print("Generating the patches complete!")
    
    
if __name__=="__main__":
    pre_UCEC()