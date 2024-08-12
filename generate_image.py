"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import shutil
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm

from core.data_loader import get_eval_loader
from core import utils
from metrics.eval import calculate_fid_for_all_tasks
from metrics.lpips import calculate_lpips_given_images

@torch.no_grad()
def generate_image(nets, args, step, mode):
    print('Calculating evaluation metrics...')
    assert mode in ['latent', 'reference']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    domains = os.listdir(args.val_img_dir)
    domains.sort()
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)
    lpips_dict = OrderedDict()

    for trg_idx, trg_domain in enumerate(domains):
        src_domains = [x for x in domains if x != trg_domain]

        if mode == 'reference':
            path_ref = os.path.join(args.val_img_dir, trg_domain)
            loader_ref = get_eval_loader(root=path_ref,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size,
                                         imagenet_normalize=False,
                                         drop_last=True)

        for src_idx, src_domain in enumerate(src_domains):
            path_src = os.path.join(args.val_img_dir, src_domain)
            loader_src = get_eval_loader(root=path_src,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size,
                                         imagenet_normalize=False)
            if src_domain not in "BCL_mask":
                continue
            task = '%s2%s' % (src_domain, trg_domain)
            path_root = os.path.join(args.eval_dir, str(step), task)
            path_fake = os.path.join(path_root, mode)
            path_origin = os.path.join(path_root, "origin")
            path_recon = os.path.join(path_root, "recon")
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)
            os.makedirs(path_origin)
            os.makedirs(path_recon)

            lpips_values = []

            print('Generating images and calculating LPIPS for %s...' % task)
            for i, x_src in enumerate(tqdm(loader_src, total=len(loader_src))):
                N = x_src.size(0)
                x_src = x_src.to(device)
                y_trg = torch.tensor([trg_idx] * N).to(device)
                masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

                # generate 10 outputs from the same input
                group_of_images = []
                for j in range(args.num_outs_per_domain):
                    if mode == 'latent':
                        z_trg = torch.randn(N, args.latent_dim).to(device)
                        s_trg = nets.mapping_network(z_trg, y_trg)
                    else:
                        try:
                            x_ref = next(iter_ref).to(device)
                        except:
                            iter_ref = iter(loader_ref)
                            x_ref = next(iter_ref).to(device)

                        if x_ref.size(0) > N:
                            x_ref = x_ref[:N]
                        s_trg = nets.style_encoder(x_ref, y_trg)

                    x_fake = nets.generator(x_src, s_trg, masks=masks)
                    group_of_images.append(x_fake)

                    # reconstruct image
                    y_src = torch.tensor([1] * N).to(device) # results 폴더가 3임
                    s_src = nets.style_encoder(x_src, y_src)
                    x_rec = nets.generator(x_fake, s_src)

                    # save generated images to calculate FID later
                    for k in range(N):
                        file_num = i*args.val_batch_size+(k+1)
                        filename = f'{file_num:04}_{j+1:02}.png'
                        fake_file_path = os.path.join(path_fake, filename)
                        utils.save_image(x_fake[k], ncol=1, filename=fake_file_path)
                        recon_file_path = os.path.join(path_recon, filename)
                        utils.save_image(x_rec[k], ncol=1, filename=recon_file_path)

                for k in range(N):
                    origin_file_num = i*args.val_batch_size+(k+1)
                    origin_file_name = f'{origin_file_num:04}.png'
                    filename = os.path.join(path_origin, origin_file_name)
                    utils.save_image(x_src[k], ncol=1, filename=filename)

                lpips_value = calculate_lpips_given_images(group_of_images)
                lpips_values.append(lpips_value)

            # calculate LPIPS for each task (e.g. cat2dog, dog2cat)
            lpips_mean = np.array(lpips_values).mean()
            lpips_dict[f'LPIPS_{mode}/{task}'] = lpips_mean

        # delete dataloaders
        del loader_src
        if mode == 'reference':
            del loader_ref
            del iter_ref

    # calculate the average LPIPS for all tasks
    lpips_mean = 0
    for _, value in lpips_dict.items():
        lpips_mean += value / len(lpips_dict)
    lpips_dict[f'LPIPS_{mode}/mean'] = lpips_mean

    # report LPIPS values
    filename = os.path.join(args.eval_dir, f'LPIPS_{step:5}_{mode}.json')
    utils.save_json(lpips_dict, filename)

    # calculate and report fid values
    calculate_fid_for_all_tasks(args, domains, step=step, mode=mode)
