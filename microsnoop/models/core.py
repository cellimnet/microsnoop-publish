"""
Define C:CoreModel, including f:_embed_imgs and f:load_model
"""

import os
import sys
import numpy as np
import cv2 as cv
import pandas as pd
from microsnoop.models.models_utils import *
import time


try:
    import torch
    from torch import optim, nn
    from net_desc import *

    TORCH_ENABLED = True
    torch_GPU = torch.device('cuda')
    torch_CPU = torch.device('cpu')
except:
    TORCH_ENABLED = False


class CoreModel():
    def __init__(self):
        rank = get_rank()
        if rank==0:
            print(">>>> Model Init Start")

    def _embed_imgs(self, X, y, chan, args, embed_fn, tile=False):
        # set data sample and loader
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X * 1.0).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if isinstance(chan, np.ndarray):
            chan = torch.from_numpy(chan)
        data_sampler = torch.utils.data.DistributedSampler(
            X, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
        code = torch.range(0, len(X)-1, dtype=torch.int32)  # Note: 是为了查找哪些数据重复了, 方便后面删除
        dataset_pair = torch.utils.data.TensorDataset(X, y, chan, code)
        data_loader = torch.utils.data.DataLoader(dataset=dataset_pair, drop_last=False,
            sampler=data_sampler, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.pin_mem)

        global_rank = get_rank()

        # for each batch
        embeddings, ys, chans, codes = None, None, None, None
        self.net.eval()
        start_time = time.time()
        for ibatch, (Xi, yi, chani, codei) in enumerate(data_loader):
            # print('ibatch， len(Xi), rank', ibatch, len(Xi), get_rank())

            # run net
            Xi = Xi.cuda(non_blocking=True)
            latent = self.net.module.encoder(Xi, mask_ratio=0)[0]  # transformor网络拿的是最后一层中的cls_token
            embedding = embed_fn(latent)

            gathered_embedding = [torch.empty_like(embedding) for _ in range(args.world_size)]
            yi = yi.cuda(non_blocking=True)
            chani = chani.cuda(non_blocking=True)
            codei = codei.cuda(non_blocking=True)
            gathered_yi = [torch.empty_like(yi) for _ in range(args.world_size)]
            gathered_chani = [torch.empty_like(chani) for _ in range(args.world_size)]
            gathered_codei = [torch.empty_like(codei) for _ in range(args.world_size)]
            dist.barrier()  # tile模式下 先把多个GPU上的结果汇总起来，不然同一张大图上的数据就会被分成四份
            dist.all_gather(gathered_embedding, embedding.contiguous())  # contiguous()
            dist.all_gather(gathered_yi, yi.contiguous())  # contiguous()
            dist.all_gather(gathered_chani, chani.contiguous())  # contiguous()
            dist.all_gather(gathered_codei, codei.contiguous())  # contiguous()
            embedding = [embeddingi.cpu().detach().numpy() for gathered_embeddingi in gathered_embedding for embeddingi in gathered_embeddingi]
            yi = [yij.cpu().detach().numpy() for gathered_yij in gathered_yi for yij in gathered_yij]
            chani = [chanij.cpu().detach().numpy() for gathered_chanij in gathered_chani for chanij in gathered_chanij]
            codei = [codeij.cpu().detach().numpy() for gathered_codeij in gathered_codei for codeij in gathered_codeij]

            if global_rank == args.local_rank:
                if embeddings is None:
                    embeddings = np.array(embedding)
                else:
                    embeddings = np.concatenate((embeddings, np.array(embedding)), axis=0)
                if ys is None:
                    ys = yi
                    chans = chani
                    codes = codei
                else:
                    ys = np.concatenate((ys, yi), axis=0)
                    chans = np.concatenate((chans, chani), axis=0)
                    codes = np.concatenate((codes, codei), axis=0)
        torch.cuda.synchronize()  # 时间相关
        if global_rank == args.local_rank:
            if not tile: print('>>>> step extract time', time.time() - start_time)
            codes = [str(codeii) for codeii in codes]
            code_df = pd.DataFrame({'embeddings':list(embeddings), 'ys':list(ys), 'chans':list(chans), 'codes':codes})
            code_df_drop = code_df.groupby(by=['codes']).sample(n=1).sort_values('codes',ascending=True)
            embeddings = np.array(code_df_drop.embeddings.to_list())
            ys = code_df_drop.ys.to_numpy()
            chans = code_df_drop.chans.to_numpy()
            if not tile: print('>>>> step aggregate time', time.time() - start_time)
            if tile:
                embeddings = embeddings.mean(axis=0)  # Note 同一张图上的embedding聚合方式，这里就简单的求了下平均
                ys = ys[0]  # tile模式下默认一样的
                chans= chans[0]  # tile模式下默认一样的

            # print('len(embeddings)&len(ys)', len(ys), len(embeddings))
            return embeddings, ys, chans
        else:
            return None, None, None

    def load_model(self, checkpoint, resume=False):
        checkpoint_name = None
        if isinstance(checkpoint, str):
            checkpoint_name = checkpoint
            checkpoint = torch.load(checkpoint, map_location='cpu')
        assert 'model' in checkpoint, 'Please provide correct checkpoint'
        model_dict = checkpoint['model']
        try:
            self.net.load_state_dict(model_dict)
        except:
            model_dict = {k.replace('module.', ''): v for k, v in model_dict.items() if k.startswith('module.')}
            self.net.load_state_dict(model_dict)

        if resume:
            if 'optimizer' in checkpoint and 'checkpoint_epoch' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if checkpoint['checkpoint_epoch'] is not None:
                    self.start_epoch = checkpoint['checkpoint_epoch'] + 1
                    print(">>>> Resume checkpoint %s with optimizer" % str(self.start_epoch))
                if 'scaler' in checkpoint:
                    self.loss_scaler.load_state_dict(checkpoint['scaler'])
                    print(">>>> Resume checkpoint also with loss_scaler!")
        else:
            print(">>>> Successfullly load pre-trained checkpoint", checkpoint_name)
