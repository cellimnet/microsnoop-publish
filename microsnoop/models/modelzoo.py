"""
规定训练、eval的具体实施
"""
import datetime
import h5py
import os, time
import numpy as np
import torch.distributed as dist
import torch.utils.data
from scellseg.transforms import make_tiles
from microsnoop.datasets.datasets_utils import pad_image, normalize_img
from microsnoop.models.net_desc import *
from microsnoop.models.core import CoreModel
from microsnoop.models.models_utils import get_rank


class Microsnoop(CoreModel):
    """
    定义Microsnoop，包含train和eval流程，后续子类可以简单重载__init__即可
    """
    def __init__(self):
        super().__init__()

    def embed(self, local_rank, X, y, chan, args, model_type='cnn', tile=False, tile_overlap=0.1, tile_size=224, tile_rate=1.0, normalize=True):
        start_time = time.time()
        rank = args.node_rank * args.ngpu + local_rank
        dist.init_process_group(args.dist_backend,
                                init_method='env://',
                                rank=rank,
                                world_size=args.world_size)
        torch.cuda.set_device(local_rank)

        seed = args.seed + rank  # 设置种子主要是为了限制random_masking那里
        torch.manual_seed(seed)
        np.random.seed(seed)

        # print param
        global_rank = get_rank()
        nimg = len(X)
        nchan = X[0].shape[0]
        if global_rank == args.local_rank:
            print(f'>>>> Input images with {nchan} channel, Batch_size: {args.batch_size}, Number of images: {nimg}')

        embed_fn = None
        if model_type == 'cnn':
            def embed_fn(latent):
                embedding = latent[-1]
                embedding = F.avg_pool2d(embedding, kernel_size=(embedding.shape[-2], embedding.shape[-1]))
                flatten = nn.Flatten()
                embedding = flatten(embedding)
                embedding = embedding / torch.sum(embedding ** 2, axis=1, keepdim=True) ** .5
                return embedding

        # set distributed model
        self.net.cuda(local_rank)
        find_unused_parameters=False
        if 'uformer' in args.model_name: find_unused_parameters=True
        self.net = nn.parallel.DistributedDataParallel(self.net, device_ids=[local_rank], find_unused_parameters=find_unused_parameters)
        # print('Rank state_dict()', local_rank, self.net.module.state_dict()['encoder.cls_token'])

        if not tile:
            embeddings, ys, chans = self._embed_imgs(X, y, chan, args, embed_fn=embed_fn, tile=tile)
            if global_rank == args.local_rank:
                embeddings, ys, chans = list(embeddings), list(ys), list(chans)
        else:
            embeddings, ys, chans = [], [], []
            for i in range(len(X)):
                Xi, ysub, xsub, Ly, Lx = make_tiles(X[i], bsize=tile_size, augment=False, tile_overlap=tile_overlap)
                ny, nx, nchan, ly, lx = Xi.shape
                Xi = np.reshape(Xi, (ny * nx, nchan, ly, lx))
                if args.input_size != tile_size: Xi = pad_image(Xi, xy=[args.input_size, args.input_size])  # Note 1: resize到网络输入的大小
                Xi = np.array(Xi)[np.random.choice(range(len(Xi)), math.ceil(tile_rate * len(Xi)), replace=False)]
                if normalize:
                    Xi = np.array([normalize_img(Xi[j], axis=0) for j in range(len(Xi))])  # Note 2: 放在resize之后做归一化
                yi = np.expand_dims(y[i], 0).repeat(Xi.shape[0], axis=0)
                chani = np.expand_dims(chan[i], 0).repeat(Xi.shape[0], axis=0)
                embeddingsi, ysi, chansi = self._embed_imgs(Xi, yi, chani, args, embed_fn=embed_fn, tile=tile)
                if global_rank == args.local_rank:
                    embeddings.append(embeddingsi)
                    ys.append(ysi)
                    chans.append(chansi)

        if global_rank == args.local_rank:
            if not os.path.isdir(args.embed_dir): os.makedirs(args.embed_dir)
            file_path = os.path.join(args.embed_dir, args.name_meta+'.h5')
            f_embedding = h5py.File(file_path, 'w')
            # f_embedding = h5c.File(file_path, 'w', chunk_cache_mem_size=1024**3*10)
            f_embedding['embeddings'] = embeddings
            f_embedding['inds'] = ys
            f_embedding['chans'] = chans
            f_embedding.close()

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('\033[1;32mEmbedding Time: {} \033[0m'.format(total_time_str))


class MicrosnoopCNN(Microsnoop):
    def __init__(self, model_init_param, checkpoint=None, input_size=224, patch_size=16):
        super().__init__()
        # self.input_size = model_init_param['input_size']
        # self.patch_size = model_init_param['patch_size']
        self.input_size = input_size
        self.patch_size = patch_size
        self.net = CNNNet(**model_init_param)
        if checkpoint is not None:
            self.load_model(checkpoint=checkpoint)
