import datetime
from sklearn import preprocessing
import numpy as np
import pandas as pd
import scipy.stats as stats
import faiss
import os, sys
import h5py
from tqdm import trange
project_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.path.sep + ".")
sys.path.append(project_path)  # 从命令行运行需要添加这个
from microsnoop.models.modelzoo import *
from microsnoop.datasets.datasets_utils import resize_image, square_image, normalize_img
from microsnoop.models.models_utils import reshape_data, pop_nan_data
from microsnoop.postprocess.aggregate import aggregate_sc_data
import torch.multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import BorderlineSMOTE

# faiss kNN Implementation
class FaissKNeighbors:
    """Efficient kNN Implementation using faiss library, following scikit-learn
    conventions.

    Modified from a TowardsDataScience article.
    Link: https://towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb

    Cosine similarity code modified from GitHub Issue.
    Link: https://github.com/facebookresearch/faiss/issues/1119#issuecomment-596514782
    """

    def __init__(self, k=5, metric='euclidean'):
        self.metric = metric
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        X_copy = X.copy()
        if self.metric == 'euclidean':
            self.index = faiss.IndexFlatL2(X.shape[1])
        else:  # cosine distance
            quantizer = faiss.IndexFlatIP(X.shape[1])
            self.index = faiss.IndexIVFFlat(quantizer, X.shape[1],
                                            int(np.sqrt(X.shape[0])),
                                            faiss.METRIC_INNER_PRODUCT)
            faiss.normalize_L2(X_copy.astype(np.float32))
            self.index.train(X_copy.astype(np.float32))
        self.index.add(X_copy.astype(np.float32))
        self.y = y

    def predict(self, X):
        # Create deep copy
        X_copy = X.copy()

        if self.metric == 'cosine':
            # L2 Normalize
            faiss.normalize_L2(X_copy.astype(np.float32))

        distances, indices = self.index.search(X_copy.astype(np.float32),
                                               k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

    def kneighbors(self, X, n_neighbors: int):
        if isinstance(X, list):
            X_copy = np.array(X)
        else:
            # Create deep copy
            X_copy = X.copy()

        if self.metric == 'cosine':
            # L2 Normalize
            faiss.normalize_L2(X_copy.astype(np.float32))

        dist, ind = self.index.search(X_copy.astype(np.float32),
                                      k=n_neighbors)

        return dist, ind

class EvalProcedure:
    """Abstract ValidationProcedure Class. Meant to serve as a base for subclasses for evaluation datasets.

    Generally, the same procedure is taken for all evaluation sets:
        1. Create Image Generators for each channel.  # 生成图像数据流
        2. Extract embeddings based on specific preprocessing method.  # 基于特殊的预处理方法提取特征
        3. Classify using k-Nearest Neighbors.  # 使用KNN进行分类
        4. Save results.
    """

    def __init__(self):
        print('>>>> Dataset Eval Init')

    def load_data(self, dataset_dir, gen_size=None):
        raise NotImplementedError()

    def set_model(self, model_type='vit', args=None):
        # set model
        model=None
        print('>>>> Use model:', model_type)
        if 'cnn' in model_type.lower():
            model_init_param = {'in_chans': args.in_chans, 'out_chans': 1,
                                'depths': [args.embed_dim // 8, args.embed_dim // 4, args.embed_dim // 2, args.embed_dim]}
            model = MicrosnoopCNN(model_init_param, input_size=args.input_size, patch_size=args.patch_size,
                                  checkpoint=args.checkpoint)
        return model


    def extract_embeddings(self, dataset_name, data_loader, checkpoint_path, args, model_type='vit', rsc_to_diam=1.0, rescale_to_input=True,
                           tile=False, tile_overlap=0.1, tile_size=224, tile_rate=1.0, normalize=True):
        """
        用于生成评估数据集的embedding，把embedding文件单独保存出来以供后续分析使用
        model_type: cnn
        assert channel first
        """
        # set distributed mode
        args.ngpu = torch.cuda.device_count()
        args.world_size = args.ngpu * args.nnode
        print(">>>> Totally GPU Devices: ", str(args.world_size))

        # set params
        args.checkpoint = checkpoint_path
        print(f'>>>> Use model file: {args.checkpoint}')
        if not os.path.isdir(args.embed_dir): os.makedirs(args.embed_dir)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12345'

        model = self.set_model(model_type=model_type, args=args)

        for i, (imgs, inds, name_meta) in enumerate(data_loader):
            # print('['+str(i+1)+']:', data_root, len(data))
            args.name_meta = name_meta
            if imgs[0].ndim == 2:
                imgs = [imgi[np.newaxis, :, :] for imgi in imgs]  # 灰度图的话，在前面添加一维
            X = np.array(imgs)
            n, c, h, w = X.shape  # Note: 前面要统一好，这里拿到的图片
            X = X.reshape(n * c // args.in_chans, args.in_chans, h, w)  # 对每个通道进行处理，reshape之后，同一张图片的几个通道是靠在一起的
            y = np.expand_dims(inds, 0).repeat(c, axis=1)
            chan = np.zeros_like(y)
            for ci in range(c): chan[:, ci::c] = ci  # 保存通道信息，方便后面恢复
            y = y.reshape(n * c, )
            chan = chan.reshape(n * c, )
            y, chan = y[::args.in_chans], chan[::args.in_chans]

            # resize before input
            if rescale_to_input:
                X = square_image(X)
                X = resize_image(X, xy=(args.input_size, args.input_size))
            else:
                X = resize_image(X, rsc=rsc_to_diam)  # Note: 根据diam计算的rsc_to_diam来resize
                _, flag_y, flag_x = X[0].shape
                flag_y_max = np.array([X[i].shape[1] for i in range(len(X))]).max()
                flag_x_max = np.array([X[i].shape[2] for i in range(len(X))]).max()
                if flag_y_max > args.input_size or flag_x_max > args.input_size: tile=True  # Note: 如果resize之后还是比网络输入大, 需要使用tile模式。

            if not tile:
                X = pad_image(X, xy=[args.input_size, args.input_size])  # Note 2: pad到网络输入的大小
                if normalize:
                    X = [normalize_img(X[i], axis=0) for i in range(len(X))]  # Note 3: 放在resize之后做归一化
            else:
                print('>>>> Using tile mode')
                X = [X[i] for i in range(len(X))]  # 不加这一行会报错（OverflowError: cannot serialize a bytes object larger than 4 GiB）
            mp.spawn(model.embed,
                     args=(X, y, chan, args, model_type, tile, tile_overlap, tile_size, tile_rate, normalize),
                     nprocs=args.world_size,
                     join=True)  # 第一个参数相当于是由mp提供的. 只有一个GPU跑不通这个脚本

    def load_embeddings(self, embed_dir, filter_str=None):
        """
        :param filter_str: 必须包含的字符
        """
        embeddings, inds, chans = [], [], []
        use_embed_paths = []
        embed_paths = os.listdir(embed_dir)
        embed_paths = [embed_path for embed_path in embed_paths if '.h5' in embed_path]
        embed_paths.sort()
        for embed_path in embed_paths:
            if filter_str is not None:
                if filter_str not in embed_path:
                    continue
            use_embed_paths.append(embed_path)
            embeddings_file = h5py.File(os.path.join(embed_dir, embed_path))
            embeddings.extend(embeddings_file['embeddings'][:])
            inds.extend(embeddings_file['inds'][:])
            chans.extend(embeddings_file['chans'][:])

        # 对embeddings进行排序，首先把同一张图片排到一起，然后根据通道排序
        df = pd.DataFrame({'embeddings': embeddings, 'inds': inds, 'chans': chans})
        df = df.sort_values(by=['inds', 'chans'])
        embeddings = df['embeddings'].tolist()
        inds = df['inds'].tolist()
        print('>>>> Load embeddings from:', use_embed_paths)
        return embeddings, inds

    def prepare_embeddings(self, embed_dir, reshape_chans, seg=False, reshape_mode='concat', filter_str=None, is_train=False, instmap_path=None):
        embeddings, inds = self.load_embeddings(embed_dir, filter_str=filter_str)

        embeddings, ys = reshape_data(embeddings, c=reshape_chans, mode=reshape_mode, kys=[inds])
        inds = ys[0]
        n_total = len(embeddings)
        embeddings, ys = pop_nan_data(embeddings, kys=[inds])
        inds = ys[0]

        tip_str = 'Eval'
        if is_train: tip_str = 'Train'

        if seg:
            inds = self.get_inst_img_inds(instmap_path, inds)
            embeddings, ys = aggregate_sc_data(embeddings, inds, kys={'inds': inds})
            inds = ys['inds']
            print(">>>> Total {}: {}".format(tip_str, len(embeddings)))
        else:
            print(">>>> Total {}: {} Use: {}, NAN: {}".format(tip_str, n_total, len(embeddings), n_total-len(embeddings)))
        # embeddings = list(normalize_embeddings(embeddings))
        return embeddings, inds

    def get_inst_img_inds(self, dataset_path, inds):
        inst_map = np.load(dataset_path, allow_pickle=True).item()
        img_inds = [inst_map[indi] for indi in inds]
        return np.array(img_inds)

    def knn_classify(self, test_X, test_y, nclass, k, train_X=None, train_y=None,
                     dist_metric='euclidean', label_map=None, overwrite=False):
        test_X = np.array(test_X)
        test_y = np.array(test_y)
        if train_X is not None:
            train_X = np.array(train_X)
            train_y = np.array(train_y)

        label_inds = np.unique(test_y)  # 一共有几种类别
        if label_map is not None: nclass = len(label_map)
        correct_by_class = np.zeros(nclass, dtype=np.uint32)  #
        total_by_class = np.zeros(nclass, dtype=np.uint32)  #
        correct = 0.0
        total = 0.0

        # 构建模型并训练，Fit on the training set, if specified. Else, fit on the test set.
        knn_model = FaissKNeighbors(k=k, metric=dist_metric)  # 构建模型
        if train_X is not None:
            knn_model.fit(train_X, train_y)  # 模型训练
        else:
            knn_model.fit(test_X, test_y)

        predicts = []
        # 找到每个点的最近邻，Iterate through each sample. Match to existing samples.
        for i in trange(len(test_X)):  # 对于每张图片
            # Ignore duplicate point if kNN fitted on test set， 因为肯定会找到自己，所以k+1
            if train_y is None:
                neigh_dist, neigh_ind = knn_model.kneighbors([test_X[i, :]],
                                                             n_neighbors=k + 1)
                neigh_dist = neigh_dist[0][1:]
                neigh_ind = neigh_ind[0][1:]
                neigh_labels = test_y[neigh_ind]
            else:
                neigh_dist, neigh_ind = knn_model.kneighbors([test_X[i, :]],
                                                             n_neighbors=k)
                # remove outer list
                neigh_dist = neigh_dist[0]  # 最近的几个点的距离
                neigh_ind = neigh_ind[0]  # 最近的几个点的索引
                neigh_labels = train_y[neigh_ind]  # 最近的几个点的标签

            # 记录结果
            predicted = 0
            # if only one value predicted
            if isinstance(neigh_labels, int):
                predicted = neigh_labels
            # elif more than one non-unique label, get mode 有一类标签超过两个，取众数
            elif len(np.unique(neigh_labels)) < len(neigh_labels):
                predicted = stats.mode(neigh_labels)[0][0]
            # else, take the label of the closest point  否则，选择最近点的标签
            else:
                smallest_ind = np.argmin(neigh_dist)
                predicted = neigh_labels[smallest_ind]

            # Check prediction accuracy
            if predicted == test_y[i]:
                correct += 1.0
                correct_by_class[test_y[i]] += 1
            total += 1.0
            total_by_class[test_y[i]] += 1
            predicts.append(predicted)

        f1s = f1_score(test_y, predicts, average=None)
        print('f1s:', f1s, 'f1s_mean:', np.array(f1s).mean())

        # Save Results
        df_results = pd.DataFrame()
        correct_by_class = correct_by_class[label_inds]
        total_by_class = total_by_class[label_inds]
        df_results['correct_by_class'] = correct_by_class
        df_results['total_by_class'] = total_by_class
        df_results['accuracy_by_class'] = correct_by_class / total_by_class
        df_results['total_accuracy'] = correct / total
        # print("Include label inds", label_inds)
        print(df_results)
        return df_results
