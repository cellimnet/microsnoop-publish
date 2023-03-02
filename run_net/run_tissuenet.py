import os, gc, math, h5py
import re, sys, time, datetime
project_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.path.sep + ".")
sys.path.append(project_path)  # 从命令行运行需要添加这个
import numpy as np
from microsnoop.eval import EvalProcedure
from microsnoop.misc import get_embed_args_parser
from microsnoop.preprocess.crop import crop_inst


class Dataset_tissuenet(EvalProcedure):

    def load_data(self, dataset_path, gen_size=None, seg=False, rsc_crop=True,
                  sta=224, rsc_crop_ratio=1, crop_step=1, oversize_to_sta=False):
        data = np.load(dataset_path, allow_pickle=True)
        images = data['X']
        images = images.transpose(0, 3, 1, 2)
        if seg:
            maskss = data['y']
            maskss = np.array([mask[..., 0] for mask in maskss])
        img_inds = np.array(range(0, len(images)))

        if gen_size is None: gen_size=len(img_inds)
        ngens = math.ceil(len(img_inds) / gen_size)
        num_insts = 0
        for ngen in range(0, ngens):
            lind = ngen * gen_size
            rind = (ngen + 1) * gen_size if (ngen + 1) * gen_size < len(img_inds) else len(img_inds)
            imgs = images[lind:rind]
            n, _, h, w = imgs.shape
            imgs = imgs.reshape(n//2, 2, h, w)
            inds = img_inds[lind:rind][::2]
            if seg:
                masks = maskss[lind:rind][::2]
                meta = {'inds': inds}
                imgs, masks, meta = crop_inst(imgs.transpose(0,2,3,1), masks, mode='bbox', rsc_crop=rsc_crop,
                                           meta=meta, oversize_to_sta=oversize_to_sta,
                                           sta=sta, rsc_crop_ratio=rsc_crop_ratio, crop_step=crop_step)
                inst_inds = np.array(range(0, len(imgs))) + num_insts
                num_insts += len(imgs)
                print('>>>> Accumulate insts num:', num_insts)
                imgs = np.array(imgs).transpose(0,3,1,2)
                inds = meta['inds']

                dataset_name = os.path.basename(dataset_path).split('.npz')[0]
                instmap_path = os.path.join(os.path.dirname(dataset_path), dataset_name+'_instmap.npy')
                inst_map = {}
                if os.path.isfile(instmap_path): inst_map = np.load(instmap_path, allow_pickle=True).item()
                inst_mapi = dict(zip(inst_inds, inds))  # 建立inds_inds和img_inds的对应关系, 方便后面获取label等元数据
                inst_map.update(inst_mapi)
                np.save(instmap_path, inst_map)
                inds = inst_inds

            name_meta = os.path.basename(dataset_path).split('.npz')[0]+'_ind-'+str(lind)+'to'+str(rind-1)
            yield imgs, inds, name_meta
            del imgs, inds, name_meta
            gc.collect()

    def get_platform(self, dataset_path, inds):
        data = np.load(dataset_path, allow_pickle=True)
        platform_list = data['platform_list']
        return platform_list[inds]

    def get_tissue(self, dataset_path, inds):
        data = np.load(dataset_path, allow_pickle=True)
        tissue_list = data['tissue_list']
        return tissue_list[inds]

    def get_labels(self, dataset_path, inds):
        tissue_list = self.get_tissue(dataset_path, inds)
        labels = tissue_list
        labelmap_path = os.path.join(os.path.dirname(dataset_path), 'tissuenet_labelmap.npy')
        label_map = np.load(labelmap_path, allow_pickle=True).item()
        print('>>>> Label map:{}; Totally {} classes'.format(label_map, len(label_map)))
        map_label = dict(zip(label_map.values(), label_map.keys()))
        labels = [map_label[labeli] for labeli in labels]
        return np.array(labels)


if __name__ == '__main__':
    dataset_dir = r'/Data1/files/example_datasets'  # Note：input the root dir of your data
    # dataset_dir = r'/Data1'  # Note：aws

    dataset_name = 'tissuenet'
    output_dir = os.path.join(project_path, 'output')

    data_name = 'tissuenet_test'
    eval_dataset = Dataset_tissuenet()
    dataset_path = os.path.join(dataset_dir, dataset_name, '{}.npz'.format(data_name))
    reshape_mode = 'concat'

    ###### 1. extract embeddings ######
    checkpoint_name = \
        r'Microsnoop_cnn_trData-microsnoop_trMode-mae_batchSize-16_inputSize-224_inChans-1_embedDim-256_maskRatio-0.25_epoch-999.pth'
    checkpoint_path = os.path.join(output_dir, 'models', checkpoint_name)
    model_type = str(re.findall(r"_(.+?)_trData", checkpoint_path)[0])
    args = get_embed_args_parser().parse_args()
    args.batch_size = 64
    args.input_size = int(re.findall(r"_inputSize-(.+?)_", checkpoint_path)[0])
    args.embed_dim = int(re.findall(r"_embedDim-(.+?)_", checkpoint_path)[0])
    args.in_chans = int(re.findall(r"_inChans-(.+?)_", checkpoint_path)[0])
    if 'patchSize' in checkpoint_path:
        args.patch_size = int(re.findall(r"_patchSize-(.+?)_", checkpoint_path)[0])
    args.embed_dir = os.path.join(output_dir, 'embeddings', dataset_name)

    seg = True
    data_loader = eval_dataset.load_data(dataset_path, gen_size=200,
                                         seg=seg, sta=args.input_size, rsc_crop=True,
                                         rsc_crop_ratio=1.5, crop_step=1)
    start_time = time.time()
    eval_dataset.extract_embeddings(dataset_name, data_loader, checkpoint_path, args, model_type=model_type,
                                    normalize=False, rsc_to_diam=1.0, rescale_to_input=False,
                                    tile=False) # rsc_to_diam: 0.4
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('\033[1;33mTotal Embed Time: {} \033[0m'.format(total_time_str))

    ##### 2. prepare embeddings ######
    reshape_chans = 1 if args.in_chans != 1 else 2
    filter_str = data_name
    instmap_path = os.path.join(dataset_dir, dataset_name, 'tissuenet_test_instmap.npy')
    embeddings, inds = eval_dataset.prepare_embeddings(args.embed_dir, reshape_chans, reshape_mode=reshape_mode, filter_str=filter_str, seg=seg, instmap_path=instmap_path)

    labels = eval_dataset.get_labels(dataset_path, inds)
    labelmap_path = os.path.join(dataset_dir, dataset_name, 'tissuenet_labelmap.npy')
    label_map = np.load(labelmap_path, allow_pickle=True).item()
    print('>>>> Label map:{}; Totally {} classes'.format(label_map, len(label_map)))

    ###### 3.1 plot embeddings ######
    # eval_dataset.plot_embeddings(embeddings, labels, label_map=label_map, mode='tsne', step=1)

    # ###### 3.2 knn classify ######
    classify_dir = os.path.join(output_dir, 'evaluation')
    if not os.path.isdir(classify_dir): os.makedirs(classify_dir)
    df_results = eval_dataset.knn_classify(embeddings, labels, nclass=len(label_map), k=1)
    df_results.to_csv(f"{classify_dir}/{dataset_name}_knn_classfy.csv", index=False)

    # df_results = eval_dataset.mlp_classify(embeddings, labels, len(label_map), nfold=5, unbalance=False)
    # df_results.to_csv(f"{classify_dir}/{dataset_name}_mlp_classfy.csv", index=False)

    # df_results = eval_dataset.dnn_classify(embeddings, labels, len(label_map), nfold=5, unbalance=False,
    #                                        depths=[256, 128, 64], use_gpu=True,
    #                                        lr=0.001, batch_size=1024, epoch=500)
    # df_results.to_csv(f"{classify_dir}/{dataset_name}_dnn_classfy.csv", index=False)
