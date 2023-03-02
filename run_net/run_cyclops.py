import os, gc, math
import re, sys, time, datetime
import pandas as pd
project_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.path.sep + ".")
sys.path.append(project_path)
import numpy as np
from scellseg.io import imread
from microsnoop.eval import EvalProcedure
from microsnoop.misc import get_embed_args_parser, check_chans


class Dataset_cyclops(EvalProcedure):

    def load_data(self, dataset_path, gen_size=None):
        """
        for extract_embeddings
        """
        data = np.load(dataset_path, allow_pickle=True)
        img_paths = data['img_paths']
        dataset_dir = os.path.dirname(dataset_path)
        img_paths = [os.path.join(dataset_dir, img_path) for img_path in img_paths]
        img_inds = np.array(range(0, len(img_paths)))
        if gen_size is None: gen_size=len(img_inds)
        ngens = math.ceil(len(img_inds) / gen_size)
        for ngen in range(ngens):  # Task distribution module
            lind = ngen * gen_size
            rind = (ngen + 1) * gen_size if (ngen + 1) * gen_size < len(img_inds) else len(img_inds)
            imgs = np.array([imread(img_path) for img_path in img_paths[lind:rind]])
            n, h, w = imgs.shape
            imgs = imgs.reshape((int(n/2), 2, h, w))
            inds = img_inds[lind:rind][::2]
            name_meta = 'ind-'+str(lind)+'to'+str(rind-1)
            yield imgs, inds, name_meta
            del imgs, inds, name_meta
            gc.collect()

    def check_dataset_chans(self, dataset_path):
        """
        check dataset chans
        """
        data = np.load(dataset_path, allow_pickle=True)
        img_paths = data['img_paths']
        img_batchs = list(set([img_path.split('/')[0] for img_path in img_paths]))
        img_batchs.sort()
        check_results = {}
        for img_batch in img_batchs:
            img_paths_i = [img_path for img_path in img_paths if img_batch in img_path]
            img_paths_i.sort()
            check_result = check_chans(img_paths_i, ['_gfp', '_rfp'], task_name=img_batch)
            check_results[img_batch] = check_result
        print(pd.DataFrame(check_results, index=[0]).T.to_markdown())

    def get_labels(self, dataset_path, inds):
        data = np.load(dataset_path, allow_pickle=True)
        labels = data['labels']
        return labels[inds]


if __name__ == '__main__':
    dataset_dir = r'/Data1/files/example_datasets'  # Note：input the root dir of your data
    # dataset_dir = r'/Data1'  # Note：if aws

    dataset_name = 'cyclops'
    output_dir = os.path.join(project_path, 'output')

    eval_dataset = Dataset_cyclops()
    dataset_path = os.path.join(dataset_dir, dataset_name, 'cyclops_eval.npz')
    reshape_mode = 'concat'
    eval_dataset.check_dataset_chans(dataset_path)

    ###### 1. extract embeddings ######
    checkpoint_name = \
        r'Microsnoop_cnn_trData-microsnoop_trMode-mae_batchSize-16_inputSize-224_inChans-1_embedDim-256_maskRatio-0.25_epoch-999.pth'
    checkpoint_path = os.path.join(output_dir, 'models', checkpoint_name)
    model_type = str(re.findall(r"_(.+?)_trData", checkpoint_path)[0])
    args = get_embed_args_parser().parse_args()
    args.batch_size = 16  # Note: depend on GPU memory
    args.input_size = int(re.findall(r"_inputSize-(.+?)_", checkpoint_path)[0])
    args.embed_dim = int(re.findall(r"_embedDim-(.+?)_", checkpoint_path)[0])
    args.in_chans = int(re.findall(r"_inChans-(.+?)_", checkpoint_path)[0])
    if 'patchSize' in checkpoint_path:
        args.patch_size = int(re.findall(r"_patchSize-(.+?)_", checkpoint_path)[0])
    args.embed_dir = os.path.join(output_dir, 'embeddings', dataset_name)

    data_loader = eval_dataset.load_data(dataset_path, gen_size=1024)  # Note: ’gen_size‘： depend on GPU memory
    start_time = time.time()
    eval_dataset.extract_embeddings(dataset_name, data_loader, checkpoint_path, args, model_type=model_type,
                                    rsc_to_diam=1.0, rescale_to_input=False)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('\033[1;33mTotal Embed Time: {} \033[0m'.format(total_time_str))

    ###### 2. prepare embeddings ######
    reshape_chans = 1 if args.in_chans != 1 else 2
    filter_str = None
    seg = False
    embeddings, inds = eval_dataset.prepare_embeddings(args.embed_dir, reshape_chans, seg=seg, reshape_mode=reshape_mode, filter_str=filter_str)

    labels = eval_dataset.get_labels(dataset_path, inds)
    labelmap_path = os.path.join(dataset_dir, dataset_name, 'cyclops_labelmap.npy')
    label_map = np.load(labelmap_path, allow_pickle=True).item()
    print('>>>> Label map:{}; Totally {} classes'.format(label_map, len(label_map)))

    ###### 3. knn classify ######
    classify_dir = os.path.join(output_dir, 'evaluation')
    if not os.path.isdir(classify_dir): os.makedirs(classify_dir)
    df_results = eval_dataset.knn_classify(embeddings, labels, nclass=len(label_map), k=11, label_map=label_map)
    df_results.to_csv(f"{classify_dir}/{dataset_name}_knn_classfy.csv", index=False)
