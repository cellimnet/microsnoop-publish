import os, gc, math
import re, sys, time, datetime
project_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.path.sep + ".")
sys.path.append(project_path)
import numpy as np
from scellseg.io import imread
from microsnoop.eval import EvalProcedure
from microsnoop.misc import get_embed_args_parser
from microsnoop.preprocess.crop import crop_inst


class Dataset_livecell(EvalProcedure):

    def load_data(self, dataset_path, gen_size=None, seg=False, rsc_crop=True,
                  sta=224, rsc_crop_ratio=1, crop_step=1, oversize_to_sta=False):
        data = np.load(dataset_path, allow_pickle=True)
        img_paths = data['img_paths']
        dataset_dir = os.path.dirname(dataset_path)
        img_paths = [os.path.join(dataset_dir, img_path) for img_path in img_paths]
        if seg:
            mask_paths = [img_path.replace('_img.tif', '_masks.png') for img_path in img_paths]
        img_inds = np.array(range(0, len(img_paths)))

        if gen_size is None: gen_size=len(img_inds)
        ngens = math.ceil(len(img_inds) / gen_size)

        num_insts = 0
        for ngen in range(0, ngens):
            lind = ngen * gen_size
            rind = (ngen + 1) * gen_size if (ngen + 1) * gen_size < len(img_inds) else len(img_inds)
            imgs = np.array([imread(img_path) for img_path in img_paths[lind:rind]])
            imgs = imgs[:, np.newaxis, :, :]
            inds = img_inds[lind:rind]
            if seg:
                masks = np.array([imread(mask_path) for mask_path in mask_paths[lind:rind]])
                meta = {'inds': inds}
                imgs, masks, meta = crop_inst(imgs.transpose(0,2,3,1), masks, mode='bbox', rsc_crop=rsc_crop,
                                           meta=meta, oversize_to_sta=oversize_to_sta,
                                           sta=sta, rsc_crop_ratio=rsc_crop_ratio, crop_step=crop_step)
                inst_inds = np.array(range(0, len(imgs))) + num_insts
                num_insts += len(imgs)
                print('>>>> Accumulate insts num:', num_insts)
                imgs = np.array(imgs).transpose(0,3,1,2)
                inds = meta['inds']

                dataset_name = os.path.basename(dataset_path).split('_eval')[0]
                instmap_path = os.path.join(os.path.dirname(dataset_path), dataset_name+'_instmap.npy')
                inst_map = {}
                if os.path.isfile(instmap_path): inst_map = np.load(instmap_path, allow_pickle=True).item()
                inst_mapi = dict(zip(inst_inds, inds))
                inst_map.update(inst_mapi)
                np.save(instmap_path, inst_map)
                inds = inst_inds

            name_meta = os.path.basename(dataset_path).split('.npz')[0]+'_ind-'+str(lind)+'to'+str(rind-1)
            yield imgs, inds, name_meta
            del imgs, inds, name_meta
            gc.collect()

    def get_labels(self, dataset_path, inds):
        data = np.load(dataset_path, allow_pickle=True)
        labels = data['labels']
        return labels[inds]


if __name__ == '__main__':
    dataset_dir = r'/Data2/datasets'  # Note：input the root dir of your data

    dataset_name = 'livecell'
    output_dir = os.path.join(project_path, 'output')

    data_name = 'livecell_test'
    eval_dataset = Dataset_livecell()
    dataset_path = os.path.join(dataset_dir, dataset_name, data_name, '{}_eval.npz'.format(data_name))
    reshape_mode = 'concat'

    ###### 1. extract embeddings ######
    checkpoint_name = \
        r'20220904-1246_cnn_trData-microsnoop_trMode-mae_batchSize-16_inputSize-224_inChans-1_embedDim-256_maskRatio-0.25_epoch-999.pth'
    checkpoint_path = os.path.join(output_dir, 'models', checkpoint_name)
    model_type = str(re.findall(r"_(.+?)_trData", checkpoint_path)[0])
    args = get_embed_args_parser().parse_args()
    args.batch_size = 64  # Note: depend on GPU memory
    args.input_size = int(re.findall(r"_inputSize-(.+?)_", checkpoint_path)[0])
    args.embed_dim = int(re.findall(r"_embedDim-(.+?)_", checkpoint_path)[0])
    args.in_chans = int(re.findall(r"_inChans-(.+?)_", checkpoint_path)[0])
    if 'patchSize' in checkpoint_path:
        args.patch_size = int(re.findall(r"_patchSize-(.+?)_", checkpoint_path)[0])
    args.embed_dir = os.path.join(output_dir, 'embeddings', dataset_name)

    seg = True
    data_loader = eval_dataset.load_data(dataset_path, gen_size=300,
                                         seg=seg, sta=args.input_size, rsc_crop=True,
                                         rsc_crop_ratio=1)  # Note: ’gen_size‘： depend on GPU memory
    start_time = time.time()
    eval_dataset.extract_embeddings(dataset_name, data_loader, checkpoint_path, args, model_type=model_type,
                                    rsc_to_diam=1.0, rescale_to_input=False,
                                    tile=False)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('\033[1;33mTotal Embed Time: {} \033[0m'.format(total_time_str))

    ##### 2. prepare embeddings ######
    reshape_chans = 1
    filter_str = data_name
    instmap_path = os.path.join(dataset_dir, dataset_name, data_name, 'livecell_test_instmap.npy')
    embeddings, inds = eval_dataset.prepare_embeddings(args.embed_dir, reshape_chans, reshape_mode=reshape_mode, filter_str=filter_str, seg=seg, instmap_path=instmap_path)

    labels = eval_dataset.get_labels(dataset_path, inds)
    labelmap_path = os.path.join(dataset_dir, dataset_name, data_name, 'livecell_test_labelmap.npy')
    label_map = np.load(labelmap_path, allow_pickle=True).item()
    print('>>>> Label map:{}; Totally {} classes'.format(label_map, len(label_map)))

    ###### 3. knn classify ######
    classify_dir = os.path.join(output_dir, 'evaluation')
    if not os.path.isdir(classify_dir): os.makedirs(classify_dir)
    df_results = eval_dataset.knn_classify(embeddings, labels, nclass=len(label_map), k=1)
    df_results.to_csv(f"{classify_dir}/{dataset_name}_knn_classfy.csv", index=False)
