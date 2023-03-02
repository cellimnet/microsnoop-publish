import os, gc, math
import re, sys, time, datetime
project_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.path.sep + ".")
sys.path.append(project_path)
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from scellseg.io import imread
from microsnoop.eval import EvalProcedure
from microsnoop.preprocess.crop import crop_inst
from microsnoop.misc import get_embed_args_parser, check_chans
from microsnoop.postprocess.spherize import ZCA_corr
from imblearn.over_sampling import BorderlineSMOTE


class Dataset_bbbc021(EvalProcedure):
    def load_data(self, dataset_path, gen_size=None, seg=False, rsc_crop=True, crop_to_sta=False,
                  sta=224, rsc_crop_ratio=2.5, crop_step=1, oversize_to_sta=False):
        data = np.load(dataset_path, allow_pickle=True)
        img_paths = data['img_paths']
        dataset_dir = os.path.dirname(dataset_path)
        if seg:
            mask_names = [img_path.replace('/', '-') for img_path in img_paths if '_w1' in img_path]
            mask_names = [os.path.basename(mask_name).split('_w1')[0] for mask_name in mask_names]
            mask_paths = [os.path.join(dataset_dir, 'eval_masks', mask_name+'_masks.png') for mask_name in mask_names]
        img_paths = [os.path.join(dataset_dir, img_path) for img_path in img_paths]
        img_inds = np.array(range(0, len(img_paths)))

        # for img_path, label in zip(img_paths, labels):
        #     print(img_path,'   |   ', label)
        if gen_size is None: gen_size=len(img_inds)
        ngens = math.ceil(len(img_inds) / gen_size)

        num_insts = 0
        for ngen in range(0, ngens):  # Task distribution module
            lind = ngen * gen_size
            rind = (ngen + 1) * gen_size if (ngen + 1) * gen_size < len(img_inds) else len(img_inds)
            imgs = np.array([imread(img_path) for img_path in img_paths[lind:rind]])
            n, h, w = imgs.shape
            imgs = imgs.reshape((int(n/3), 3, h, w))
            inds = img_inds[lind:rind][::3]

            if seg:
                lind_ = lind//3
                rind_ = rind//3
                masks = [imread(mask_path) for mask_path in mask_paths[lind_:rind_]]
                meta = {'inds': inds}
                imgs, masks, meta = crop_inst(imgs.transpose(0,2,3,1), masks, mode='bbox', rsc_crop=rsc_crop,
                                           meta=meta, oversize_to_sta=oversize_to_sta,
                                           sta=sta, crop_to_sta=crop_to_sta, rsc_crop_ratio=rsc_crop_ratio, crop_step=crop_step)
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
            check_result = check_chans(img_paths_i, ['_w1', '_w2', '_w4'], task_name=img_batch)
            check_results[img_batch] = check_result
        print(pd.DataFrame(check_results, index=[0]).T.to_markdown())

    def polish_embeddings(self, dataset_path, reshape_chans, embeddings, inds):
        """ The embeddings in seg mode are <= the actual number of images due to the fact
        that some images do not contain any single cells """
        data = np.load(dataset_path, allow_pickle=True)
        img_paths = data['img_paths']
        img_inds = np.array(range(0, len(img_paths)))
        img_inds = img_inds[::reshape_chans]
        miss_inds = list(set(img_inds).difference(set(inds)))
        if len(miss_inds) > 0:
            embeddings += [np.zeros(embeddings[0].shape[0]) for i in range(len(miss_inds))]
            inds += list(miss_inds)
        return embeddings, inds

    def get_labels(self, dataset_path, inds):
        data = np.load(dataset_path, allow_pickle=True)
        labels = data['labels']
        return labels[inds]

    def get_compounds(self, dataset_path, inds):
        data = np.load(dataset_path, allow_pickle=True)
        compounds = data['compounds']
        return compounds[inds]

    def get_concentrations(self, dataset_path, inds):
        data = np.load(dataset_path, allow_pickle=True)
        concentrations = data['concentrations']
        return concentrations[inds]

    def get_batches(self, dataset_path, inds, level='plate'):
        data = np.load(dataset_path, allow_pickle=True)
        img_paths = data['img_paths']
        img_paths = list(img_paths)
        batches = [img_path.split('/')[0] for img_path in img_paths]
        if level=='week':
            batches = [batch.split('_')[0] for batch in batches]
        batches = np.array(batches)
        return batches[inds]

    def bbbc021_nsc_classify(self, test_X, test_y, compounds, unbalance=False):
        """
        refer to https://github.com/broadinstitute/DeepProfilerExperiments/tree/master/bbbc021
        """
        treatments = pd.DataFrame({'embeddings': list(test_X), 'labels': list(test_y), 'Compound': list(compounds)})
        model = KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric="cosine")

        correct, total = 0, 0
        for i in treatments.index:
            # Leave one compound out
            mask = treatments["Compound"] != treatments.loc[i, "Compound"]
            trainFeatures = np.array(treatments['embeddings'][mask].to_list())
            trainLabels = np.array(treatments['labels'][mask].to_list())
            testFeatures = np.array(treatments['embeddings'][[i]].to_list())
            testLabelsi = np.array(treatments['labels'][[i]].to_list())

            if unbalance:
                sm = BorderlineSMOTE(random_state=42, kind="borderline-1", k_neighbors=1)
                trainFeatures, trainLabels = sm.fit_resample(trainFeatures, trainLabels)

            model.fit(trainFeatures, trainLabels)
            prediction = model.predict(testFeatures)

            # Check prediction
            if testLabelsi[0] == prediction[0]:
                correct += 1
            total += 1
        print("NSC Accuracy: {} correct out of {} = {}".format(correct, total, correct / total))
        nn_acc = correct / total
        return nn_acc

    def bbbc021_nscb_classify(self, treatments, unbalance=False):
        """
        refer to https://github.com/broadinstitute/DeepProfilerExperiments/tree/master/bbbc021
        """
        # Cholesterol-lowering and Kinase inhibitors are only in one batch
        valid_treatments = treatments[~treatments["label_names"].isin(["Cholesterol-lowering", "Kinase inhibitors"])]

        model = KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric="cosine")

        correct, total = 0, 0
        for i in valid_treatments.index:
            # Leave one compound out
            mask1 = valid_treatments["Compound"] != valid_treatments.loc[i, "Compound"]
            mask2 = valid_treatments["Batch"] != valid_treatments.loc[i, "Batch"]
            mask = mask1 & mask2
            trainFeatures = np.array(valid_treatments['embeddings'][mask].to_list())
            trainLabels = np.array(valid_treatments['labels'][mask].to_list())
            testFeatures = np.array(valid_treatments['embeddings'][[i]].to_list())
            testLabelsi = np.array(valid_treatments['labels'][[i]].to_list())

            if unbalance:
                sm = BorderlineSMOTE(random_state=42, kind="borderline-1", k_neighbors=1)
                trainFeatures, trainLabels = sm.fit_resample(trainFeatures, trainLabels)

            model.fit(trainFeatures, trainLabels)
            prediction = model.predict(testFeatures)

            # Check prediction
            if testLabelsi[0] == prediction[0]:
                correct += 1
            total += 1
        print("NSCB Accuracy: {} correct out of {} = {}".format(correct, total, correct / total))
        nn_acc = correct / total
        return nn_acc


if __name__ == '__main__':
    dataset_dir = r'/Data1/files/example_datasets'  # Note：input the root dir of your data

    dataset_name = 'bbbc021'
    output_dir = os.path.join(project_path, 'output')

    data_names = ['bbbc021_comp', 'bbbc021_dmso']
    for data_name in data_names:
        eval_dataset = Dataset_bbbc021()
        dataset_path = os.path.join(dataset_dir, dataset_name, data_name+'_eval.npz')
        reshape_mode = 'concat'
        eval_dataset.check_dataset_chans(dataset_path)

        ###### 1. extract embeddings ######
        checkpoint_name = \
            r'Microsnoop_cnn_trData-microsnoop_trMode-mae_batchSize-16_inputSize-224_inChans-1_embedDim-256_maskRatio-0.25_epoch-999.pth'
        checkpoint_path = os.path.join(output_dir, 'models', checkpoint_name)
        model_type = str(re.findall(r"_(.+?)_trData", checkpoint_path)[0])
        args = get_embed_args_parser().parse_args()
        args.batch_size = 96  # Note: depend on GPU memory
        args.input_size = int(re.findall(r"_inputSize-(.+?)_", checkpoint_path)[0])
        args.embed_dim = int(re.findall(r"_embedDim-(.+?)_", checkpoint_path)[0])
        args.in_chans = int(re.findall(r"_inChans-(.+?)_", checkpoint_path)[0])
        if 'patchSize' in checkpoint_path:
            args.patch_size = int(re.findall(r"_patchSize-(.+?)_", checkpoint_path)[0])
        args.embed_dir = os.path.join(output_dir, 'embeddings', dataset_name)

        seg = True
        data_loader = eval_dataset.load_data(dataset_path, gen_size=300,
                                             seg=seg, rsc_crop=True,
                                             sta=224, crop_to_sta=False,
                                             rsc_crop_ratio=2)  # Note: ’gen_size‘： depend on CPU memory
        start_time = time.time()
        eval_dataset.extract_embeddings(dataset_name, data_loader, checkpoint_path, args, model_type=model_type,
                                        rsc_to_diam=1.0, rescale_to_input=False,
                                        tile=False, tile_overlap=0.1, tile_size=224)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('\033[1;33mTotal Embed Time: {} \033[0m'.format(total_time_str))

    ###### 2. prepare embeddings ######
    dataset_path = os.path.join(dataset_dir, dataset_name, 'bbbc021_comp_eval.npz')
    reshape_chans = 1 if args.in_chans != 1 else 3
    instmap_path = os.path.join(dataset_dir, dataset_name, 'bbbc021_comp_instmap.npy')
    embeddings, inds = eval_dataset.prepare_embeddings(args.embed_dir, reshape_chans, reshape_mode=reshape_mode, filter_str='bbbc021_comp',
                                                       seg=seg, instmap_path=instmap_path)

    embeddings, inds = eval_dataset.polish_embeddings(dataset_path, reshape_chans, embeddings, inds)
    batch_level = 'plate'

    dmso_dataset_path = os.path.join(dataset_dir, dataset_name, 'bbbc021_dmso_eval.npz')
    dmso_instmap_path = os.path.join(dataset_dir, dataset_name, 'bbbc021_dmso_instmap.npy')
    dmso_embeddings, dmso_inds = eval_dataset.prepare_embeddings(args.embed_dir, reshape_chans, reshape_mode=reshape_mode, filter_str='bbbc021_dmso',
                                                                 seg=seg, instmap_path=dmso_instmap_path)

    dmso_embeddings, dmso_inds = eval_dataset.polish_embeddings(dmso_dataset_path, reshape_chans, dmso_embeddings, dmso_inds)
    dmso_batches = eval_dataset.get_batches(dmso_dataset_path, dmso_inds, level=batch_level)
    dmso_embed_df_batches = pd.DataFrame({'embeddings':dmso_embeddings, 'batches': dmso_batches, 'inds': dmso_inds})

    """
    Spherize embeddings
    refer to https://github.com/jump-cellpainting/2021_Chandrasekaran_submitted/blob/main/benchmark/old_notebooks/3.spherize_profiles.ipynb
    """
    batches = eval_dataset.get_batches(dataset_path, inds, level=batch_level)
    batches_all = list(set(batches))
    embed_df_batches = pd.DataFrame({'embeddings':embeddings, 'batches': batches, 'inds': inds})
    embeddings_batches = []
    inds_batches = []
    for batchi in batches_all:
        dmso_embeddings_batch = np.array(dmso_embed_df_batches.query('batches==@batchi').embeddings.to_list())
        spherizer = ZCA_corr()
        spherizer.fit(dmso_embeddings)

        embeddings_batch = np.array(embed_df_batches.query('batches==@batchi').embeddings.to_list())
        embeddings_batch = spherizer.transform(embeddings_batch)
        embeddings_batches += list(embeddings_batch)
        inds_batch = np.array(embed_df_batches[batches == batchi].inds.to_list())
        inds_batches += list(inds_batch)
    embeddings = embeddings_batches
    inds = np.array(inds_batches)

    labels = eval_dataset.get_labels(dataset_path, inds)
    compounds = eval_dataset.get_compounds(dataset_path, inds)
    aggregate_on_treatment = True

    ### aggregate on treatment
    concentrations = eval_dataset.get_concentrations(dataset_path, inds)
    batches = eval_dataset.get_batches(dataset_path, inds, level=batch_level)
    embed_df = pd.DataFrame({'embeddings':embeddings, 'compounds':compounds, 'concentrations':concentrations,
                             'labels':labels, 'batches': batches})
    embed_df['treatment'] = embed_df.apply(lambda x: f"{x.compounds}-{x.concentrations}", axis=1)
    labels = embed_df.groupby(by=['treatment']).sample(n=1).labels.to_numpy()
    compounds = embed_df.groupby(by=['treatment']).sample(n=1).compounds.to_numpy()
    treatments = embed_df.groupby(by=['treatment']).sample(n=1).treatment.to_numpy()
    embeddings = np.array(embed_df.groupby(by=['treatment']).embeddings.mean().to_list())
    ################################

    labelmap_path = os.path.join(dataset_dir, dataset_name, 'bbbc021_comp_labelmap.npy')
    label_map = np.load(labelmap_path, allow_pickle=True).item()
    print('>>>> Label map:{}, Totally {} classes'.format(label_map, len(label_map)))

    ###### 3. knn classify ######
    classify_dir = os.path.join(output_dir, 'evaluation')
    if not os.path.isdir(classify_dir): os.makedirs(classify_dir)

    df_results = eval_dataset.bbbc021_nsc_classify(embeddings, labels, compounds)

    label_names = [label_map[i] for i in labels]
    treatments_df = pd.DataFrame({'embeddings': list(embeddings), 'labels': list(labels), 'label_names': label_names,
                                  'Compound': list(compounds), 'treatments': list(treatments)})
    treatments_df["Batch"] = ""
    for i in treatments_df.index:
        result = embed_df.query("treatment == '{}'".format(treatments_df.loc[i, "treatments"]))
        treatments_df.loc[i, "Batch"] = ",".join(result["batches"].unique())
    df_results = eval_dataset.bbbc021_nscb_classify(treatments_df)
