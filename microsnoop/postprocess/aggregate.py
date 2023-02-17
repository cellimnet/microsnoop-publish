import numpy as np


def aggregate_sc_data(embeddings, inds, kys=None):
    """
    aggreate single-cell level representations to fully-imaged level representations
    """
    embeddings = np.array(embeddings)
    inds = np.array(inds)
    ys = {}
    for kyi in kys.keys(): ys[kyi] = []
    embeddingss = []

    inds_set = set(inds)
    for indi in inds_set:
        c = len(inds[inds==indi])
        embeddingi = embeddings[inds==indi].mean(axis=0)
        # embeddingi = np.median(embeddings[inds==indi], axis=0)
        embeddingss.append(embeddingi)

        if kys is not None:
            for kyi in kys.keys():
                y = np.array(kys[kyi])
                y = np.array(y[inds==indi])
                flag = True
                if c > 1:
                    for i in range(1, c):
                        if y.ndim == 1:
                            flag *= np.any(y[0] == y[i])
                        else:
                            flag *= np.any(y[..., 0] == y[..., i])
                assert flag, 'Maybe there are some problems during aggregate embeddings'
                if y.ndim == 1:
                    y = y[0]
                else:
                    y = y[..., 0]
                ys[kyi].append(y)

    if kys is not None:
        return embeddingss, ys
    else:
        return embeddingss


if __name__ == '__main__':
    inds = np.array([1,2,3,1,1,2])
    inds_set = set(inds)
    for indi in inds_set:
        c = len(inds[inds==indi])
        print(c)
