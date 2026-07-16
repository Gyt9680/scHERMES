import os
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph


def _norm_adj_sparse(A):
    """
    等价于原来的 norm_adj：D^{-1/2} A D^{-1/2}
    使用显式掩码计算倒数平方根，数值稳定性最高，避免 np.power 的边缘情况
    """
    A = A.tocsr()
    # 展平并转为 float64 以保证精度
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)

    # 构造 D^(-1/2)
    deg_inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    deg_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])

    D = sp.diags(deg_inv_sqrt)
    # D * A * D，保持 CSR 格式
    return (D @ A @ D).tocsr()


def get_adj(count, k=15, mode="connectivity"):
    try:
        A = kneighbors_graph(count, k, mode=mode, metric="euclidean",
                             include_self=True, n_jobs=-1)
    except TypeError:
        A = kneighbors_graph(count, k, mode=mode, metric="euclidean",
                             include_self=True)

    adj_n = _norm_adj_sparse(A)
    return A, adj_n


def load_data(dataset, view, method, k, show_details=True):
    folder = './input/' + dataset + '/'
    label = np.load('{}label.npy'.format(folder), allow_pickle=True)
    fea = np.load('{}{}_fea.npy'.format(folder, view), allow_pickle=True)

    graph_path = '{}{}_{}_{}.npz'.format(folder, view, method, k)

    if not os.path.exists(graph_path):
        _, adj_n = get_adj(count=fea, k=k)

        # [优化] Error rate 计算：从 O(N^2) -> O(nnz)
        tmp = adj_n.tocsr(copy=True)
        tmp.setdiag(0)
        tmp.eliminate_zeros()

        r, c = tmp.nonzero()
        num = len(label)

        # 向量化对比
        counter = np.count_nonzero(label[r] != label[c])
        print('error rate: {}'.format(counter / (num * k)))

        sp.save_npz(graph_path, adj_n)

    # 直接加载 sparse，不要 toarray()
    adj = sp.load_npz(graph_path).tocsr()

    if show_details:
        print("---details of graph dataset---")
        print("------------------------------")
        print("dataset name         :", dataset + '_' + view)
        print("feature shape        :", fea.shape)
        print("label shape          :", label.shape)
        print("adj shape            :", adj.shape, "nnz:", adj.nnz)
        print("category num         :", len(np.unique(label)))  # 更准确的计算
        print("category distribution:")
        # [优化] 通用的 label 打印逻辑
        uniq = np.unique(label)
        for i in uniq:
            print("label", i, ":", np.sum(label == i))
        print("------------------------------")

    return fea, label, adj