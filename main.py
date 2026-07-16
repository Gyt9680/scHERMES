import tqdm
from torch.optim import Adam
from time import time
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from encoder import AE, IGAE
except ImportError:
    sys.path.append(os.getcwd())
    from encoder import AE, IGAE

from utils import *
from scHERMES import scHERMES
from data_loader import load_data
import opt

# 关闭 anomaly 检测以提高速度
torch.autograd.set_detect_anomaly(False)


def pretrain_ae(model, x):
    print("Pretraining AE...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.rec_epoch)):
        z = model.encoder(x)
        x_hat = model.decoder(z)
        loss = F.mse_loss(x_hat, x)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def pretrain_gae(model, x, adj):
    print("Pretraining GAE...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)

    with torch.no_grad():
        target_x = torch.spmm(adj, x)
        target_adj_dense = adj.to_dense()

    for epoch in tqdm.tqdm(range(opt.args.rec_epoch)):
        z, a = model.encoder(x, adj)
        z_hat, z_adj_hat = model.decoder(z, adj)
        a_hat = a + z_adj_hat

        loss_w = F.mse_loss(z_hat, target_x)
        loss_a = F.mse_loss(a_hat, target_adj_dense)
        loss = loss_w + opt.args.alpha_value * loss_a

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def pre_train(model, X1, A1, X2, A2):
    print("Pretraining scHERMES fusion model...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)

    lambda_cross = opt.args.lambda_cross_pretrain

    for epoch in tqdm.tqdm(range(opt.args.fus_epoch)):

        X_hat1, Z_hat1, A_hat1, X_hat2, Z_hat2, A_hat2, _, _, _, _, _, X_hat1_cross, X_hat2_cross = \
            model(X1, A1, X2, A2, pretrain=True)

        L_REC1 = reconstruction_loss(X1, A1, X_hat1, Z_hat1, A_hat1)
        L_REC2 = reconstruction_loss(X2, A2, X_hat2, Z_hat2, A_hat2)

        L_CROSS1 = F.mse_loss(X_hat1_cross, X1)
        L_CROSS2 = F.mse_loss(X_hat2_cross, X2)

        loss = L_REC1 + L_REC2 + lambda_cross * (L_CROSS1 + L_CROSS2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    os.makedirs(os.path.dirname(opt.args.pretrained_path), exist_ok=True)
    torch.save(model.state_dict(), opt.args.pretrained_path)
    print(f"[OK] saved pretrained weights -> {opt.args.pretrained_path}")


def train(model, X1, A1, X2, A2, y, reload_model=True):
    lambda_cross = opt.args.lambda_cross
    if not opt.args.pretrain:
        if reload_model:
            model_path = getattr(opt.args, 'pretrained_path', None) or './model_pretrained/{}_pretrain.pkl'.format(
                opt.args.name)
            if os.path.exists(model_path):
                print(f"Loading pretrained model from {model_path}...")
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            else:
                print(f"Warning: Pretrained model not found at {model_path}, training from scratch might be unstable!")
        else:
            print("Using in-memory pretrained model (Skipping disk reload)...")

        # KMeans 初始化
        with torch.no_grad():

            _, _, _, _, _, _, _, _, Z1, Z2, _, _, _ = model(X1, A1, X2, A2, pretrain=True)

        _, _, _, _, centers1 = clustering(Z1.detach(), y)
        _, _, _, _, centers2 = clustering(Z2.detach(), y)

        model.cluster_centers1.data = torch.tensor(centers1).to(opt.args.device)
        model.cluster_centers2.data = torch.tensor(centers2).to(opt.args.device)

    print("------------------------------------------------")
    print("Start scHERMES Clustering Training...")
    print("------------------------------------------------")

    optimizer = Adam(model.parameters(), lr=(opt.args.lr))
    best_epoch = 0

    for epoch in range(opt.args.epoch):

        X_hat1, Z_hat1, A_hat1, X_hat2, Z_hat2, A_hat2, Q1, Q2, Z1, Z2, _, X_hat1_cross, X_hat2_cross = \
            model(X1, A1, X2, A2)

        L_REC1 = reconstruction_loss(X1, A1, X_hat1, Z_hat1, A_hat1)
        L_REC2 = reconstruction_loss(X2, A2, X_hat2, Z_hat2, A_hat2)

        L_CROSS1 = F.mse_loss(X_hat1_cross, X1)
        L_CROSS2 = F.mse_loss(X_hat2_cross, X2)

        q1_detach = Q1[0].detach()
        q2_detach = Q2[0].detach()

        if opt.args.first_view == 'RNA':
            target_q = q1_detach if epoch % 400 < 200 else q2_detach
        else:
            target_q = q2_detach if epoch % 400 < 200 else q1_detach


        L_KL1 = distribution_loss(Q1, target_distribution(target_q))
        L_KL2 = distribution_loss(Q2, target_distribution(target_q))

        loss = L_REC1 + L_REC2 + \
               opt.args.lambda3 * (L_KL1 + L_KL2) + \
               lambda_cross * (L_CROSS1 + L_CROSS2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        q_sum = (Q1[0] + Q2[0]).detach()
        ari, nmi, ami, acc, y_pred = assignment(q_sum, y)

        print("Epoch {:03d} | Loss: {:.4f} | ARI: {:.4f} | NMI: {:.4f} | ACC: {:.4f}".format(
            epoch, loss.item(), ari, nmi, acc))

        if ari > opt.args.ari:
            opt.args.acc = acc
            opt.args.nmi = nmi
            opt.args.ari = ari
            opt.args.ami = ami
            best_epoch = epoch

    print("------------------------------------------------")
    print("Training Finished (scHERMES).")
    print("Best_epoch: {},".format(best_epoch), "ARI: {:.4f},".format(opt.args.ari),
          "NMI: {:.4f},".format(opt.args.nmi),
          "AMI: {:.4f}".format(opt.args.ami), "ACC: {:.4f}".format(opt.args.acc))

    if not os.path.exists('./output/{}'.format(opt.args.name)):
        os.makedirs('./output/{}'.format(opt.args.name))

    np.save('./output/{}/seed{}_label.npy'.format(opt.args.name, opt.args.seed), y_pred)
    np.save('./output/{}/seed{}_z.npy'.format(opt.args.name, opt.args.seed), ((Z1 + Z2) / 2).cpu().detach().numpy())


if __name__ == '__main__':
    print("scHERMES setting:")
    setup_seed(opt.args.seed)
    opt.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt.args.n_z = 64
    opt.args.pretrain = True

    print("------------------------------")
    print("dataset       : {}".format(opt.args.name))
    print("device        : {}".format(opt.args.device))
    print("random seed   : {}".format(opt.args.seed))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    opt.args.run_dir = os.path.join("./model", ts)
    os.makedirs(opt.args.run_dir, exist_ok=True)
    opt.args.pretrained_path = os.path.join(
        opt.args.run_dir,
        f"{opt.args.name}_seed{opt.args.seed}_scHERMES_pretrain.pkl"
    )
    print("pretrain_ckpt : {}".format(opt.args.pretrained_path))
    print("------------------------------")

    Xr, y, Ar = load_data(opt.args.name, 'RNA', opt.args.method, opt.args.k, show_details=False)
    Xa, y, Aa = load_data(opt.args.name, 'ATAC', opt.args.method, opt.args.k, show_details=False)
    opt.args.n_clusters = int(max(y) - min(y) + 1)

    # 获取节点数量 (样本数)
    num_nodes = Xr.shape[0]

    Xr = numpy_to_torch(Xr).to(opt.args.device)
    Ar = numpy_to_torch(Ar, sparse=True).to(opt.args.device)
    Xa = numpy_to_torch(Xa).to(opt.args.device)
    Aa = numpy_to_torch(Aa, sparse=True).to(opt.args.device)

    ae1 = AE(
        ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2,
        ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2,
        n_input=opt.args.n_d1, n_z=opt.args.n_z).to(opt.args.device)

    ae2 = AE(
        ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2,
        ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2,
        n_input=opt.args.n_d2, n_z=opt.args.n_z).to(opt.args.device)

    if opt.args.pretrain:
        opt.args.dropout = 0.4
    gae1 = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
        n_input=opt.args.n_d1, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)

    gae2 = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
        n_input=opt.args.n_d2, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)

    if opt.args.pretrain:
        t0 = time()
        print(">>> Stage 1: Component Pretraining...")
        pretrain_ae(ae1, Xr)
        pretrain_ae(ae2, Xa)
        pretrain_gae(gae1, Xr, Ar)
        pretrain_gae(gae2, Xa, Aa)

        print(">>> Stage 2: Fusion Pretraining (scHERMES)...")

        model = scHERMES(ae1, ae2, gae1, gae2, n_node=num_nodes).to(opt.args.device)
        pre_train(model, Xr, Ar, Xa, Aa)

        print(">>> Stage 3: Clustering Training...")
        opt.args.pretrain = False
        train(model, Xr, Ar, Xa, Aa, y, reload_model=True)

        t1 = time()
        print("Total Time Cost: {}".format(t1 - t0))
    else:
        t0 = time()

        model = scHERMES(ae1, ae2, gae1, gae2, n_node=num_nodes).to(opt.args.device)
        train(model, Xr, Ar, Xa, Aa, y, reload_model=True)
        t1 = time()
        print("Total Time Cost: {}".format(t1 - t0))