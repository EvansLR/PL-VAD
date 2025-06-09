
import numpy as np
import torch.utils.data as data
import utils.utils as utils
from options import *
from config import *
from train import *
from scripts.ucf_test import test
from model import *

import os
from dataset_loader import *
from tqdm import tqdm

if __name__ == "__main__":
    args = parse_args()
    config = Config(args)
    worker_init_fn = None
    gpus = [0] 
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    config.len_feature = 1024
    net = WSAD(config.len_feature, flag = "Train")
    net = net.cuda()




    normal_train_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature, is_normal = True),
            batch_size = config.batch_size,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    
    abnormal_train_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments =config.num_segments, len_feature = config.len_feature, is_normal = False),
            batch_size =config.batch_size,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    test_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
            batch_size =1,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)

    test_info = {"step": [], "auc": [],"ap":[],"ac":[],"b_auc":[]}
    
    best_auc = 0

    criterion = AD_Loss()
    
    optimizer = torch.optim.Adam(net.parameters(), lr = config.lr[0],
        betas = (0.9, 0.999), weight_decay = 0.00005)

 
    for step in tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,
            dynamic_ncols = True
        ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]
        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)

        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)
       
        train(net, normal_loader_iter,abnormal_loader_iter, optimizer, criterion, step)
        if step % 15 == 0 and step > 10:
        
            test(net, config, test_loader, test_info, step)
            utils.save_best_record(test_info, 
                os.path.join(config.output_path, "ucf_best_record_{}.txt".format(config.seed)))
            if test_info["auc"][-1] > best_auc:
                best_auc = test_info["auc"][-1]
                torch.save(net.state_dict(), os.path.join(args.model_path, \
                    "ucf_trans_{}.pkl".format(config.seed)))
                # 保存mem
                torch.save(net.Nor_m_items,'nor_mem.pt')
                torch.save(net.Abn_m_items,'abn_mem.pt')

            if step == config.num_iters:
                torch.save(net.state_dict(), os.path.join(args.model_path, \
                    "ucf_trans_{}.pkl".format(step)))

