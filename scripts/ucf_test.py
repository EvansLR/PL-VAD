import torch
from options import *
from config import *
from model import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

def test(net, config, test_loader, test_info, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/gt-ucf.npy")
        frame_predict = None
        
        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0)).cuda()

        b_frame_predict = None
        b_temp_predict = torch.zeros((0)).cuda()
        
        for i in range(len(test_loader.dataset)):
          
            _data, _label,name= next(load_iter)
            _data = _data.cuda()
            _label = _label.cuda()
            
            res = net(_data)

            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)


            b_predict=res["video"]
            b_temp_predict = torch.cat([b_temp_predict, b_predict], dim=0)

            if (i + 1) % 10 == 0 :
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()

                cls_pre.append(1 if a_predict.max()>0.5 else 0)          
                fpre_ = np.repeat(a_predict, 16)



                b_predict=b_temp_predict.mean(0).cpu().numpy()
                b_fpre_=np.repeat(b_predict,16)

                # 保存到文件中
                # 如果文件夹不存在，则创建文件夹
                path="UCF_test_NO_PM/"+str(step)+"/"
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(path+name[0],b_fpre_)



                if frame_predict is None:         
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])  
                temp_predict = torch.zeros((0)).cuda()

                if b_frame_predict is None:
                    b_frame_predict = b_fpre_
                else:
                    b_frame_predict = np.concatenate([b_frame_predict, b_fpre_])
                b_temp_predict = torch.zeros((0)).cuda()
   
        fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)

        bfpr,btpr,_ = roc_curve(frame_gt, b_frame_predict)
        b_auc_score = auc(bfpr, btpr)

    
        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))
        
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        ap_score = auc(recall, precision)



        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)
        test_info["b_auc"].append(b_auc_score)
        