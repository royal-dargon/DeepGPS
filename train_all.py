"""用于训练所有模态信息的版本"""
import os
import argparse
import pickle
import logging

from tqdm import tqdm
import numpy as np
import wandb
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score, roc_curve
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from .model.model_config import ModelConfig
from .model.main_model import MainModel, AllModel
from .read_data import ProteinDataset, ProteinNewData, ProteinAllData

# Determine if it is a multi-card
IF_ONE_OR_MORE = True
num_card = 4

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset options.
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--root_data_path", type=str, default="/home/bingxing2/ailab/group/ai4agr/jinzhel/protein_work_data/dataset_splits/",
                        help="Root path of the dataset.")
    parser.add_argument("--data_type", type=str, default="protein_17kind", help="Type of the dataset.")

    parser.add_argument("--voxel_encode",type=str, default="simple", choices=["vot", "simple", "pointnet2",], help="Choose model to encode voxel data.")
    parser.add_argument("--seq_pretrain", type=bool, default=False, help="whether to use pretrain model to encode sequence data")
    parser.add_argument("--seq_encode", type=str, default="esm_2", choices=["protein_bert","esm_2"], help="Choose model to encode sequence data.")
    parser.add_argument("--combine", type=str, default="attention", choices=["concat_dim","concat_len","separate","attention"], help="Choose model to combine two modal inputs.")
    parser.add_argument("--classify", type=str, default="easy_mlp", choices=["mlp","fusion"], help="Choose model for classification.")  
    parser.add_argument("--loc_predict", type=str, default="unet_add", choices=["unet_concat","unet_add"], help="Choose model for location prediction.") 

    parser.add_argument("--hid_size", type=int, default=320, help="The size of hidden dimmension.")
    parser.add_argument("--seq_pad_id", type=int, default=1, help="The id of padding token in sequence encoder.")
    parser.add_argument("--voxel_size", type=int, default=200, help="The size of voxel data.")
    parser.add_argument("--block_size", type=int, default=20, help="The size of VoT block.")
    parser.add_argument("--vox_channels", type=int, default=3, help="The num of channels of voxel data.")
    parser.add_argument("--num_labels", type=int, default=2, help="The number of labels.")
    parser.add_argument("--dropout", type=float, default=0.1, help="The ratio to dropout parameters.")
    parser.add_argument("--init_factor", type=float, default=1.0, help="Factor for parameters normal_initialize.") 
    parser.add_argument("--predict_label", type=bool, default=True, help="Whether to do multi-labels prediction.")
    parser.add_argument("--use_voxel_enc", type=bool, default=False, help="Whether to use voxel encoder.")
    parser.add_argument("--use_loc_pre", type=bool, default=False, help="Whether to use location prediction model.")
    parser.add_argument("--in_channels", type=int, default=1, help="The channals of input image.")       
    parser.add_argument("--out_channels", type=int, default=1, help="The channals of output image.")
    parser.add_argument("--label_loss", type=str, default="bce", choices=["bce","mse","mae"], help="Loss function for multi-labels classification")
    parser.add_argument("--image_loss", type=str, default="mae", choices=["bce","mse","mae"], help="Loss function for image generation")

    # Training options
    parser.add_argument("--epochs_num", type=int, default=50, help="Number of epoches.")
    parser.add_argument("--report_steps", type=int, default=100, help="Number of steps in every which to report.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warm up value.")
    parser.add_argument("--train_batch_size_per_device", type=int, default=8, help="The size of one training batch per device.")
    parser.add_argument("--valid_batch_size_pre_device", type=int, default=8, help="The size of one validation batch per device.")
    parser.add_argument("--metric_label", type=str, default="sklearn", choices=["iou","accu","sklearn"], help="Metric of evalution for multi-label prediction")
    parser.add_argument("--metric_image", type=str, default="psnr", choices=["psnr","ssim"], help="Metric of evalution for image reconstruction")

    # Other options
    parser.add_argument("--seed", type=int, default=112, help="Random seed.")
    parser.add_argument("--result_path", type=str, default="/../", help="Path to store the model.")

    args = parser.parse_args()
    return args


def set_log(file_path):
    logger = logging.getLogger(file_path)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger


def plot_pic(train_res, valid_res, filepath):
    lenth = len(train_res)
    x = [i for i in range(1, lenth + 1)]

    plt.figure(figsize=(14.4, 14.4))
    print(x, train_res)
    plt.plot(x, train_res, 's-', color='r', label="train-res")
    plt.plot(x, valid_res, 'o-', color='g', label="valid-res")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("epochs", fontsize=14 )
    # plt.xlabel('Ls (Degree)', fontsize=14)
    plt.ylabel("PSNR", fontsize=14)
    # plt.ylabel('Ls (Degree)', fontsize=14)
    plt.legend(loc="best", fontsize=14)

    plt.savefig(filepath, dpi=1000)

def get_datasets(args, config):
    """获得训练集与测试集"""
    # load dataset
    train_data_path = os.path.join(args.root_data_path, f"{args.data_type}_train.pkl")
    valid_data_path = os.path.join(args.root_data_path, f"{args.data_type}_valid.pkl")

    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(valid_data_path, 'rb') as f:
        valid_data = pickle.load(f)

    # f_read_id2seq = None
    # if config.seq_pretrain:
    #     f_read_id2seq = open("/home/bingxing2/ailab/group/ai4agr/jinzhel/protein_work_data/pdb_data/ESM2_id_to_seq_list_se_tensor.pkl","rb")
    # elif not config.seq_pretrain:
    #     f_read_id2seq = open("/home/bingxing2/ailab/group/ai4agr/jinzhel/protein_work_data/pdb_data/ESM2_id_to_seq_list_se_tensor_mask.pkl","rb")
    # id2seq = pickle.load(f_read_id2seq)
    seq_path = "/home/bingxing2/ailab/group/ai4agr/jinzhel/protein_work_data/seq_data_esm2_650M"

    f_read_id2lab = open("/home/bingxing2/ailab/group/ai4agr/jinzhel/protein_work_data/label_data_1301/id_to_label_list_2kind.pkl","rb")
    id2lab = pickle.load(f_read_id2lab)

    # vr_point_surface_6w_color.pt
    f_read_point = open("/home/bingxing2/ailab/group/ai4agr/jinzhel/protein_work_data/voxel_data/vr_point_surface_6w_color.pt","rb")
    point_data = torch.load(f_read_point)

    nuc_path = "/home/bingxing2/ailab/group/ai4agr/jinzhel/protein_work_data/image_data/nuc_img"
    ptp_path = "/home/bingxing2/ailab/group/ai4agr/jinzhel/protein_work_data/image_data/ptp_img"
    train_dataset = ProteinAllData(pre_data=train_data, seq_path=seq_path, id2lab=id2lab, nuc_path=nuc_path, ptp_path=ptp_path, points=point_data)
    valid_dataset = ProteinAllData(pre_data=valid_data, seq_path=seq_path, id2lab=id2lab, nuc_path=nuc_path, ptp_path=ptp_path, points=point_data)
    print("DataSet Ready!")
    return train_dataset, valid_dataset


def main(args):
    # some run serve code: multi-gpu
    if IF_ONE_OR_MORE == True:
        dist.init_process_group(backend='nccl', init_method='env://',world_size=num_card, rank=int(os.environ['LOCAL_RANK']))

        print(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        device = torch.device("cuda", int(os.environ['LOCAL_RANK']))
        print(f"Device Ready : {os.environ['LOCAL_RANK']}")
    elif IF_ONE_OR_MORE == False:
        device = torch.device("cuda")
    elif IF_ONE_OR_MORE == "CPU":
        device = torch.device("cpu")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    config = ModelConfig(
        voxel_encode = args.voxel_encode,
        seq_pretrain = args.seq_pretrain,
        seq_encode = args.seq_encode,
        combine = args.combine,
        classify = args.classify,
        loc_predict = args.loc_predict,
        voxel_size = args.voxel_size,
        block_size = args.block_size,
        vox_channels = args.vox_channels,
        hid_size = args.hid_size,
        num_labels = args.num_labels,
        dropout = args.dropout,
        init_factor = args.init_factor,
        predict_label = args.predict_label, 
        in_channels = args.in_channels,
        out_channels = args.out_channels,
        label_loss = args.label_loss,
        image_loss = args.image_loss)
    

    model = AllModel(config)
    model = model.to(device)
    model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])], output_device=int(os.environ['LOCAL_RANK']), find_unused_parameters=True)

    # if int(os.environ['LOCAL_RANK']) == 0:
    #     run = wandb.init(
    #         project="protein_work_1",
    #         entity='jingzhel_cn',
    #         config=vars(args),
    #         name="all_img_3e5_ExponentialLR_16",
    #     )
    #     wandb.watch(model)


    # every experiments need to create a new path!
    if not os.path.exists(args.result_path) and int(os.environ['LOCAL_RANK']) == 0:
        os.mkdir(args.result_path)

    train_dataset, valid_dataset = get_datasets(args, config)
    
    instances_num = train_dataset.__len__()

    if int(os.environ['LOCAL_RANK']) == 0:
        valid_num = valid_dataset.__len__()
        print(f"The number of training instances: {instances_num}")
        print(f"The number of valid instances: {valid_num}")

    # this is needed when multi cards
    train_sampler = DistributedSampler(train_dataset, num_replicas=num_card, rank=int(os.environ['LOCAL_RANK']))
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=num_card, rank=int(os.environ['LOCAL_RANK']))
    # do not need sampler[None] when IF_ONE_OR_MORE is False; os.environ need to remove
    TrainLoader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size_per_device, sampler=train_sampler)
    ValidLoader = DataLoader(dataset=valid_dataset, batch_size=args.valid_batch_size_pre_device, sampler=valid_sampler)
    print(f"DataLoader Ready : {os.environ['LOCAL_RANK']}")

    #print("------------------------------------------- Optim -------------------------------------------")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    num_training_steps = int(instances_num * args.epochs_num) + 1
    num_warmup_steps = num_training_steps * args.warmup
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    # scheduler = ReduceLROnPlateau(optimizer,mode='min', factor=0.8, patience=5, threshold=1e-4, threshold_mode='abs', cooldown=0, min_lr=0.001, eps=1e-8)

    print(f"AdamW optimizer Ready : {os.environ['LOCAL_RANK']}")

    # print("------------------------------------------- Main -------------------------------------------")
    total_loss = 0.
    train_res = []
    valid_res = []
    all_pic = os.path.join(args.result_path, f"2kind_pic.jpg")
    
    for epoch in range(1, args.epochs_num+1):
        train_sampler.set_epoch(epoch)
        #print("------------------------------------------- Train -------------------------------------------")
        if int(os.environ['LOCAL_RANK']) == 0:
            print("Start training in epoch:", epoch)

        train_image_quality = 0.0
        train_total_samples = 0
        
        model.train()
        train_nuc_img = np.ones((1, 208, 208))
        train_res_img = np.ones((1, 208, 208))
        train_gt_img = np.ones((1, 208, 208))
        for i, (seq, lab, nuc, ptp, points) in tqdm(enumerate(TrainLoader)):
            optimizer.zero_grad()

            seq = seq.to(device)
            lab = lab.to(device)
            nuc = nuc.to(device)
            ptp = ptp.to(device)
            points = points.to(device)
            lab = lab.float()
            # 包含分类结果
            loss, _, outputs = model(x_seq = seq, x_img = nuc, y_lab = lab, y_img = ptp, p = points)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() 

            res_np = outputs.cpu().squeeze().detach().numpy()
            nuc_np = nuc.cpu().squeeze().detach().numpy()
            gt_np = ptp.cpu().squeeze().detach().numpy()

            train_nuc_img = np.append(train_nuc_img, nuc_np, axis=0)
            train_res_img = np.append(train_res_img, res_np, axis=0)
            train_gt_img = np.append(train_gt_img, gt_np, axis=0)         

            if (i + 1) % args.report_steps == 0 and int(os.environ['LOCAL_RANK']) == 0:
                current_learning_rate = optimizer.param_groups[0]['lr']
                wandb.log({
                    "train_loss": total_loss / args.report_steps,
                    "learn_rate": current_learning_rate,
                })
                total_loss = 0.

            train_total_samples += nuc.size(0)
            if i > 1:
                break

        if int(os.environ['LOCAL_RANK']) == 0:
            wandb.log({"train_samples_res": [wandb.Image(train_gt_img[1]), wandb.Image(train_res_img[1]), wandb.Image(train_nuc_img[1]),
                                            wandb.Image(train_gt_img[2]), wandb.Image(train_res_img[2]), wandb.Image(train_nuc_img[2]),
                                            wandb.Image(train_gt_img[3]), wandb.Image(train_res_img[3]), wandb.Image(train_nuc_img[3]),
                                            wandb.Image(train_gt_img[4]), wandb.Image(train_res_img[4]), wandb.Image(train_nuc_img[4]),],
            })

        scheduler.step()
        if args.metric_image == "psnr":
            train_image_quality = peak_signal_noise_ratio(train_res_img[1:], train_gt_img[1:], data_range=1)
        elif args.metric_image == "ssim":
            train_image_quality = structural_similarity(train_res_img[1:], train_gt_img[1:], data_range=1)
        
        train_image_quality_cuda = torch.as_tensor(torch.from_numpy(np.array(train_image_quality)), dtype=torch.float32).to(device)
        total_samples_tensor = torch.as_tensor(torch.from_numpy(np.array(train_total_samples)), dtype=torch.float32).to(device)
        # all_img_quality = torch.as_tensor(torch.from_numpy(np.array(train_all_image_quality.cpu())), dtype=torch.float32).to(device)
        
        if torch.cuda.device_count() > 1:
            torch.distributed.all_reduce(train_image_quality_cuda, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
        
        if int(os.environ['LOCAL_RANK']) == 0:
            # 除卡数量
            avg_img_quality = train_image_quality_cuda / num_card
            wandb.log({
                "Train_PSNR": avg_img_quality
            })
            train_res.append(avg_img_quality.cpu().item())
        

        print("------------------------------------------- Valid -------------------------------------------")
        print(f"Start validation in epoch: {epoch}")
        model.eval()      
            
        all_acc_lab = 0.0
        all_f1_lab = 0.0
        all_R_lab = 0.0
        all_precision = 0.0
        test_image_quality = 0.0
        total_samples = 0
        all_sensitivity = 0.0
        all_specificity = 0.0
        all_auroc = 0.0
        
        test_nuc_img = np.ones((1, 208, 208))
        test_res_img = np.ones((1, 208, 208))
        test_gt_img = np.ones((1, 208, 208))
        test_pre = np.ones((1, 1))
        test_truth = np.ones((1, 1))
        test_scores = np.ones((1, 2))

        best_avg_img_quality = 0.0
        best_model = None
        test_loss = 0
        for j, (seq, lab, nuc, ptp, points) in tqdm(enumerate(ValidLoader)):
                
            seq = seq.to(device)
            lab = lab.to(device)
            nuc = nuc.to(device)
            ptp = ptp.to(device)
            points = points.to(device)
            lab = lab.float()

            with torch.no_grad():
                loss, res_cls, logits_img = model(x_seq=seq, x_img=nuc, y_lab=lab, y_img=ptp, p=points)

            test_loss += loss.item() 

            res_np = logits_img.cpu().squeeze().numpy()
            nuc_np = nuc.cpu().squeeze().numpy()
            gt_np = ptp.cpu().squeeze().numpy()

            test_nuc_img = np.append(test_nuc_img, nuc_np, axis=0)
            test_res_img = np.append(test_res_img, res_np, axis=0)
            test_gt_img = np.append(test_gt_img, gt_np, axis=0)
            
            # 计算总样本数量
            total_samples += nuc.size(0)

            if config.predict_label:
                predictions_np = res_cls.cpu()
                labels_np = lab.cpu()
                _, labels_np = torch.max(labels_np, 1)
                _, predictions_np = torch.max(predictions_np, 1)
                scores_np = res_cls.cpu().numpy()
                labels_np = labels_np.numpy().reshape((-1, 1))
                pre_np = predictions_np.numpy().reshape((-1, 1))
                # scores_np = scores.numpy()
                # print(test_pre, pre_np)
                # print(test_pre.shape, pre_np.shape)
                test_pre = np.append(test_pre, pre_np, axis=0)
                test_truth = np.append(test_truth, labels_np, axis=0)
                test_scores = np.append(test_scores, scores_np, axis=0)
                # print(labels_np, pre_np)

        if args.metric_image == "psnr":
            test_image_quality = peak_signal_noise_ratio(test_res_img[1:], test_gt_img[1:], data_range=1)
        elif args.metric_image == "ssim":
            test_image_quality = structural_similarity(test_res_img[1:], test_gt_img[1:], data_range=1)
        
        if config.predict_label:
            acc_lab = accuracy_score(test_truth[1:], test_pre[1:])
            f1_lab = f1_score(test_truth[1:], test_pre[1:])
            R_lab = recall_score(test_truth[1:], test_pre[1:])
            precision = precision_score(test_truth[1:], test_pre[1:])
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(test_truth[1:], test_pre[1:]).ravel()
            # 计算敏感性和特异性
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            # 计算AUROC
            # labels_binarized = label_binarize(test_truth[1:], classes=[0, 1])
            # print(labels_binarized.shape, test_scores.shape)
            # auroc = roc_auc_score(labels_binarized, test_scores[1:], average='macro', multi_class='ovr')
            # auroc = roc_auc_score(test_truth, test_scores)

        # 全部指标基元转成tensor
        total_samples_tensor = torch.as_tensor(torch.from_numpy(np.array(total_samples)), dtype=torch.float32).to(device)
        # all_img_quality = torch.as_tensor(torch.from_numpy(np.array(all_img_quality.cpu())), dtype=torch.float32).to(device)
        test_image_quality_cuda = torch.as_tensor(torch.from_numpy(np.array(test_image_quality)), dtype=torch.float32).to(device)

        if config.predict_label:
            all_acc_lab = torch.tensor(acc_lab).to(device)
            all_f1_lab = torch.tensor(f1_lab).to(device)
            all_R_lab = torch.tensor(R_lab).to(device)
            all_precision = torch.tensor(precision).to(device)
            all_sensitivity = torch.tensor(sensitivity).to(device)
            all_specificity = torch.tensor(specificity).to(device)
            # all_auroc = torch.tensor(auroc).to(device)

        if torch.cuda.device_count() > 1:
            if config.predict_label : 
                torch.distributed.all_reduce(all_acc_lab, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(all_f1_lab, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(all_R_lab, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(all_precision, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(all_sensitivity, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(all_specificity, op=torch.distributed.ReduceOp.SUM)
                # torch.distributed.all_reduce(all_auroc, op=torch.distributed.ReduceOp.SUM)
                    
            torch.distributed.all_reduce(test_image_quality_cuda, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)

        if int(os.environ['LOCAL_RANK']) == 0:
            wandb.log({"test_samples_res": [wandb.Image(test_gt_img[1]), wandb.Image(test_res_img[1]), wandb.Image(test_nuc_img[1]),
                                            wandb.Image(test_gt_img[2]), wandb.Image(test_res_img[2]), wandb.Image(test_nuc_img[2]),
                                            wandb.Image(test_gt_img[3]), wandb.Image(test_res_img[3]), wandb.Image(test_nuc_img[3]),
                                            wandb.Image(test_gt_img[4]), wandb.Image(test_res_img[4]), wandb.Image(test_nuc_img[4]),],
            })
            avg_img_quality = test_image_quality_cuda / num_card
            avg_acc = all_acc_lab / num_card
            all_f1 = all_f1_lab / num_card
            all_R = all_R_lab / num_card
            all_precision = all_precision / num_card
            all_sensitivity = all_sensitivity / num_card
            all_specificity = all_specificity / num_card
            # all_auroc = all_auroc / num_card
            # all_R = all_R_lab / total_samples_tensor
        
            if epoch == 1:
                best_avg_img_quality = avg_img_quality
                best_model = model
            elif epoch > 1 and best_avg_img_quality < avg_img_quality:
                best_avg_img_quality = avg_img_quality
                best_model = model
            last_model = model

            wandb.log({
                "test_psnr": avg_img_quality,
                "cl_acc": avg_acc,
                "cl_f1": all_f1,
                "cl_R": all_R,
                "cl_p": all_precision,
                "all_sensitivity": all_sensitivity,
                "all_specificity": all_specificity
                # "all_auroc": all_auroc
            })
            valid_res.append(avg_img_quality.cpu().item())

    if int(os.environ['LOCAL_RANK']) == 0:
        save_path = os.path.join(args.result_path, f"2kind_BestModel.pth")
        last_save_path = os.path.join(args.result_path, f"2kind_LastModel.pth")
        torch.save(best_model.module.state_dict(), save_path)
        torch.save(last_model.module.state_dict(), last_save_path)

    dist.barrier()
    if int(os.environ['LOCAL_RANK']) == 0:
        plot_pic(train_res, valid_res, all_pic)

if __name__ == "__main__":
    args = get_args()
    main(args=args)
