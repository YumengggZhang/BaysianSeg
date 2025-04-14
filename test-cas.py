import os
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path

from monai.metrics import DiceMetric
from monai.visualize.utils import matshow3d, blend_images
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from torch.utils.data import DataLoader
from monai.metrics import compute_dice, do_metric_reduction
from monai.transforms import (
    Orientationd,
    CenterSpatialCropd,
    Invertd,
    Resize,
    Resized,
    NormalizeIntensityd,
    Spacingd,
    Transposed,
    ToDeviced,
    AsDiscrete,
    AsDiscreted,)

from models import build_model
from args import add_management_args, add_experiment_args, add_bayes_args
from data.transform import Mask2To1d, FilterOutBackgroundSliced, ClipHistogram,transform_test
from monai.data import MetaTensor
import SimpleITK as sitk
from data import build_dataset
from models.BayeSeg3D import BayeSeg_Criterion
from models.Basic_module import Criterion

class Tester:
    def __init__(self, args):
        self.args = args
        args.dataset = 'ImageCAS3d'
        self.checkpoint_dir = r'./logs/model/try'
        args.model = 'BayeSeg3d'

        self.plot = False
        self.visualize = False

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self.device = torch.device(args.device)

        self.model, _, _ = build_model(args)
        self.model.to(self.device)
        self.model_type = args.model
        self.criterion = BayeSeg_Criterion(args)
        dataset_test = build_dataset(image_set="test", args=args)[:10]
        self.test_loader = DataLoader(
            dataset_test,
            args.batch_size,
            False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        self.process_label = Compose([
        Resize(spatial_size=[256, 256, 128], mode='nearest'),
        AsDiscrete(to_onehot=args.num_classes, dim=0)])

        self.process_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes, dim=0)


        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])

        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("number of params:{}".format(n_parameters))

        self.model.eval()

        self.post_pred = Compose(
            [
                # ToDeviced(keys="pred", device="cpu"),
                # AsDiscreted(keys="pred", argmax=True, dim=1),
                Invertd(keys="pred", transform=transform_test, orig_keys="image"),
                AsDiscreted(keys="pred", argmax=True, to_onehot=args.num_classes, dim=0),
                AsDiscreted(keys="label", to_onehot=args.num_classes, dim=0),
                # Transposed(keys=["pred", "label"], indices=[3, 0, 1, 2])
                # ToDeviced(keys="image", device=self.device),

                ]
        )

        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    @torch.no_grad()
    def test_ImageCAS(self):

        patient_dices_before = []
        patient_dices_after = []

        total_step = len(self.test_loader)
        print('Test patient number: ',total_step)
        test_loader = iter(self.test_loader)
        plot_dir = '/outputs/unet_baye_crop'

        for step in range(total_step):
            data_dict = next(test_loader)
            samples = data_dict["image"].to(self.device)
            outputs = self.model(samples)

            # appearance_tensor = outputs["visualize"]['appearance'][1].squeeze(dim=0).permute(2,1,0)
            # appearance_np = appearance_tensor.cpu().numpy()
            # appearance_sitk = sitk.GetImageFromArray(appearance_np)
            # save_path = os.path.join(plot_dir,f"appearance_{step}.nii")
            # appearance_sitk = sitk.Cast(appearance_sitk, sitk.sitkFloat32)
            # sitk.WriteImage(appearance_sitk, save_path)

            # shape_tensor = outputs["visualize"]['shape'][0].squeeze(dim=0).permute(2,1,0)
            # shape_np = shape_tensor.cpu().numpy()
            # shape_sitk = sitk.GetImageFromArray(shape_np)
            # save_path = os.path.join(plot_dir,f"shape{step}.nii")
            # shape_sitk = sitk.Cast(shape_sitk, sitk.sitkFloat32)
            # sitk.WriteImage(shape_sitk, save_path)

            #calculate dice before resize
            data_dict['label'] = data_dict['label'].to('cuda')
            data_dict["pred"] = outputs["pred_masks"].squeeze(dim=0)

            data_dict["image"] = data_dict["image"].squeeze(dim=1)
            data_dict["label"] = data_dict["label"].squeeze(dim=1)
            targes = self.process_label(data_dict["label"])
            preds  = self.process_pred( data_dict["pred"])
            dice_before, _ = do_metric_reduction(
            compute_dice(targes, preds, include_background=False))
            patient_dices_before.append(dice_before.item())

            data_dict = self.post_pred(data_dict)
            data_dict['pred'] = data_dict['pred'].to('cuda')
            data_dict['label'] = data_dict['label'].to('cuda')
            #calculate dice with full mask
            dice_after, _ = do_metric_reduction(
            compute_dice(data_dict["pred"], data_dict["label"], include_background=False))
            patient_dices_after.append(dice_after.item())
            # print(data_dict['pred'].shape)
            # print(outputs["visualize"]["shape"].shape)

            # Plot the full prediction mask
            if self.plot == True:
                data_dict['pred'] = torch.argmax(data_dict["pred"], dim=0, keepdim=True)
                data_dict['label'] = torch.argmax(data_dict["label"], dim=0, keepdim=True)

                pred_tensor = data_dict['pred'][0].permute(2,1,0)
                # target_tensor = data_dict['label'][0].permute(2,1,0)
                pred_np = pred_tensor.cpu().numpy()
                # target_np = target_tensor.cpu().numpy()

                pred_sitk = sitk.GetImageFromArray(pred_np)
                # target_sitk = sitk.GetImageFromArray(target_np)

                # save_path = os.path.join(plot_dir,f"target_segmentation_{step}.nii")
                # target_sitk = sitk.Cast(target_sitk, sitk.sitkFloat32)
                # sitk.WriteImage(target_sitk, save_path)

                save_path = os.path.join(plot_dir,f"pred_segmentation_{step}.nii")
                pred_sitk = sitk.Cast(pred_sitk, sitk.sitkFloat32)
                sitk.WriteImage(pred_sitk, save_path)

        # compute dice
        avg_dice_before = torch.tensor(patient_dices_before).mean().item()
        avg_dice_after = torch.tensor(patient_dices_after).mean().item()

        print(f'Average Dice Score Before Resize: {avg_dice_before:.4f}')
        print(f'Average Dice Score After Resize: {avg_dice_after:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("BayeSeg testing", allow_abbrev=False)
    add_experiment_args(parser)
    add_management_args(parser)
    add_bayes_args(parser)
    args = parser.parse_args()

    tester = Tester(args)
    tester.test_ImageCAS()
