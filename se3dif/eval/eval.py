import os
import copy
import configargparse
import tqdm
from se3dif.eval.earth_mover_distance import earth_mover_distance
from se3dif.samplers.grasp_samplers import Grasp_AnnealedLD
from se3dif.utils import get_root_src

import torch
from torch.utils.data import DataLoader

from se3dif import datasets, losses, summaries, trainer
from se3dif.models import loader
from se3dif.utils import load_experiment_specifications
from se3dif.trainer.learning_rate_scheduler import get_learning_rate_schedules

base_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.dirname(__file__ + '/../../../../../'))


def parse_args():
    p = configargparse.ArgumentParser()

    p.add_argument('--device',  type=str, default='cuda',)
    p.add_argument('--class_type', type=str, default='Mug')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=1)

    p.add_argument('--pretrained_model', type=str, required=True)

    opt = p.parse_args()
    return opt


def main(opt):
    args = {
        'pretrained_model': opt.pretrained_model,
        'device': opt.device,
    }

    if opt.device =='cuda':
        cuda_device = 0
        device = torch.device('cuda:' + str(cuda_device) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    num_grasps = 1000

    ## Dataset
    test_dataset = datasets.PointcloudAcronymAndSDFDataset(n_density=num_grasps, augmented_rotation=False)
    test_dataset.set_test_data()
    test_dataloader = DataLoader(
        test_dataset,
        num_workers=opt.num_workers,
        batch_size=opt.batch_size,
        shuffle=True, drop_last=False
    )

    ## Model
    model = loader.load_model(args)
    model.eval()
    sum_emd = 0
    cnt = 0
    for model_input, gt in tqdm.tqdm(test_dataloader):
        observation = model_input['visual_context']

        model.set_latent(observation.to(device), batch=num_grasps)
        generator = Grasp_AnnealedLD(model, batch=num_grasps, T=70, T_fit=50, k_steps=2, device=device)
        H = generator.sample()
        emd = earth_mover_distance(H, model_input["x_ene_pos"][0].to(device))
        sum_emd += emd
        cnt += 1
        print(emd, sum_emd/cnt)

if __name__ == '__main__':
    args = parse_args()
    main(args)
