from .denoising_loss import ProjectedSE3DenoisingLoss, SE3DenoisingLoss
from .sdf_loss import SDFLoss
from se3dif.grasp_diffusion_parametric_baseline.losses import NoisePredictionLoss


def get_losses(args):
    losses = args['Losses']

    loss_fns = {}
    if 'sdf_loss' in losses:
        loss_fns['loss/sdf'] = SDFLoss().loss_fn
    if 'projected_denoising_loss' in losses:
        loss_fns['loss/denoise'] = ProjectedSE3DenoisingLoss().loss_fn
    if 'denoising_loss' in losses:
        loss_fns['loss/denoise'] = SE3DenoisingLoss().loss_fn
    if 'noise_prediction_loss' in losses:
        loss_fns['loss/noise_prediction'] = NoisePredictionLoss(args).loss_fn

    loss_dict = LossDictionary(loss_dict=loss_fns)
    return loss_dict


class LossDictionary():
    def __init__(self, loss_dict):
        self.fields = loss_dict.keys()
        self.loss_dict = loss_dict

    def loss_fn(self, model, model_input, ground_truth, val=False):

        losses = {}
        infos = {}
        for field in self.fields:
            loss_fn_k = self.loss_dict[field]
            loss, info = loss_fn_k(model, model_input, ground_truth, val)
            losses = {**losses, **loss}
            infos = {**infos, **info}

        return losses, infos
