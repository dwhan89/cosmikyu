import os

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from cosmikyu import config, model


class UNET(object):
    def __init__(self, identifier, shape, nin_channel, nout_channel, output_path=None, experiment_path=None,
                 cuda=False, ngpu=1):
        self.cuda = cuda
        self.ngpu = 0 if not self.cuda else ngpu
        self.shape = shape
        self.identifier = identifier
        self.nin_channel = nin_channel
        self.nout_channel = nout_channel

        self.output_path = output_path or os.path.join(config.default_output_dir)
        self.tracking_path = os.path.join(self.output_path, "mlruns")
        self.experiment_path = experiment_path or os.path.join(self.output_path, identifier)
        mlflow.set_tracking_uri(self.tracking_path)
        self.experiment = mlflow.get_experiment_by_name(identifier) or mlflow.get_experiment(
            mlflow.create_experiment(identifier))

        if torch.cuda.is_available() and not self.cuda:
            print("[WARNING] You have a CUDA device. You probably want to run with CUDA enabled")
        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.unet = None

        self.model_params = {"shape": shape, "sampler": "normal"}
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

    def load_states(self, output_path, postfix="", mlflow_run=None):
        saving_point_tracker_file = os.path.join(output_path, "saving_point.txt")
        if os.path.exists(saving_point_tracker_file) and not postfix:
            with open(saving_point_tracker_file, "r") as handle:
                postfix = handle.readline()
        unet_state_file = os.path.join(output_path, "unet{}.pt".format(postfix))
        try:
            print("loading saved states", postfix)
            if mlflow_run and False:
                self.unet = mlflow.pytorch.load_model(unet_state_file)
            else:
                self.unet.load_state_dict(torch.load(unet_state_file, map_location=self.device))
        except Exception:
            print("failed to load saved states")

    def save_states(self, output_path, postfix="", mlflow_run=None):
        postfix = "" if postfix == "" else "_{}".format(str(postfix))
        print("saving states", postfix)
        unet_state_file = os.path.join(output_path, "unet{}.pt".format(postfix))
        saving_point_tracker_file = os.path.join(output_path, "saving_point.txt")
        with open(saving_point_tracker_file, "w") as handle:
            handle.write(postfix)
        if mlflow_run and False:
            mlflow.pytorch.save_model(self.unet, unet_state_file)
        else:
            torch.save(self.unet.state_dict(), unet_state_file)

    def _get_optimizers(self, **kwargs):
        raise NotImplemented()

    def _eval_loss(self, gen_imgs, labels, **kwargs):
        raise NotImplemented()

    def _weights_init_normal(self, layer):
        classname = layer.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.constant_(layer.bias.data, 0.0)

    def train(self, dataloader, nepochs=200, sample_interval=1000,
              save_interval=10000, load_states=True, save_states=True, verbose=True, mlflow_run=None,
              **kwargs):
        kwargs.update({"nepochs": nepochs})
        kwargs.update(self.model_params)
        # Logging parameters
        if mlflow_run:
            for key, value in kwargs.items():
                mlflow.log_param(key, value)

        # Base Setup
        run_id = "trial" if not mlflow_run else mlflow_run.info.run_id
        run_path = os.path.join(self.experiment_path, run_id)
        artifacts_path = os.path.join(run_path, "artifacts")
        model_path = os.path.join(run_path, "model")
        self.unet.train()

        os.makedirs(artifacts_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        if load_states:
            self.load_states(model_path)

        self.save_states(model_path, 0)
        # Get Optimizers
        opt = self._get_optimizers(**kwargs)
        batches_done = 0
        for epoch in range(nepochs):
            for i, sample in enumerate(dataloader):
                imgs = sample[0]
                input_imgs = Variable(imgs[:, :self.nin_channel, ...].type(self.Tensor))
                labels = Variable(imgs[:, self.nin_channel:, ...].type(self.Tensor))
                opt.zero_grad()
                gen_imgs = self.unet(input_imgs)

                # Adversarial loss
                loss = self._eval_loss(gen_imgs, labels, **kwargs)
                mlflow.log_metric("loss", loss.item())
                loss.backward()
                opt.step()
                if verbose:
                    print("[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                          % (epoch, nepochs, batches_done % len(dataloader), len(dataloader), loss.item())
                          )
                if batches_done % sample_interval == 0 and False:
                    # temp = torch.cat((input_imgs.data[:1, ...], labels.data[:1, ...], gen_imgs[:1,...].data), 0)
                    # temp = torch.cat((temp[:1, ...].data, gen_imgs[:1,...].data), 0)
                    # temp = torch.cat((labels.data[:1, ...], (input_imgs.data[:1,0,...]-labels.data[:1,...]),
                    #    gen_imgs[:1,...].data,  (gen_imgs.data[:1,...]-input_imgs.data.data[:1,0,...])), 0) 
                    temp = torch.cat(((input_imgs.data[:1, 0, ...] - labels.data[:1, ...]),
                                      (input_imgs.data[:1, 0, ...] - gen_imgs.data.data[:1, ...])), 0)

                    save_image(temp, os.path.join(artifacts_path, "%d.png" % batches_done), normalize=True,
                               nrow=int(temp.shape[0]))
                batches_done += 1

            if int(epoch + 1) % save_interval == 0 and save_states:
                self.save_states(model_path, int(epoch + 1))
        if mlflow_run:
            mlflow.log_artifacts(artifacts_path)
        if save_states:
            self.save_states(model_path, nepochs)

    def generate_samples(self, input_imgs, concat=False, train=False):
        if input_imgs.ndim == 3: input_imgs = input_imgs[np.newaxis, ...]
        if not train:
            self.unet.eval()
        else:
            self.unet.train()
        input_imgs = Variable(self.Tensor(input_imgs[:, :self.nin_channel, ...]))
        ret = self.unet(input_imgs).detach()
        if concat:
            ret = torch.cat((input_imgs, ret), 1)
        return ret


class ResUNET(UNET):
    def __init__(self, identifier, shape, activation=[nn.Tanh()], nin_channel=3, nout_channel=3, nconv_layer=2,
                 nthresh_layer=1, ncov_fc=64, identity=False, sharpen=0, output_path=None, experiment_path=None,
                 cuda=False, ngpu=1, beta_sl1=0.0025):
        super().__init__(identifier, shape, nin_channel=nin_channel, nout_channel=nout_channel, output_path=output_path,
                         experiment_path=experiment_path,
                         cuda=cuda, ngpu=ngpu)
        self.beta_sl1 = beta_sl1
        if beta_sl1 <= 0:
            self.l1_loss_function = nn.L1Loss().to(device=self.device)
            self.l2_loss_function = nn.MSELoss().to(device=self.device)
        else:
            self.loss_function = nn.SmoothL1Loss(beta=beta_sl1).to(device=self.device)

        self.unet = model.ResUNET_Generator(shape, nconv_layer=nconv_layer, nconv_fc=ncov_fc, ngpu=ngpu,
                                            activation=activation, nin_channel=nin_channel, nout_channel=nout_channel,
                                            nthresh_layer=nthresh_layer, identity=identity).to(device=self.device)
        self.unet.apply(self._weights_init_normal)
        self.sharpen = sharpen
        self.sharpen_kernel = None
        if self.sharpen > 0:
            sharpen_kernel = np.empty((nout_channel, 1, 3, 3), dtype=np.float32)
            I = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            S = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            sharpen_kernel[:, :, ...] = I + self.sharpen * S
            self.sharpen_kernel = self.Tensor(sharpen_kernel)

        self.model_params["sharpen"] = sharpen
        self.model_params["beta_sl1"] = beta_sl1

    def _get_optimizers(self, **kwargs):
        lr, betas = kwargs['lr'], kwargs["betas"]
        return torch.optim.Adam(self.unet.parameters(), lr=lr, betas=betas)

    def _eval_loss(self, gen_imgs, labels, **kwargs):
        if self.sharpen > 0:
            gen_imgs = torch.nn.functional.conv2d(gen_imgs, weight=self.sharpen_kernel, stride=1, padding=1,
                                                  groups=self.nout_channel)
            labels = torch.nn.functional.conv2d(labels, weight=self.sharpen_kernel, stride=1, padding=1,
                                                groups=self.nout_channel)

        if self.beta_sl1 <= 0:
            assert ("lambda_l1" in kwargs or "lambda_l2" in kwargs)
            loss = 0
            if kwargs["lambda_l1"] > 0:
                loss += self.l1_loss_function(gen_imgs, labels) * kwargs["lambda_l1"]
            if kwargs["lambda_l2"] > 0:
                loss += self.l2_loss_function(gen_imgs, labels) * kwargs["lambda_l2"]
        else:
            loss = self.loss_function(gen_imgs, labels)
        return loss

    def train(self, dataloader, nepochs=200, sample_interval=1000,
              save_interval=10000, load_states=True, save_states=True, verbose=True, mlflow_run=None,
              lambda_l1=1, lambda_l2=1, **kwargs):
        super().train(dataloader=dataloader, nepochs=nepochs, sample_interval=sample_interval,
                      save_interval=save_interval, load_states=load_states, save_states=save_states, verbose=verbose,
                      mlflow_run=mlflow_run,
                      lambda_l1=lambda_l1, lambda_l2=lambda_l2, **kwargs)
