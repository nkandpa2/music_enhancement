import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import pdb

import augmentation
import utils


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

    
def set_requires_grad(nets, requires_grad):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                 param.requires_grad = requires_grad
                    
                    
def train_epoch(global_step, model_params, models, optimizers, train_loaders, summary_writer):
    data_loader, reverb_loader, noise_loader = train_loaders
    eq_model = augmentation.MicrophoneEQ(rate=model_params.sample_rate).cuda()
    low_cut_filter = augmentation.LowCut(35, rate=model_params.sample_rate).cuda()
    melspec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=model_params.sample_rate,
                                                             n_fft=model_params.n_fft,
                                                             hop_length=model_params.hop_length,
                                                             n_mels=model_params.n_mels).cuda()
    
    generator, discriminator = models
    optim_generator, optim_discriminator = optimizers
    gan_criterion = gan_criterion = GANLoss(model_params.gan_loss).cuda()
    l1_criterion = nn.L1Loss()
    
    loss_meter_generator = utils.AverageMeter('G loss', fmt=':.4f')
    loss_meter_discriminator = utils.AverageMeter('D loss', fmt=':.4f')
    loss_meter_l1 = utils.AverageMeter('L1 loss', fmt=':.4f')

    loop = tqdm(data_loader)
    for i, x_waveform in enumerate(loop):
        batch_size = x_waveform.shape[0]
        x_waveform, noise, rir = x_waveform.cuda(), next(noise_loader).cuda(), next(reverb_loader).cuda()
        x_waveform, x_aug_waveform = augmentation.augment(x_waveform, rir=rir, noise=noise, eq_model=eq_model, low_cut_model=low_cut_filter)
        x_waveform, x_aug_waveform = x_waveform[:,:,:(x_waveform.shape[2]//256)*256], x_aug_waveform[:,:,:(x_aug_waveform.shape[2]//256)*256]
        x = melspec_transform(x_waveform).log2()
        x_aug = melspec_transform(x_aug_waveform).log2()
        
        # Generate mel spec
        x_hat = generator(x_aug)#[:,:,:,:x.shape[3]]
        
        # Compute discriminator loss and optimize D
        set_requires_grad(discriminator, True)
        x_fake = torch.cat((x_aug, x_hat), dim=1).detach()
        x_real = torch.cat((x_aug, x), dim=1)
        d_loss_fake = gan_criterion(discriminator(x_fake), False)
        d_loss_real = gan_criterion(discriminator(x_real), True)
        d_loss = 0.5*(d_loss_fake + d_loss_real)
        loss_meter_discriminator.update(d_loss.item(), n=batch_size)
        
        optim_discriminator.zero_grad()
        d_loss.backward(retain_graph=True)
        optim_discriminator.step()
        
        # Compute generator loss and optimize G
        set_requires_grad(discriminator, False)
        g_disc_loss = gan_criterion(discriminator(x_fake), True)
        g_l1_loss = l1_criterion(x_hat, x)
        g_loss = g_disc_loss + model_params.l1_coeff*g_l1_loss
        loss_meter_generator.update(g_disc_loss.item(), n=batch_size)
        loss_meter_l1.update(g_l1_loss.item(), n=batch_size)
        
        optim_generator.zero_grad()
        g_loss.backward()
        optim_generator.step()
        
        loop.set_postfix_str(f'{loss_meter_generator}, {loss_meter_discriminator}, {loss_meter_l1}')
        summary_writer.add_scalar('training/l1_loss', g_l1_loss.item(), global_step)
        summary_writer.add_scalar('training/generator_loss', g_disc_loss.item(), global_step)
        summary_writer.add_scalar('training/discriminator_loss', d_loss.item(), global_step)
        global_step += 1

    return global_step, (generator, discriminator), (optim_generator, optim_discriminator)
        
def validate(model_params, models, vocoder_model_params, vocoder_model, vocode_func, val_loaders):
    data_loader, reverb_loader, noise_loader = val_loaders
    eq_model = augmentation.MicrophoneEQ(rate=model_params.sample_rate).cuda()
    low_cut_filter = augmentation.LowCut(35, rate=model_params.sample_rate).cuda()
    melspec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=model_params.sample_rate,
                                                             n_fft=model_params.n_fft,
                                                             hop_length=model_params.hop_length,
                                                             n_mels=model_params.n_mels).cuda()
    
    generator, discriminator = models
    gan_criterion = gan_criterion = GANLoss('lsgan').cuda()
    l1_criterion = nn.L1Loss()
    
    loss_meter_generator = utils.AverageMeter('G loss', fmt=':.4f')
    loss_meter_discriminator = utils.AverageMeter('D loss', fmt=':.4f')
    loss_meter_l1 = utils.AverageMeter('L1 loss', fmt=':.4f')

    with torch.no_grad():
        for i, x_waveform in enumerate(data_loader):
            batch_size = x_waveform.shape[0]
            x_waveform, noise, rir = x_waveform.cuda(), next(noise_loader).cuda(), next(reverb_loader).cuda()
            x_waveform, x_aug_waveform = augmentation.augment(x_waveform, rir=rir, noise=noise, eq_model=eq_model, low_cut_model=low_cut_filter)
            x_waveform, x_aug_waveform = x_waveform[:,:,:(x_waveform.shape[2]//256)*256], x_aug_waveform[:,:,:(x_aug_waveform.shape[2]//256)*256]
            x = melspec_transform(x_waveform).log2()
            x_aug = melspec_transform(x_aug_waveform).log2()

            # Generate mel spec
            x_hat = generator(x_aug)#[:,:,:,:x.shape[3]]

            # Compute discriminator loss and optimize D
            x_fake = torch.cat((x_aug, x_hat), dim=1).detach()
            x_real = torch.cat((x_aug, x), dim=1)
            d_loss_fake = gan_criterion(discriminator(x_fake), False)
            d_loss_real = gan_criterion(discriminator(x_real), True)
            d_loss = 0.5*(d_loss_fake + d_loss_real)
            loss_meter_discriminator.update(d_loss.item(), n=batch_size)

            # Compute generator loss and optimize G
            set_requires_grad(discriminator, False)
            g_disc_loss = gan_criterion(discriminator(x_fake), True)
            g_l1_loss = l1_criterion(x_hat, x)
            g_loss = g_disc_loss + model_params.l1_coeff*g_l1_loss
            loss_meter_generator.update(g_disc_loss.item(), n=batch_size)
            loss_meter_l1.update(g_l1_loss.item(), n=batch_size)
           
    x_melspec, x_vocoded, x_griffin_lim = generate(model_params, models, vocoder_model_params, vocoder_model, vocode_func, x_aug)
        
    return x_waveform, x_aug_waveform, x_melspec, x_vocoded, x_griffin_lim, loss_meter_generator.avg, loss_meter_discriminator.avg, loss_meter_l1.avg


def generate(model_params, models, vocoder_model_params, vocoder_model, vocode_func, x_aug):
    generator, _ = models
    x_melspec = generator(x_aug)[:,:,:,:x_aug.shape[3]].detach()

    inverse_melscale_transform = torchaudio.transforms.InverseMelScale(n_stft=vocoder_model_params.n_fft//2 + 1,
                                                                       n_mels=vocoder_model_params.n_mels,
                                                                       sample_rate=vocoder_model_params.sample_rate,
                                                                       max_iter=1000).cuda()
    griffin_lim_transform = torchaudio.transforms.GriffinLim(n_fft=vocoder_model_params.n_fft,
                                                            hop_length=vocoder_model_params.hop_length,
                                                            n_iter=1000).cuda()
    x_griffin_lim = griffin_lim_transform(inverse_melscale_transform(2**x_melspec))
 
    with torch.no_grad():
        x_vocoded = vocode_func(vocoder_model_params, vocoder_model, x_melspec, torch.randn(x_melspec.shape[0], 1, x_melspec.shape[3]*256).cuda())

    return torch.exp(x_melspec), x_vocoded, x_griffin_lim
