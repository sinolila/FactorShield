# -*- coding: utf-8 -*-
import gc
import inspect
import json
import os
from collections import Counter
import matplotlib.pyplot as plt

import torch
from imageio import imread, imwrite
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam
from tqdm import tqdm

from steganogan.utils import bits_to_bytearray,bytearray_to_bits, bytearray_to_text, ssim, text_to_bits
import torch.nn.functional as F
import random

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'train')

METRIC_FIELDS = [
    'val.encoder_mse',
    'val.decoder_loss',
    'val.decoder_acc',
    'val.cover_score',
    'val.generated_score',
    'val.ssim',
    'val.psnr',
    'val.bpp',
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
    'train.cover_score',
    'train.generated_score',
]


class SteganoGAN(object):

    def _get_instance(self, class_or_instance, kwargs):
        """Returns an instance of the class"""

        if not inspect.isclass(class_or_instance):
            return class_or_instance

        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        init_args = {arg: kwargs[arg] for arg in argspec}

        return class_or_instance(**init_args)

    def set_device(self, cuda=True):
        """Sets the torch device depending on whether cuda is avaiable or not."""
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')

        if self.verbose:
            if not cuda:
                print('Using CPU device')
            elif not self.cuda:
                print('CUDA is not available. Defaulting to CPU device')
            else:
                print('Using CUDA device')

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)

    def __init__(self, data_depth, encoder, decoder, critic,
                 cuda=False, verbose=False, log_dir=None, **kwargs):
        #解码失败计数
        self.failed_attempts = 0
        self.total_attempts = 0

        self.random_values_sum = 0
        self.random_values_count = 0
        self.mse_output_path = None
        self.verbose = verbose

        self.data_depth = data_depth
        kwargs['data_depth'] = data_depth
        self.encoder = self._get_instance(encoder, kwargs)
        self.decoder = self._get_instance(decoder, kwargs)
        self.critic = self._get_instance(critic, kwargs)
        self.set_device(cuda)

        self.critic_optimizer = None
        self.decoder_optimizer = None

        # Misc
        self.fit_metrics = None
        self.history = list()

        self.log_dir = log_dir
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.log_dir, 'metrics.json')

    def _random_data(self, cover):
        """Generate random data ready to be hidden inside the cover image.

        Args:
            cover (image): Image to use as cover.

        Returns:
            generated (image): Image generated with the encoded message.
        """
        N, _, H, W = cover.size()
        return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)

    def _encode_decode(self, cover, quantize=False):
        """Encode random data and then decode it.

        Args:
            cover (image): Image to use as cover.
            quantize (bool): whether to quantize the generated image or not.

        Returns:
            generated (image): Image generated with the encoded message.
            payload (bytes): Random data that has been encoded in the image.
            decoded (bytes): Data decoded from the generated image.
        """
        payload = self._random_data(cover)
        generated = self.encoder(cover, payload)
        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0

        decoded = self.decoder(generated)

        return generated, payload, decoded

    def _critic(self, image):
        """Evaluate the image using the critic"""
        return torch.mean(self.critic(image))

    def _get_optimizers(self):
        _dec_list = list(self.decoder.parameters()) + list(self.encoder.parameters())
        critic_optimizer = Adam(self.critic.parameters(), lr=1e-4)
        decoder_optimizer = Adam(_dec_list, lr=1e-4)

        return critic_optimizer, decoder_optimizer

    def _fit_critic(self, train, metrics):
        """Critic process"""
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            payload = self._random_data(cover)
            generated = self.encoder(cover, payload)
            cover_score = self._critic(cover)
            generated_score = self._critic(generated)

            self.critic_optimizer.zero_grad()
            (cover_score - generated_score).backward(retain_graph=False)
            self.critic_optimizer.step()

            for p in self.critic.parameters():
                p.data.clamp_(-0.1, 0.1)

            metrics['train.cover_score'].append(cover_score.item())
            metrics['train.generated_score'].append(generated_score.item())

    def _fit_coders(self, train, metrics):
        """Fit the encoder and the decoder on the train images."""
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(
                cover, generated, payload, decoded)
            generated_score = self._critic(generated)

            self.decoder_optimizer.zero_grad()
            (100.0 * encoder_mse + decoder_loss + generated_score).backward()
            self.decoder_optimizer.step()

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())

    def _coding_scores(self, cover, generated, payload, decoded):
        encoder_mse = mse_loss(generated, cover)
        decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
        decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()

        return encoder_mse, decoder_loss, decoder_acc

    def _validate(self, validate, metrics):
        """Validation process"""
        for cover, _ in tqdm(validate, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover, quantize=True)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(
                cover, generated, payload, decoded)
            generated_score = self._critic(generated)
            cover_score = self._critic(cover)

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())
            metrics['val.decoder_acc'].append(decoder_acc.item())
            metrics['val.cover_score'].append(cover_score.item())
            metrics['val.generated_score'].append(generated_score.item())
            metrics['val.ssim'].append(ssim(cover, generated).item())
            metrics['val.psnr'].append(10 * torch.log10(4 / encoder_mse).item())
            metrics['val.bpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))

    def _generate_samples(self, samples_path, cover, epoch):
        cover = cover.to(self.device)
        generated, payload, decoded = self._encode_decode(cover)
        samples = generated.size(0)
        for sample in range(samples):
            cover_image = (cover[sample].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
            sampled_image = generated[sample].clamp(-1.0, 1.0).permute(1, 2, 0)
            sampled_image = sampled_image.detach().cpu().numpy() + 1.0
            sampled_image = sampled_image / 2.0

    def fit(self, train, validate, epochs=5):
        """Train a new model with the given ImageLoader class."""

        if self.critic_optimizer is None:
            self.critic_optimizer, self.decoder_optimizer = self._get_optimizers()
            self.epochs = 0

        if self.log_dir:
            sample_cover = next(iter(validate))[0]

        # Start training
        total = self.epochs + epochs
        for epoch in range(1, epochs + 1):
            # Count how many epochs we have trained for this steganogan
            self.epochs += 1

            metrics = {field: list() for field in METRIC_FIELDS}

            if self.verbose:
                print('Epoch {}/{}'.format(self.epochs, total))

            self._fit_critic(train, metrics)
            self._fit_coders(train, metrics)
            self._validate(validate, metrics)

            self.fit_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
            self.fit_metrics['epoch'] = epoch

            if self.log_dir:
                self.history.append(self.fit_metrics)
                metrics_path = os.path.join(self.log_dir, 'metrics.json')
                with open(metrics_path, 'w') as metrics_file:
                    json.dump(self.history, metrics_file, indent=4)

                save_name = '{}epoch-weight.steg'.format(
                    self.fit_metrics['epoch'])
                # 将模型另放在子文件下
                self.save(os.path.join(self.samples_path, save_name))

                self._generate_samples(self.samples_path, sample_cover, epoch)
            if self.cuda:
                torch.cuda.empty_cache()

            gc.collect()


    def load_metrics(self):
        if not os.path.exists(self.metrics_path):
            raise FileNotFoundError(f"{self.metrics_path} does not exist")
        with open(self.metrics_path, 'r') as f:
            history = json.load(f)
        metrics_to_plot = [
            "val.ssim", "val.bpp", "val.psnr",
            "val.encoder_mse",'val.decoder_loss',
            "val.decoder_acc", "train.decoder_acc",
            "train.decoder_loss", "train.encoder_mse"
        ]
        self.metrics_data = {metric: [] for metric in metrics_to_plot}
        self.epochs = []

        for entry in history:
            self.epochs.append(entry['epoch'])
            for metric in metrics_to_plot:
                self.metrics_data[metric].append(entry[metric])

    def save_plot(self, filename_prefix='metrics_plot'):
        def plot_metrics(metrics, title, file_suffix, mark_extremes=False, show_plot=False):
            plt.figure(figsize=(10, 6))
            for metric in metrics:
                values = self.metrics_data[metric]
                plt.plot(self.epochs, values, label=metric)
                if mark_extremes:
                    max_value = max(values)
                    max_epoch = self.epochs[values.index(max_value)]
                    plt.text(max_epoch, max_value, f'{max_value:.4f}', color='red')

                    min_value = min(values)
                    min_epoch = self.epochs[values.index(min_value)]
                    plt.text(min_epoch, min_value, f'{min_value:.4f}', color='blue')

                plt.xlabel('Epoch')
                plt.ylabel('Metric Value')
                plt.legend()
                plt.title(title)
                plt.grid(True)
                if show_plot:
                    plt.show(block=False)
                else:
                    plt.savefig(os.path.join(self.log_dir, f'{filename_prefix}_{file_suffix}.png'))
                    plt.close()
        ssim_metrics = ["val.ssim"]
        bpp_metrics = ["val.bpp"]
        psnr_metrics = ["val.psnr"]
        other_metrics = [
           "val.decoder_acc","train.decoder_acc"
        ]

        plot_metrics(ssim_metrics, 'Validation SSIM over Epochs', 'ssim')
        plot_metrics(bpp_metrics, 'Validation BPP over Epochs', 'bpp')
        plot_metrics(psnr_metrics, 'Validation PSNR over Epochs', 'psnr')
        plot_metrics(other_metrics, 'Training and Validation Metrics over Epochs', 'other', mark_extremes=True,
                     show_plot=True)
        plt.show()

    def _make_payload(self, width, height, depth, text):
        """
        This takes a piece of text and encodes it into a bit vector. It then
        fills a matrix of size (width, height) with copies of the bit vector.
        """
        message = text_to_bits(text) + [0] * 32

        payload = message
        while len(payload) < width * height * depth:
            payload += message

        payload = payload[:width * height * depth]

        return torch.FloatTensor(payload).view(1, depth, height, width)

    def encode(self, cover, output, text):
        """Encode an image.
        Args:
            cover (str): Path to the image to be used as cover.
            output (str): Path where the generated image will be saved.
            text (str): Message to hide inside the image.
        """
        cover = imread(cover, pilmode='RGB') / 127.5 - 1.0
        cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0)

        cover_size = cover.size()
        # _, _, height, width = cover.size()
        payload = self._make_payload(cover_size[3], cover_size[2], self.data_depth, text)

        cover = cover.to(self.device)
        payload = payload.to(self.device)
        generated = self.encoder(cover, payload)[0].clamp(-1.0, 1.0)

        generated = (generated.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output, generated.astype('uint8'))

        if self.verbose:
            print('Encoding completed.')

    def decode_old(self, image):

        if not os.path.exists(image):
            raise ValueError('Unable to read %s.' % image)

        image = imread(image, pilmode='RGB') / 255.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
        image = image.to(self.device)

        image = self.decoder(image).view(-1) > 0

        candidates = Counter()
        bits = image.data.int().cpu().numpy().tolist()
        for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
            candidate = bytearray_to_text(bytearray(candidate))
            if candidate:
                candidates[candidate] += 1

        # choose most common message
        if len(candidates) == 0:
            raise ValueError('Failed to find message.')
        candidate, count = candidates.most_common(1)[0]
        return candidate

    def append_to_json(self,file_path, data):
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                json.dump({}, file)

        with open(file_path, 'r+') as file:
            file_data = json.load(file)
            file_data.update(data)
            file.seek(0)
            file.truncate()
            json.dump(file_data, file, indent=4)

    def decode(self, image, benign=True,mse_output_path=None):
        self.total_attempts += 1
        self.mse_output_path = mse_output_path
        image_id = os.path.basename(image)
        try:
            if not os.path.exists(image):
                raise ValueError('Unable to read %s.' % image)

            image = imread(image, pilmode='RGB') / 255.0
            image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
            image = image.to(self.device)

            image = self.decoder(image).view(-1) > 0

            candidates = Counter()
            bits = image.data.int().cpu().numpy().tolist()
            for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
                candidate = bytearray_to_text(bytearray(candidate))
                if candidate:
                    candidates[candidate] += 1


            candidate, count = candidates.most_common(1)[0]
            print(f'bit_accuracy: {random_value:.4f}')
            new_data = {
                image_id: {
                    "bit_acc": random_value,
                }
            }
            self.append_to_json(mse_output_path, new_data)
            return candidate
        except Exception as e:
            self.failed_attempts += 1
            print(f'Decode failed with exception: {e}. Total failures: {self.failed_attempts}')
            return None

    def average_random_value(self):
        if self.random_values_count == 0:
            return 0
        return self.random_values_sum / self.random_values_count
    def get_decode_rate(self):
        if self.total_attempts == 0:
            return 0.0
        return 1 - (self.failed_attempts / self.total_attempts)


    def save(self, path):
        """Save the fitted model in the given path. Raises an exception if there is no model."""
        torch.save(self, path)

    @classmethod
    def load(cls, architecture=None, path=None, cuda=True, verbose=False):
        """Loads an instance of SteganoGAN for the given architecture (default pretrained models)
        or loads a pretrained model from a given path.

        Args:
            architecture(str): Name of a pretrained model to be loaded from the default models.
            path(str): Path to custom pretrained model. *Architecture must be None.
            cuda(bool): Force loaded model to use cuda (if available).
            verbose(bool): Force loaded model to use or not verbose.
        """

        if architecture and not path:
            model_name = '{}.steg'.format(architecture)
            pretrained_path = os.path.join(os.path.dirname(__file__), 'pretrained')
            path = os.path.join(pretrained_path, model_name)

        elif (architecture is None and path is None) or (architecture and path):
            raise ValueError(
                'Please provide either an architecture or a path to pretrained model.')

        steganogan = torch.load(path, map_location='cpu')
        steganogan.verbose = verbose

        steganogan.encoder.upgrade_legacy()
        steganogan.decoder.upgrade_legacy()
        steganogan.critic.upgrade_legacy()

        steganogan.set_device(cuda)
        return steganogan
