import json
import torch

from .tinynas.nn.networks import ProxylessNASNets
from .utils import download_url

__all__ = ['net_id_list', 'build_model', 'download_tflite']

""" Note: all the memory and latency profiling is done with TinyEngine """
NET_INFO = {
    ##### imagenet models ######
    # mcunet models
    'mcunet-10fps': {
        'net_name': 'mcunet-10fps_imagenet',
        'description': 'MCUNet model that runs 10fps on STM32F746 (ImageNet)'
    },
    'mcunet-5fps': {
        'net_name': 'mcunet-5fps_imagenet',
        'description': 'MCUNet model that runs 5fps on STM32F746 (ImageNet)'
    },
    'mcunet-256kB': {
        'net_name': 'mcunet-256kb-1mb_imagenet',
        'description': 'MCUNet model that fits 256KB SRAM and 1MB Flash (ImageNet)',
    },
    'mcunet-320kB': {
        'net_name': 'mcunet-320kb-1mb_imagenet',
        'description': 'MCUNet model that fits 320KB SRAM and 1MB Flash (ImageNet)',
    },
    'mcunet-512kB': {
        'net_name': 'mcunet-512kb-2mb_imagenet',
        'description': 'MCUNet model that fits 512KB SRAM and 2MB Flash (ImageNet)',
    },
    # baseline models
    'mbv2-320kB': {
        'net_name': 'mbv2-w0.35-r144_imagenet',
        'description': 'scaled MobileNetV2 that fits 320KB SRAM and 1MB Flash (ImageNet)',
    },
    'proxyless-320kB': {
        'net_name': 'proxyless-w0.3-r176_imagenet',
        'description': 'scaled ProxylessNet that fits 320KB SRAM and 1MB Flash (ImageNet)'
    },

    ##### vww models ######
    'mcunet-10fps-vww': {
        'net_name': 'mcunet-10fps_vww',
        'description': 'MCUNet model that runs 10fps on STM32F746 (VWW)'
    },
    'mcunet-5fps-vww': {
        'net_name': 'mcunet-5fps_vww',
        'description': 'MCUNet model that runs 5fps on STM32F746 (VWW)'
    },
    'mcunet-320kB-vww': {
        'net_name': 'mcunet-320kb-1mb_vww',
        'description': 'MCUNet model that fits 320KB SRAM and 1MB Flash (VWW)'
    },

    ##### detection demo model ######
    'person-det': {
        'net_name': 'person-det',
        'description': 'person detection model used in our demo'
    },
}

net_id_list = list(NET_INFO.keys())

url_base = "https://hanlab.mit.edu/projects/tinyml/mcunet/release/"


def build_model(net_id, pretrained=True):
    assert net_id in NET_INFO, 'Invalid net_id! Select one from {})'.format(list(NET_INFO.keys()))
    net_info = NET_INFO[net_id]

    #net_config_url = url_base + net_info['net_name'] + ".json"
    #sd_url = url_base + net_info['net_name'] + ".pth"

    #net_config = json.load(open(download_url(net_config_url)))
    net_config = json.load(open("/home/ubuntu/Workspace/toandev/Pretrain_od/nanodet-main/nanodet/model/backbone/mcunet-10fps_imagenet.json"))
    resolution = net_config['resolution']
    model = ProxylessNASNets.build_from_config(net_config)

    if pretrained:
        sd = torch.load("/home/ubuntu/Workspace/toandev/Pretrain_od/nanodet-main/mcunet-10fps_imagenet.pth", map_location='cpu')
        model.load_state_dict(sd['state_dict'],strict=False)
    return model


def download_tflite(net_id):
    assert net_id in NET_INFO, 'Invalid net_id! Select one from {})'.format(list(NET_INFO.keys()))
    net_info = NET_INFO[net_id]
    tflite_url = url_base + net_info['net_name'] + ".tflite"
    return download_url(tflite_url)  # the file path of the downloaded tflite model
