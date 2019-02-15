# -*- coding: utf-8 -*-
# @Time    : 18-7-23 7:49
# @Author  : Xinge

import torch
import logging
import pprint
logger = logging.getLogger('global')
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def check_keys(model, pretrained_state_dict):
    # with open('/mnt/lustre/pangjiangmiao/scripts/general/txt/model.txt', 'wt') as f:
    #     for k in model.state_dict().keys():
    #         f.writelines("{}\n".format(k))
    # with open('/mnt/lustre/pangjiangmiao/scripts/general/txt/ckpt.txt', 'wt') as f:
    #     for k in pretrained_state_dict.keys():
    #         f.writelines("{}\n".format(k))
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    pprint.pprint(model_keys)
    pprint.pprint(ckpt_keys)
    logger.info('missing keys:{}'.format(len(missing_keys)))
    logger.info('unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    pprint.pprint("Unused: {}".format(unused_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location = lambda storage, loc: storage.cuda(device))
    if 'state_dict' in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def restore_from(model, optimizer, ckpt_path):
    logger.info('restore from {}'.format(ckpt_path))
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path, map_location = lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']
    # best_recall = ckpt['best_recall']
    # arch = ckpt['arch']
    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch


def restore_from2(model, ckpt_path):
    logger.info('restore from {}'.format(ckpt_path))
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path, map_location = lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']
    # best_recall = ckpt['best_recall']
    # arch = ckpt['arch']
    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    # optimizer.load_state_dict(ckpt['optimizer'])
    return model, epoch
