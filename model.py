import torch
from torch import nn

from models import cnnlstm
from models import custom_cnnlstm

def generate_model(opt, device):
	assert opt.model in [
		'cnnlstm',
		'custom_cnnlstm'
	]

	if opt.model == 'cnnlstm':
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)
	elif opt.model == 'custom_cnnlstm':
		model = custom_cnnlstm.CNNLSTM(num_classes=opt.n_classes)
	return model.to(device)