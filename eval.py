import time
import torch
import subprocess
import os
from model import EAST
from detect import detect_dataset
import numpy as np
import shutil


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
	if os.path.exists(submit_path):
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(False).to(device)
	model.load_state_dict(torch.load(model_name))
	model.eval()
	
	start_time = time.time()
	detect_dataset(model, device, test_img_path, submit_path)
	# os.chdir(submit_path)
	# res = subprocess.getoutput('zip -q submit.zip *.txt')
	# res = subprocess.getoutput('mv submit.zip ../')
	# os.chdir('../')
	# res = subprocess.getoutput('python /data/maris/cv_EAST/evaluate/script.py –g=/data/maris/cv_EAST/evaluate/gt.zip –s=/data/maris/cv_EAST/submit.zip')
	# print(res)
	# os.remove('./submit.zip')
	print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)


if __name__ == '__main__': 
	# model_name = './pths/east_vgg16.pth'
	model_name = './pths/model_real_480epochs/model_epoch_480.pth'
	test_img_path = os.path.abspath('../data_EAST/test_img')
	submit_path = './submit'
	eval_model(model_name, test_img_path, submit_path)
