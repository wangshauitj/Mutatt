#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:wpaifang
# datetime:2019/9/18 15:52
# software: PyCharm
# function:
import json
# opt_content = json.load(open('/home/wangshuai/project/CM-Erase-REG/output/refcocog_umd/erase.json', 'r'))
# print(opt_content['val_accuracies'])
# opt_content = json.load(open('/home/wangshuai/project/CM-Erase-REG/output/refcocog_umd/mattnet.json', 'r'))
# print(opt_content['val_accuracies'])
# opt_content = json.load(open('/home/wangshuai/project/CM-Erase-REG/output/refcocog_umd/erase2.json', 'r'))
# print(opt_content['val_accuracies'])
# opt_content = json.load(open('/home/wangshuai/project/CM-Erase-REG/output/refcocog_umd/coco+_erase.json', 'r'))
# print(opt_content['val_accuracies'])
# opt_content = json.load(open('/home/wangshuai/project/CM-Erase-REG/output/refcocog_umd/our_method_no_rel.json', 'r'))
# print(opt_content['val_accuracies'])
opt_content = json.load(open('/home/lizhongguo/src/MAttNet/output/refcoco+_unc/out_method_seed_0.json', 'r'))
print(opt_content['val_accuracies'])
opt_content = json.load(open('/home/lizhongguo/src/MAttNet/output/refcoco+_unc/best.json', 'r'))
print(opt_content['val_accuracies'])

