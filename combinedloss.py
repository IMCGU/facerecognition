import mxnet as mx
import numpy as np
import symbol_utils
import sklearn    
import os
import sys
import math
import random
import logging
import pickle    
from mxnet import ndarray as nd
import argparse    
    
    
    
def Intraloss(args,_weight,embedding,gt_label):    
    s = args.margin_s
    m = args.margin_m
    assert s>0.0
    assert args.margin_b>0.0
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes, name='fc7')
    zy = mx.sym.pick(fc7, gt_label, axis=1)##fc7中每一行找出gt_label的值，就是s*cos_t，
    cos_t = zy/s ##就是cos thelta
    t = mx.sym.arccos(cos_t)##做arc得到angle
    intra_loss = t/np.pi
    intra_loss = mx.sym.mean(intra_loss)
    #intra_loss = mx.sym.exp(cos_t*-1.0)
    intra_loss = mx.sym.MakeLoss(intra_loss, name='intra_loss', grad_scale = args.margin_b)
    print('intra_loss:',intra_loss)
    if m>0.0:
      t = t+m
      body = mx.sym.cos(t)
      new_zy = body*s
      diff = new_zy - zy
      diff = mx.sym.expand_dims(diff, 1)
      gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes, on_value = 1.0, off_value = 0.0)
      body = mx.sym.broadcast_mul(gt_one_hot, diff)
      fc7 = fc7+body
    return (intra_loss,fc7)

    
def Interloss(args,_weight,embedding,gt_label): 
    s = args.margin_s
    m = args.margin_m
    assert s>0.0
    assert args.margin_b>0.0
    assert args.margin_a>0.0
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes, name='fc7')
    zy = mx.sym.pick(fc7, gt_label, axis=1)
    cos_t = zy/s
    t = mx.sym.arccos(cos_t)
    counter_weight = mx.sym.take(_weight, gt_label, axis=0)
    counter_cos = mx.sym.dot(counter_weight, _weight, transpose_b=True)
    inter_loss = counter_cos
    inter_loss = mx.sym.mean(inter_loss)
    inter_loss = mx.sym.MakeLoss(inter_loss, name='inter_loss', grad_scale = args.margin_b)
    print('interloss:',inter_loss)
    if m>0.0:
      t = t+m
      body = mx.sym.cos(t)
      new_zy = body*s
      diff = new_zy - zy
      diff = mx.sym.expand_dims(diff, 1)
      gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes, on_value = 1.0, off_value = 0.0)
      body = mx.sym.broadcast_mul(gt_one_hot, diff)
      fc7 = fc7+body
    return (interloss,fc7) 
def arcface(args,_weight,embedding,gt_label):
    s = args.margin_s
    m = args.margin_m
    assert s>0.0
    assert m>=0.0
    assert m<(math.pi/2)
    ##start to compute the cos(theta)
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s 
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes, name='fc7')
    zy = mx.sym.pick(fc7, gt_label, axis=1)###fc7每一行找出gt_label對應的值, 即s*cos_t
    cos_t = zy/s##網路輸出output = s*x/|x|*w/|w|*cos(theta), 這裡將輸出除以s，得到實際的cos值，即cos（theta)
    ##end to compute the cos(theta)
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = math.sin(math.pi-m)*m###數學上sin(pi-m)*m = sin(m) * m ，使cos（θ+ m）函數單調遞減，而θ在[0°，180°]中。 但是實際上沒有必要，因為theta不會超過90°。
    ##threshold = 0.0
    threshold = math.cos(math.pi-m)##when theta > threshold, ' cos(theta+m)' will be changed to zy_keep , namely 'cos(theta)-msin(m)'. The min of cos(theta+m) is -1 . To make the function monotonic decreasing while theta in [0°,180°] , we need to make sure ' cos(theta)-msin(m) < = -1' when ' theta > pi - m ' ．when m=0.5,'cos(theta)-m*sin(m) < = -1' .
    if args.easy_margin:##'easy_margin' means we just add margin on the subset of "WX>0". It works but the defined formula is not monotone decreasing anymore. 'relu' here can be thought of 'if-else' condition.
      cond = mx.symbol.Activation(data=cos_t, act_type='relu')
    else:
      cond_v = cos_t - threshold
      cond = mx.symbol.Activation(data=cond_v, act_type='relu')
    ##compute cos(theta + m)  
    body = cos_t*cos_t
    body = 1.0-body
    sin_t = mx.sym.sqrt(body)
    new_zy = cos_t*cos_m
    b = sin_t*sin_m
    new_zy = new_zy - b
    new_zy = new_zy*s
    if args.easy_margin:
      zy_keep = zy
    else:
      zy_keep = zy - s*mm
    new_zy = mx.sym.where(cond, new_zy, zy_keep)

    diff = new_zy - zy
    diff = mx.sym.expand_dims(diff, 1)
    gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes, on_value = 1.0, off_value = 0.0)
    body = mx.sym.broadcast_mul(gt_one_hot, diff)
    fc7 = fc7+body
    return fc7