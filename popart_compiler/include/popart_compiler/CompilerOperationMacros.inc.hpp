// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// Auto generated file, do not modify
// Run `python3 PopParse.py to regenerate
// clang-format off

// Ops from AiGraphcoreOpset1
OP_DECL(popart, groupnormalization, groupnormalization, AiGraphcoreOpset1.groupnormalization, ARG(INT,num_groups) ARG(FLOAT,epsilon) , BODY_ARG(num_groups) BODY_ARG(epsilon) )
OP_DECL(popart, subsample, subsample, AiGraphcoreOpset1.subsample, ARG(INT_VEC,strides) , BODY_ARG(strides) )
OP_DECL(popart, printtensor, printtensor, AiGraphcoreOpset1.printtensor, ARG(INT,print_gradient) ARG(STRING,title) , BODY_ARG(print_gradient) BODY_ARG(title) )
OP_DECL(popart, nop, nop, AiGraphcoreOpset1.nop, NONE, NONE)
OP_DECL(popart, scale, scale, AiGraphcoreOpset1.scale, ARG(FLOAT,scale) , BODY_ARG(scale) )
OP_DECL(popart, scaledadd, scaledadd, AiGraphcoreOpset1.scaledadd, ARG(FLOAT,scale0) ARG(FLOAT,scale1) , BODY_ARG(scale0) BODY_ARG(scale1) )
OP_DECL(popart, lstm, lstm, AiGraphcoreOpset1.lstm, ARG(INT,outputFullSequence) , BODY_ARG(outputFullSequence) )
OP_DECL(popart, gelu, gelu, AiGraphcoreOpset1.gelu, NONE, NONE)
OP_DECL(popart, detach, detach, AiGraphcoreOpset1.detach, NONE, NONE)
OP_DECL(popart, round, round, AiGraphcoreOpset1.round, NONE, NONE)
OP_DECL(popart, dynamicslice, dynamicslice, AiGraphcoreOpset1.dynamicslice, ARG(INT_VEC,axes) ARG(INT_VEC,sizes) ARG(INT,noOverlap) , BODY_ARG(axes) BODY_ARG(sizes) BODY_ARG(noOverlap) )
OP_DECL(popart, dynamicupdate, dynamicupdate, AiGraphcoreOpset1.dynamicupdate, ARG(INT_VEC,axes) ARG(INT_VEC,sizes) ARG(INT,noOverlap) , BODY_ARG(axes) BODY_ARG(sizes) BODY_ARG(noOverlap) )
OP_DECL(popart, dynamiczero, dynamiczero, AiGraphcoreOpset1.dynamiczero, ARG(INT_VEC,axes) ARG(INT_VEC,sizes) , BODY_ARG(axes) BODY_ARG(sizes) )
OP_DECL(popart, dynamicadd, dynamicadd, AiGraphcoreOpset1.dynamicadd, ARG(INT_VEC,axes) ARG(INT_VEC,sizes) , BODY_ARG(axes) BODY_ARG(sizes) )
OP_DECL(popart, replicatedallreduce, replicatedallreduce, AiGraphcoreOpset1.replicatedallreduce, NONE, NONE)
OP_DECL(popart, l1loss, l1loss, AiGraphcoreOpset1.l1loss, ARG(FLOAT,lambda) ARG(INT,reduction) , BODY_ARG(lambda) BODY_ARG(static_cast<popart::ReductionType>(reduction)) )
OP_DECL(popart, nllloss, nllloss, AiGraphcoreOpset1.nllloss, ARG(INT,reduction) ARG(INT,ignoreIndex) ARG(INT,inputIsLogProbability) , BODY_ARG(static_cast<popart::ReductionType>(reduction)) BODY_ARG(ignoreIndex) BODY_ARG(inputIsLogProbability) )
OP_DECL(popart, identityloss, identityloss, AiGraphcoreOpset1.identityloss, ARG(INT,reduction) , BODY_ARG(static_cast<popart::ReductionType>(reduction)) )
OP_DECL(popart, shapeddropout, shapeddropout, AiGraphcoreOpset1.shapeddropout, ARG(INT_VEC,shape) ARG(FLOAT,ratio) , BODY_ARG(shape) BODY_ARG(ratio) )
OP_DECL(popart, atan2, atan2, AiGraphcoreOpset1.atan2, NONE, NONE)
OP_DECL(popart, expm1, expm1, AiGraphcoreOpset1.expm1, NONE, NONE)
OP_DECL(popart, log1p, log1p, AiGraphcoreOpset1.log1p, NONE, NONE)
// Ops from AiOnnxOpset10
OP_DECL(popart, averagepool, averagepool, AiOnnxOpset10.averagepool, ARG(INT_VEC,kernel_shape) ARG(INT,ceil_mode) ARG(INT,count_include_pad) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(kernel_shape) BODY_ARG(ceil_mode) BODY_ARG(count_include_pad) BODY_ARG(pads) BODY_ARG(strides) )
OP_DECL(popart, convinteger, convinteger, AiOnnxOpset10.convinteger, ARG(INT_VEC,dilations) ARG(INT,group) ARG(INT_VEC,kernel_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(dilations) BODY_ARG(group) BODY_ARG(kernel_shape) BODY_ARG(pads) BODY_ARG(strides) )
OP_DECL(popart, dequantizelinear, dequantizelinear, AiOnnxOpset10.dequantizelinear, NONE, NONE)
OP_DECL(popart, dropout, dropout, AiOnnxOpset10.dropout, ARG(INT,num_outputs) ARG(FLOAT,ratio) , BODY_ARG(num_outputs) BODY_ARG(ratio) )
OP_DECL(popart, isinf, isinf, AiOnnxOpset10.isinf, ARG(INT,detect_negative) ARG(INT,detect_positive) , BODY_ARG(detect_negative) BODY_ARG(detect_positive) )
OP_DECL(popart, matmulinteger, matmulinteger, AiOnnxOpset10.matmulinteger, NONE, NONE)
OP_DECL(popart, maxpool, maxpool, AiOnnxOpset10.maxpool, ARG(INT,num_outputs) ARG(INT_VEC,kernel_shape) ARG(INT,ceil_mode) ARG(INT_VEC,dilations) ARG(INT_VEC,pads) ARG(INT,storage_order) ARG(INT_VEC,strides) , BODY_ARG(num_outputs) BODY_ARG(kernel_shape) BODY_ARG(ceil_mode) BODY_ARG(dilations) BODY_ARG(pads) BODY_ARG(storage_order) BODY_ARG(strides) )
OP_DECL(popart, mod, mod, AiOnnxOpset10.mod, ARG(INT,fmod) , BODY_ARG(fmod) )
OP_DECL(popart, nonmaxsuppression, nonmaxsuppression, AiOnnxOpset10.nonmaxsuppression, ARG(INT,center_point_box) , BODY_ARG(center_point_box) )
OP_DECL(popart, qlinearconv, qlinearconv, AiOnnxOpset10.qlinearconv, ARG(INT_VEC,dilations) ARG(INT,group) ARG(INT_VEC,kernel_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(dilations) BODY_ARG(group) BODY_ARG(kernel_shape) BODY_ARG(pads) BODY_ARG(strides) )
OP_DECL(popart, qlinearmatmul, qlinearmatmul, AiOnnxOpset10.qlinearmatmul, NONE, NONE)
OP_DECL(popart, quantizelinear, quantizelinear, AiOnnxOpset10.quantizelinear, NONE, NONE)
OP_DECL(popart, resize, resize, AiOnnxOpset10.resize, ARG(STRING,mode) , BODY_ARG(mode) )
OP_DECL(popart, reversesequence, reversesequence, AiOnnxOpset10.reversesequence, ARG(INT,batch_axis) ARG(INT,time_axis) , BODY_ARG(batch_axis) BODY_ARG(time_axis) )
OP_DECL(popart, roialign, roialign, AiOnnxOpset10.roialign, ARG(STRING,mode) ARG(INT,output_height) ARG(INT,output_width) ARG(INT,sampling_ratio) ARG(FLOAT,spatial_scale) , BODY_ARG(mode) BODY_ARG(output_height) BODY_ARG(output_width) BODY_ARG(sampling_ratio) BODY_ARG(spatial_scale) )
OP_DECL(popart, slice, slice, AiOnnxOpset10.slice, NONE, NONE)
OP_DECL(popart, thresholdedrelu, thresholdedrelu, AiOnnxOpset10.thresholdedrelu, ARG(FLOAT,alpha) , BODY_ARG(alpha) )
OP_DECL(popart, topk, topk, AiOnnxOpset10.topk, ARG(INT,axis) , BODY_ARG(axis) )
OP_DECL(popart, upsample, upsample, AiOnnxOpset10.upsample, ARG(STRING,mode) , BODY_ARG(mode) )
// Ops from AiOnnxOpset9
OP_DECL(popart, acosh, acosh, AiOnnxOpset10.acosh, NONE, NONE)
OP_DECL(popart, asinh, asinh, AiOnnxOpset10.asinh, NONE, NONE)
OP_DECL(popart, atanh, atanh, AiOnnxOpset10.atanh, NONE, NONE)
OP_DECL(popart, batchnormalization, batchnormalization, AiOnnxOpset10.batchnormalization, ARG(INT,num_outputs) ARG(FLOAT,epsilon) ARG(FLOAT,momentum) , BODY_ARG(num_outputs) BODY_ARG(epsilon) BODY_ARG(momentum) )
OP_DECL(popart, cast, cast, AiOnnxOpset10.cast, ARG(STRING,to) , BODY_ARG(to) )
OP_DECL(popart, compress, compress, AiOnnxOpset10.compress, ARG(INT,axis) , BODY_ARG(axis) )
OP_DECL(popart, cosh, cosh, AiOnnxOpset10.cosh, NONE, NONE)
OP_DECL(popart, erf, erf, AiOnnxOpset10.erf, NONE, NONE)
OP_DECL(popart, eyelike, eyelike, AiOnnxOpset10.eyelike, ARG(INT,dtype) ARG(INT,k) , BODY_ARG(dtype) BODY_ARG(k) )
OP_DECL(popart, flatten, flatten, AiOnnxOpset10.flatten, ARG(INT,axis) , BODY_ARG(axis) )
OP_DECL(popart, gemm, gemm, AiOnnxOpset10.gemm, ARG(FLOAT,alpha) ARG(FLOAT,beta) ARG(INT,transA) ARG(INT,transB) , BODY_ARG(alpha) BODY_ARG(beta) BODY_ARG(transA) BODY_ARG(transB) )
OP_DECL(popart, greater, greater, AiOnnxOpset10.greater, NONE, NONE)
OP_DECL(popart, isnan, isnan, AiOnnxOpset10.isnan, NONE, NONE)
OP_DECL(popart, less, less, AiOnnxOpset10.less, NONE, NONE)
OP_DECL(popart, matmul, matmul, AiOnnxOpset10.matmul, NONE, NONE)
OP_DECL(popart, maxunpool, maxunpool, AiOnnxOpset10.maxunpool, ARG(INT_VEC,kernel_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(kernel_shape) BODY_ARG(pads) BODY_ARG(strides) )
OP_DECL(popart, meanvariancenormalization, meanvariancenormalization, AiOnnxOpset10.meanvariancenormalization, ARG(INT_VEC,axes) , BODY_ARG(axes) )
OP_DECL(popart, nonzero, nonzero, AiOnnxOpset10.nonzero, NONE, NONE)
OP_DECL(popart, onehot, onehot, AiOnnxOpset10.onehot, ARG(INT,axis) , BODY_ARG(axis) )
OP_DECL(popart, prelu, prelu, AiOnnxOpset10.prelu, NONE, NONE)
OP_DECL(popart, scatter, scatter, AiOnnxOpset10.scatter, ARG(INT,axis) , BODY_ARG(axis) )
OP_DECL(popart, shrink, shrink, AiOnnxOpset10.shrink, ARG(FLOAT,bias) ARG(FLOAT,lambd) , BODY_ARG(bias) BODY_ARG(lambd) )
OP_DECL(popart, sign, sign, AiOnnxOpset10.sign, NONE, NONE)
OP_DECL(popart, sinh, sinh, AiOnnxOpset10.sinh, NONE, NONE)
OP_DECL(popart, where, where, AiOnnxOpset10.where, NONE, NONE)
// Ops from AiOnnxOpset8
OP_DECL(popart, expand, expand, AiOnnxOpset10.expand, NONE, NONE)
OP_DECL(popart, max, max, AiOnnxOpset10.max, NONE, NONE)
OP_DECL(popart, mean, mean, AiOnnxOpset10.mean, NONE, NONE)
OP_DECL(popart, min, min, AiOnnxOpset10.min, NONE, NONE)
OP_DECL(popart, sum, sum, AiOnnxOpset10.sum, NONE, NONE)
// Ops from AiOnnxOpset7
OP_DECL(popart, acos, acos, AiOnnxOpset10.acos, NONE, NONE)
OP_DECL(popart, add, add, AiOnnxOpset10.add, NONE, NONE)
OP_DECL(popart, logical_and, logical_and, AiOnnxOpset10.logical_and, NONE, NONE)
OP_DECL(popart, asin, asin, AiOnnxOpset10.asin, NONE, NONE)
OP_DECL(popart, atan, atan, AiOnnxOpset10.atan, NONE, NONE)
OP_DECL(popart, cos, cos, AiOnnxOpset10.cos, NONE, NONE)
OP_DECL(popart, div, div, AiOnnxOpset10.div, NONE, NONE)
OP_DECL(popart, equal, equal, AiOnnxOpset10.equal, NONE, NONE)
OP_DECL(popart, mul, mul, AiOnnxOpset10.mul, NONE, NONE)
OP_DECL(popart, multinomial, multinomial, AiOnnxOpset10.multinomial, ARG(INT,dtype) ARG(INT,sample_size) ARG(FLOAT,seed) , BODY_ARG(dtype) BODY_ARG(sample_size) BODY_ARG(seed) )
OP_DECL(popart, logical_or, logical_or, AiOnnxOpset10.logical_or, NONE, NONE)
OP_DECL(popart, pow, pow, AiOnnxOpset10.pow, NONE, NONE)
OP_DECL(popart, sin, sin, AiOnnxOpset10.sin, NONE, NONE)
OP_DECL(popart, sub, sub, AiOnnxOpset10.sub, NONE, NONE)
OP_DECL(popart, tan, tan, AiOnnxOpset10.tan, NONE, NONE)
OP_DECL(popart, logical_xor, logical_xor, AiOnnxOpset10.logical_xor, NONE, NONE)
// Ops from AiOnnxOpset6
OP_DECL(popart, abs, abs, AiOnnxOpset10.abs, NONE, NONE)
OP_DECL(popart, argmax, argmax, AiOnnxOpset10.argmax, ARG(INT,axis) ARG(INT,keepdims) , BODY_ARG(axis) BODY_ARG(keepdims) )
OP_DECL(popart, argmin, argmin, AiOnnxOpset10.argmin, ARG(INT,axis) ARG(INT,keepdims) , BODY_ARG(axis) BODY_ARG(keepdims) )
OP_DECL(popart, ceil, ceil, AiOnnxOpset10.ceil, NONE, NONE)
OP_DECL(popart, clip, clip, AiOnnxOpset10.clip, ARG(FLOAT,max) ARG(FLOAT,min) , BODY_ARG(max) BODY_ARG(min) )
OP_DECL(popart, concat, concat, AiOnnxOpset10.concat, ARG(INT,axis) , BODY_ARG(axis) )
OP_DECL(popart, conv, conv, AiOnnxOpset10.conv, ARG(INT_VEC,dilations) ARG(INT,group) ARG(INT_VEC,kernel_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(dilations) BODY_ARG(group) BODY_ARG(kernel_shape) BODY_ARG(pads) BODY_ARG(strides) )
OP_DECL(popart, convtranspose, convtranspose, AiOnnxOpset10.convtranspose, ARG(INT_VEC,dilations) ARG(INT,group) ARG(INT_VEC,kernel_shape) ARG(INT_VEC,output_padding) ARG(INT_VEC,output_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(dilations) BODY_ARG(group) BODY_ARG(kernel_shape) BODY_ARG(output_padding) BODY_ARG(output_shape) BODY_ARG(pads) BODY_ARG(strides) )
OP_DECL(popart, depthtospace, depthtospace, AiOnnxOpset10.depthtospace, ARG(INT,blocksize) , BODY_ARG(blocksize) )
OP_DECL(popart, elu, elu, AiOnnxOpset10.elu, ARG(FLOAT,alpha) , BODY_ARG(alpha) )
OP_DECL(popart, exp, exp, AiOnnxOpset10.exp, NONE, NONE)
OP_DECL(popart, floor, floor, AiOnnxOpset10.floor, NONE, NONE)
OP_DECL(popart, gather, gather, AiOnnxOpset10.gather, ARG(INT,axis) , BODY_ARG(axis) )
OP_DECL(popart, globalaveragepool, globalaveragepool, AiOnnxOpset10.globalaveragepool, NONE, NONE)
OP_DECL(popart, globallppool, globallppool, AiOnnxOpset10.globallppool, ARG(INT,p) , BODY_ARG(p) )
OP_DECL(popart, globalmaxpool, globalmaxpool, AiOnnxOpset10.globalmaxpool, NONE, NONE)
OP_DECL(popart, hardsigmoid, hardsigmoid, AiOnnxOpset10.hardsigmoid, ARG(FLOAT,alpha) ARG(FLOAT,beta) , BODY_ARG(alpha) BODY_ARG(beta) )
OP_DECL(popart, hardmax, hardmax, AiOnnxOpset10.hardmax, ARG(INT,axis) , BODY_ARG(axis) )
OP_DECL(popart, identity, identity, AiOnnxOpset10.identity, NONE, NONE)
OP_DECL(popart, instancenormalization, instancenormalization, AiOnnxOpset10.instancenormalization, ARG(FLOAT,epsilon) , BODY_ARG(epsilon) )
OP_DECL(popart, lrn, lrn, AiOnnxOpset10.lrn, ARG(INT,size) ARG(FLOAT,alpha) ARG(FLOAT,beta) ARG(FLOAT,bias) , BODY_ARG(size) BODY_ARG(alpha) BODY_ARG(beta) BODY_ARG(bias) )
OP_DECL(popart, leakyrelu, leakyrelu, AiOnnxOpset10.leakyrelu, ARG(FLOAT,alpha) , BODY_ARG(alpha) )
OP_DECL(popart, log, log, AiOnnxOpset10.log, NONE, NONE)
OP_DECL(popart, logsoftmax, logsoftmax, AiOnnxOpset10.logsoftmax, ARG(INT,axis) , BODY_ARG(axis) )
OP_DECL(popart, lpnormalization, lpnormalization, AiOnnxOpset10.lpnormalization, ARG(INT,axis) ARG(INT,p) , BODY_ARG(axis) BODY_ARG(p) )
OP_DECL(popart, lppool, lppool, AiOnnxOpset10.lppool, ARG(INT_VEC,kernel_shape) ARG(INT,p) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(kernel_shape) BODY_ARG(p) BODY_ARG(pads) BODY_ARG(strides) )
OP_DECL(popart, maxroipool, maxroipool, AiOnnxOpset10.maxroipool, ARG(INT_VEC,pooled_shape) ARG(FLOAT,spatial_scale) , BODY_ARG(pooled_shape) BODY_ARG(spatial_scale) )
OP_DECL(popart, neg, neg, AiOnnxOpset10.neg, NONE, NONE)
OP_DECL(popart, logical_not, logical_not, AiOnnxOpset10.logical_not, NONE, NONE)
OP_DECL(popart, pad, pad, AiOnnxOpset10.pad, ARG(INT_VEC,pads) ARG(STRING,mode) ARG(FLOAT,value) , BODY_ARG(pads) BODY_ARG(mode) BODY_ARG(value) )
OP_DECL(popart, randomnormallike, randomnormallike, AiOnnxOpset10.randomnormallike, ARG(INT,dtype) ARG(FLOAT,mean) ARG(FLOAT,scale) ARG(FLOAT,seed) , BODY_ARG(dtype) BODY_ARG(mean) BODY_ARG(scale) BODY_ARG(seed) )
OP_DECL(popart, randomuniformlike, randomuniformlike, AiOnnxOpset10.randomuniformlike, ARG(INT,dtype) ARG(FLOAT,high) ARG(FLOAT,low) ARG(FLOAT,seed) , BODY_ARG(dtype) BODY_ARG(high) BODY_ARG(low) BODY_ARG(seed) )
OP_DECL(popart, reciprocal, reciprocal, AiOnnxOpset10.reciprocal, NONE, NONE)
OP_DECL(popart, reducel1, reducel1, AiOnnxOpset10.reducel1, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) )
OP_DECL(popart, reducel2, reducel2, AiOnnxOpset10.reducel2, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) )
OP_DECL(popart, reducelogsum, reducelogsum, AiOnnxOpset10.reducelogsum, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) )
OP_DECL(popart, reducelogsumexp, reducelogsumexp, AiOnnxOpset10.reducelogsumexp, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) )
OP_DECL(popart, reducemax, reducemax, AiOnnxOpset10.reducemax, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) )
OP_DECL(popart, reducemean, reducemean, AiOnnxOpset10.reducemean, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) )
OP_DECL(popart, reducemin, reducemin, AiOnnxOpset10.reducemin, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) )
OP_DECL(popart, reduceprod, reduceprod, AiOnnxOpset10.reduceprod, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) )
OP_DECL(popart, reducesum, reducesum, AiOnnxOpset10.reducesum, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) )
OP_DECL(popart, reducesumsquare, reducesumsquare, AiOnnxOpset10.reducesumsquare, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) )
OP_DECL(popart, relu, relu, AiOnnxOpset10.relu, NONE, NONE)
OP_DECL(popart, reshape, reshape, AiOnnxOpset10.reshape, NONE, NONE)
OP_DECL(popart, selu, selu, AiOnnxOpset10.selu, ARG(FLOAT,alpha) ARG(FLOAT,gamma) , BODY_ARG(alpha) BODY_ARG(gamma) )
OP_DECL(popart, shape, shape, AiOnnxOpset10.shape, NONE, NONE)
OP_DECL(popart, sigmoid, sigmoid, AiOnnxOpset10.sigmoid, NONE, NONE)
OP_DECL(popart, size, size, AiOnnxOpset10.size, NONE, NONE)
OP_DECL(popart, softmax, softmax, AiOnnxOpset10.softmax, ARG(INT,axis) , BODY_ARG(axis) )
OP_DECL(popart, softplus, softplus, AiOnnxOpset10.softplus, NONE, NONE)
OP_DECL(popart, softsign, softsign, AiOnnxOpset10.softsign, NONE, NONE)
OP_DECL(popart, spacetodepth, spacetodepth, AiOnnxOpset10.spacetodepth, ARG(INT,blocksize) , BODY_ARG(blocksize) )
OP_DECL(popart, split, split, AiOnnxOpset10.split, ARG(INT,num_outputs) ARG(INT,axis) ARG(INT_VEC,split) , BODY_ARG(num_outputs) BODY_ARG(axis) BODY_ARG(split) )
OP_DECL(popart, sqrt, sqrt, AiOnnxOpset10.sqrt, NONE, NONE)
OP_DECL(popart, squeeze, squeeze, AiOnnxOpset10.squeeze, ARG(INT_VEC,axes) , BODY_ARG(axes) )
OP_DECL(popart, tanh, tanh, AiOnnxOpset10.tanh, NONE, NONE)
OP_DECL(popart, tile, tile, AiOnnxOpset10.tile, NONE, NONE)
OP_DECL(popart, transpose, transpose, AiOnnxOpset10.transpose, ARG(INT_VEC,perm) , BODY_ARG(perm) )
OP_DECL(popart, unsqueeze, unsqueeze, AiOnnxOpset10.unsqueeze, ARG(INT_VEC,axes) , BODY_ARG(axes) )

