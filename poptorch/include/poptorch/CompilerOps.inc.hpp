// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// Auto generated file, do not modify
// Run `python3 PopParse.py to regenerate
// clang-format off

torch::jit::Node* createGroupnormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t num_groups,float epsilon);
torch::jit::Node* createSubsample(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & strides);
torch::jit::Node* createPrinttensor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t print_gradient,const std::string & title);
torch::jit::Node* createNop(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createScale(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float scale);
torch::jit::Node* createScaledadd(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float scale0,float scale1);
torch::jit::Node* createLstm(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t outputFullSequence);
torch::jit::Node* createGelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createDetach(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createDepthtospace(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t blocksize,const std::string & mode);
torch::jit::Node* createRound(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createDynamicslice(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,std::vector<int64_t> sizes,std::int32_t noOverlap);
torch::jit::Node* createDynamicupdate(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,std::vector<int64_t> sizes,std::int32_t noOverlap);
torch::jit::Node* createDynamiczero(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,std::vector<int64_t> sizes);
torch::jit::Node* createDynamicadd(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,std::vector<int64_t> sizes);
torch::jit::Node* createReplicatedallreduce(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createL1loss(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const float lambda,std::int32_t reduction);
torch::jit::Node* createNllloss(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t reduction,std::int32_t ignoreIndex,bool inputIsLogProbability);
torch::jit::Node* createIdentityloss(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t reduction);
torch::jit::Node* createShapeddropout(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & shape,float ratio);
torch::jit::Node* createAtan2(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createExpm1(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createLog1p(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createFmod(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createRemainder(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createReverse(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dimensions);
torch::jit::Node* createAveragepool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,int64_t ceil_mode,int64_t count_include_pad,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides);
torch::jit::Node* createConvinteger(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides);
torch::jit::Node* createDequantizelinear(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createDropout(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,float ratio);
torch::jit::Node* createIsinf(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t detect_negative,int64_t detect_positive);
torch::jit::Node* createMatmulinteger(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createMaxpool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,const std::vector<int64_t> & kernel_shape,int64_t ceil_mode,const std::vector<int64_t> & dilations,const std::vector<int64_t> & pads,int64_t storage_order,const std::vector<int64_t> & strides);
torch::jit::Node* createMod(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t fmod);
torch::jit::Node* createNonmaxsuppression(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t center_point_box);
torch::jit::Node* createQlinearconv(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides);
torch::jit::Node* createQlinearmatmul(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createQuantizelinear(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createResize(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::string & mode);
torch::jit::Node* createReversesequence(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t batch_axis,int64_t time_axis);
torch::jit::Node* createRoialign(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::string & mode,int64_t output_height,int64_t output_width,int64_t sampling_ratio,float spatial_scale);
torch::jit::Node* createSlice(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createThresholdedrelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha);
torch::jit::Node* createTopk(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis);
torch::jit::Node* createUpsample(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::string & mode);
torch::jit::Node* createAcosh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createAsinh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createAtanh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createBatchnormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,float epsilon,float momentum);
torch::jit::Node* createCast(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::string & to);
torch::jit::Node* createCompress(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t axis);
torch::jit::Node* createCosh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createErf(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createEyelike(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,int64_t k);
torch::jit::Node* createFlatten(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis);
torch::jit::Node* createGemm(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha,float beta,int64_t transA,int64_t transB);
torch::jit::Node* createGreater(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createIsnan(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createLess(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createMatmul(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createMaxunpool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides);
torch::jit::Node* createMeanvariancenormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes);
torch::jit::Node* createNonzero(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createOnehot(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis);
torch::jit::Node* createPrelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createScatter(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis);
torch::jit::Node* createShrink(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float bias,float lambd);
torch::jit::Node* createSign(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createSinh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createWhere(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createExpand(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createMax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createMean(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createMin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createSum(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createAcos(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createAdd(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createLogical_and(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createAsin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createAtan(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createCos(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createDiv(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createEqual(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createMul(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createMultinomial(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t dtype,int64_t sample_size,float seed);
torch::jit::Node* createLogical_or(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createPow(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createSin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createSub(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createTan(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createLogical_xor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createAbs(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createArgmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t keepdims);
torch::jit::Node* createArgmin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t keepdims);
torch::jit::Node* createCeil(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createClip(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float max,float min);
torch::jit::Node* createConcat(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis);
torch::jit::Node* createConv(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides);
torch::jit::Node* createConvtranspose(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & output_padding,const std::vector<int64_t> & output_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides);
torch::jit::Node* createElu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha);
torch::jit::Node* createExp(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createFloor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createGather(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis);
torch::jit::Node* createGlobalaveragepool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createGloballppool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t p);
torch::jit::Node* createGlobalmaxpool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createHardsigmoid(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha,float beta);
torch::jit::Node* createHardmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis);
torch::jit::Node* createIdentity(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createInstancenormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float epsilon);
torch::jit::Node* createLrn(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t size,float alpha,float beta,float bias);
torch::jit::Node* createLeakyrelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha);
torch::jit::Node* createLog(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createLogsoftmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis);
torch::jit::Node* createLpnormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t p);
torch::jit::Node* createLppool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,int64_t p,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides);
torch::jit::Node* createMaxroipool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & pooled_shape,float spatial_scale);
torch::jit::Node* createNeg(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createLogical_not(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createPad(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & pads,const std::string & mode,float value);
torch::jit::Node* createRandomnormallike(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,float mean,float scale,float seed);
torch::jit::Node* createRandomuniformlike(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,float high,float low,float seed);
torch::jit::Node* createReciprocal(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createReducel1(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims);
torch::jit::Node* createReducel2(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims);
torch::jit::Node* createReducelogsum(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims);
torch::jit::Node* createReducelogsumexp(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims);
torch::jit::Node* createReducemax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims);
torch::jit::Node* createReducemean(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims);
torch::jit::Node* createReducemin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims);
torch::jit::Node* createReduceprod(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims);
torch::jit::Node* createReducesum(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims);
torch::jit::Node* createReducesumsquare(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims);
torch::jit::Node* createRelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createSelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha,float gamma);
torch::jit::Node* createShape(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createSigmoid(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createSize(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createSoftmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis);
torch::jit::Node* createSoftplus(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createSoftsign(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createSpacetodepth(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t blocksize);
torch::jit::Node* createSplit(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,int64_t axis,const std::vector<int64_t> & split);
torch::jit::Node* createSqrt(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createSqueeze(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes);
torch::jit::Node* createTanh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createTile(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args);
torch::jit::Node* createTranspose(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & perm);
torch::jit::Node* createUnsqueeze(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes);

