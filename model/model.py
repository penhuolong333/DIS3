import torch.nn as nn
from model import model_original 
from model import model_cd



class TrainSE_With_WCTDecoder(nn.Module):
  def __init__(self, args):
    super(TrainSE_With_WCTDecoder, self).__init__()
    self.BE = model_original.Encoder(args.BE,fixed=True)
    self.BD = model_original.Decoder(args.BD,fixed=True)
    self.SE = model_cd.SmallEncoder(args.SE, fixed=False)
    self.args = args
    
  def forward(self,c,s,iter):
    self.alpha = 1.0
    self.mse_loss = nn.MSELoss()
    # BE forward, multi outputs: relu1_1, 2_1, 3_1, 4_1, 5_1
    cF_BE = self.BE.forward_branch(c)
    # 用BE进行style transfer:
    # BE_feat = model_cd.exact_feature_distribution_matching(cF_BE,sF_BE)#efdm
    # BE_feat = BE_feat * alpha + cF_BE * (1 - alpha) #计算feat
    # rec1 = self.BD(BE_feat) #BD解码
    # g_t_feats = self.encode_with_intermediate(rec1)

    #SE编码
    # style_feats = self.SE.forward_branch(s)
    # content_feat = self.SE.forward_branch(c)
    content_feat = self.SE.forward_aux(c,self.args.updim_relu)
    style_feats = self.SE.forward_aux(s,self.args.updim_relu)
    #efdm
    SE_feat = model_cd.exact_feature_distribution_matching(content_feat[-2],style_feats[-2])  # efdm
    SE_feat = SE_feat * self.alpha + content_feat[-2] * (1 - self.alpha)  # 计算feat
    #用BD进行重构
    rec = self.BD(SE_feat)
    #用BE进行重新编码
    rec_feats = self.BE.forward_branch(rec)
    sF_BE = self.BE.forward_branch(s)
    # for log
    sd_BE = 0
    if iter % self.args.save_interval == 0:
      rec_BE = self.BD(cF_BE[-1])
    
    # (loss 1) BE -> SE knowledge transfer loss
    feat_loss = 0
    for i in range(len(cF_BE)):
      feat_loss += nn.MSELoss()(content_feat[i], cF_BE[i].data)  #这里不变
    
    # (loss 2, 3) eval the quality of reconstructed image, pixel and perceptual loss
    # rec_pixl_loss = nn.MSELoss()(rec, c.data)
    # recF_BE = self.BE.forward_branch(rec)
    # rec_perc_loss = 0
    # for i in range(len(recF_BE)):
    #   rec_perc_loss += nn.MSELoss()(recF_BE[i], cF_BE[i].data)
    # return feat_loss, rec_pixl_loss, rec_perc_loss, rec, c

    #(loss2)
    #计算relu4_1层，风格化图片与原内容图的内容损失：
    rec_contentloss = nn.MSELoss()(rec_feats[-2], cF_BE[-2].data)

    #(loss3)
    #计算每层的风格损失差异:
    rec_styleloss = nn.MSELoss()(model_cd.gram_matrix(rec_feats[0]), model_cd.gram_matrix(sF_BE[0]))
    # rec_styleloss = model_cd.calc_style_loss((rec_feats[0]), (sF_BE[0]))
    for i in range(1, 4):
      rec_styleloss += nn.MSELoss()(model_cd.gram_matrix(rec_feats[i]), model_cd.gram_matrix(sF_BE[i]))
      # rec_styleloss = model_cd.calc_style_loss((rec_feats[i]), (sF_BE[i]))

    return feat_loss,rec_contentloss,rec_styleloss,rec,c

class TrainSD_With_WCTSE(nn.Module):
  def __init__(self, args):
    super(TrainSD_With_WCTSE, self).__init__()
    self.BE = model_original.Encoder(args.BE, fixed=True)
    self.SE = model_cd.SmallEncoder(args.SE,fixed=True)
    self.SD = model_cd.SmallDecoder(args.SD, fixed=False)
    # self.SE = eval("model_cd.SmallEncoder%d_%dx_aux" % (args.stage, args.speedup))(args.SE, fixed=True)
    # self.SD = eval("model_cd.SmallDecoder%d_%dx" % (args.stage, args.speedup))(args.SD, fixed=False)
    self.args = args
    
  def forward(self, c, iter):
    rec = self.SD(self.SE(c))
    # loss (1) pixel loss
    rec_pixl_loss = nn.MSELoss()(rec, c.data)
    
    # loss (2) perceptual loss
    recF_BE = self.BE.forward_branch(rec)
    cF_BE = self.BE.forward_branch(c)
    rec_perc_loss = 0
    for i in range(len(recF_BE)):
      rec_perc_loss += nn.MSELoss()(recF_BE[i], cF_BE[i].data)
      
    return rec_pixl_loss, rec_perc_loss, rec

class TrainSD_With_WCTSE_KD2SD(nn.Module):
  def __init__(self, args):
    super(TrainSD_With_WCTSE_KD2SD, self).__init__()
    self.BE = eval("model_original.Encoder%d" % args.stage)(args.BE, fixed=True)
    self.BD = eval("model_original.Decoder%d" % args.stage)(None, fixed=True)
    self.SE = eval("model_cd.SmallEncoder%d_%dx_aux" % (args.stage, args.speedup))(None, fixed=True)
    self.SD = eval("model_cd.SmallDecoder%d_%dx_aux" % (args.stage, args.speedup))(args.SD, fixed=False)
    self.args = args
    
  def forward(self, c, iter):
    feats_BE = self.BE.forward_branch(c) # for perceptual loss

    *_, feat_SE_aux, feat_SE = self.SE.forward_aux2(c) # output the last up-size feature and normal-size feature
    feats_BD = self.BD.forward_branch(feat_SE_aux)
    feats_SD = self.SD.forward_aux(feat_SE, relu=self.args.updim_relu)
    rec = feats_SD[-1]

    # loss (1) pixel loss
    rec_pixl_loss = nn.MSELoss()(rec, c.data)
    
    # loss (2) perceptual loss
    rec_feats_BE = self.BE.forward_branch(rec)
    rec_perc_loss = 0
    for i in range(len(rec_feats_BE)):
      rec_perc_loss += nn.MSELoss()(rec_feats_BE[i], feats_BE[i].data)
    
    # loss (3) kd feature loss
    kd_feat_loss = 0
    for i in range(len(feats_BD)):
      kd_feat_loss += nn.MSELoss()(feats_SD[i], feats_BD[i].data)

    return rec_pixl_loss, rec_perc_loss, kd_feat_loss, rec