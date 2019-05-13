import torch
import torch.nn as nn
import torch.nn.functional as F

# class FocalLoss(torch.nn.Module):
#     def __init__(self, gamma=2):
#         super().__init__()
#         self.gamma = gamma
#
#     def forward(self, log_pred_prob_onehot, target):
#         pred_prob_oh = torch.exp(log_pred_prob_onehot)
#         pt = pred_prob_oh.gather(1, target.data.view(-1, 1))
#         modulator = (1 - pt) ** self.gamma
#         mce = modulator * (-torch.log(pt))
#
#         return mce.mean()


class FocalLoss(nn.Module):
    r"""
        This criterion is a implementation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each mini-batch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classified examples (p > .5),
                                   putting more focus on hard, misclassified examples
            size_average(bool): By default, the losses are averaged over observations for each mini-batch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each mini-batch.

        Input:
            inputs: [N, C]
            targets: [N, 1]
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = torch.tensor([alpha,1-alpha])
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)      # batch_size
        C = inputs.size(1)      # 2
        P = F.softmax(inputs,dim=1)

        # class_mask = inputs.new(N, C).fill_(0)
        class_mask = torch.zeros_like(inputs)
        # ids = targets.view(-1, 1)
        ids = targets.view(-1,1)
        class_mask.scatter_(1, ids, 1)      # 生成 one-hot 标签

        # print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to("cuda")
        # alpha = self.alpha[ids.data.view(-1)]
        alpha = self.alpha[targets]

        probs = (P * class_mask).sum(1).view(-1, 1)

        probs += 1e-8

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss



class FocalLoss_bce(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss_bce, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        targets = targets.float()
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(F.softmax(inputs,dim=1)[:,1], targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(F.softmax(inputs,dim=1)[:,1], targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# if __name__ == '__main__':
#     loss = FocalLoss(2)
#
#     # conf_mask = torch.FloatTensor([0.0, 1.0, 0.0, 1.0, 1.0]) - 1
#     # conf_data = torch.FloatTensor([-0.1, -0.9, 0.0, -0.2, -0.2])
#     conf_data = torch.randn(3,2)
#     conf_mask = torch.tensor([1,0,1], dtype=torch.long)
#
#     print(loss(conf_data, conf_mask))