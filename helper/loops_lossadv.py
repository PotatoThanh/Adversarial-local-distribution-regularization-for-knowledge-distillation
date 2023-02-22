from __future__ import print_function, division

import sys
import time
import torch

from .util import AverageMeter, accuracy
from attack_PGD import pgd_attack

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    advlosses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        
        input.requires_grad = True
        num=10
        input_adv = pgd_attack(model_t, input[:num, ::].detach(), target[:num]).detach()
        input_all = torch.cat([input_adv, input], dim=0)
        target_all = torch.cat([target[:num], target], dim=0)
        
        feat_s, logit_s_all = model_s(input_all, is_feat=True, preact=preact)
        logit_s = logit_s_all[num:, ::]
        feat_s = [f[num:, ::] for f in feat_s]

        feat_t, logit_t_all = model_t(input_all, is_feat=True, preact=preact)
        logit_t = logit_t_all[num:, ::]
        feat_t = [f[num:, ::].detach() for f in feat_t]
        
        # grad 
        cost_t = torch.nn.CrossEntropyLoss(reduction='none')(logit_t_all, target_all).detach()
        # grad_t = torch.autograd.grad(cost_t, input, retain_graph=True)[0]
        # grad_t = grad_t.detach()

        cost_s =  torch.nn.CrossEntropyLoss(reduction='none')(logit_s_all, target_all)
        # grad_s = torch.autograd.grad(cost_s, input, retain_graph=True)[0]
        # grad_s.requires_grad = True

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)
        
        # batch = grad_t.size(0)
        # grad_t = grad_t.view(batch, -1)
        # grad_s = grad_s.view(batch, -1)

        # # grad_t = torch.sign(grad_t).detach()
        # # grad_s = grad_s/(torch.abs(grad_s)+1e-5)

        # grad_t = grad_t/torch.norm(grad_t, p=2, dim=-1, keepdim=True)
        # grad_s = grad_t/torch.norm(grad_s, p=2, dim=-1, keepdim=True)

        # # grad_t = (grad_t-torch.mean(grad_t, dim=-1, keepdim=True))/torch.std(grad_t, dim=-1, keepdim=True)
        # # grad_s = (grad_t-torch.mean(grad_s, dim=-1, keepdim=True))/torch.std(grad_s, dim=-1, keepdim=True)
        
        # # grad_loss = grad_s * grad_t
        # # alpha = 1000.0
        # # grad_loss = 1.0/(1.0+torch.exp(-alpha*grad_loss))
        # # grad_loss = torch.sum(grad_loss, dim=1)
        # # grad_loss = torch.mean(grad_loss, dim=0)
        # # grad_loss = torch.nn.MSELoss()(grad_t,grad_s)

        # # alpha = 2.0
        # # grad_loss = torch.log(1+torch.exp(-alpha*grad_s*grad_t))
        # # grad_loss = torch.mean(grad_loss, dim=1)
        # # grad_loss = torch.mean(grad_loss, dim=0)

        # # grad_loss = torch.tanh(grad_s) * torch.tanh(grad_t)
        # # alpha = 5e+14
        # # grad_loss = 1.0/(1.0+torch.exp(-alpha*grad_loss))
        # # grad_loss = torch.sum(grad_loss, dim=1)
        # # grad_loss = -1.0*torch.mean(grad_loss, dim=0)

        # # grad_t = grad_t.view(batch, 3, -1)
        # # grad_s = grad_s.view(batch, 3, -1)
        # # grad_t = grad_t.view(batch, 1, -1)
        # # grad_s = grad_s.view(batch, 1, -1)
        # # grad_t = grad_t.view(batch, 3, -1).permute(0, 2, 1)
        # # grad_s = grad_s.view(batch, 3, -1).permute(0, 2, 1)


        # grad_t = grad_t.view(batch, 1, -1)
        # grad_s = grad_s.view(batch, 1, -1)
        # grad_loss = torch.nn.CosineSimilarity()(grad_s, grad_t)
        # grad_loss = torch.mean(grad_loss, dim=-1)
        # grad_loss = -1.0*torch.mean(grad_loss, dim=0)
        # if opt.l1 == 1:
        #     mae_loss = torch.nn.L1Loss()(grad_s, grad_t)
        #     # mae_loss = 0.0
        #     loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd + opt.g*(grad_loss + mae_loss)
        # else:
        #     loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd + opt.g*(grad_loss)
        grad_loss = torch.nn.MSELoss()(cost_s, cost_t)
        # print(grad_loss)
        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd + opt.g*grad_loss

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        advlosses.update(grad_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Advloss {advlosse.val:.4f} ({advlosse.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, advlosse=advlosses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
