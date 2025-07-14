import logging
from copy import deepcopy
import torch
import torch.nn as nn
import clip
from base import Trainer
from dataloader.data_utils import get_dataloader, get_id2label
from utils import *
# from .Network_backup2 import MYNET
from .Network import MYNET
from .helper import *


class FSCILTrainer(Trainer):
    def __init__(self, args):
        """
        Initialize FSCILTrainer and pre-extract text features for all categories
        """
        super().__init__(args)
        self.args = args
        self.set_up_model()

    def set_up_model(self):
        """ Initialize MYNET model """
        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            logging.info('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir,
                                              map_location={'cuda:3': 'cuda:0'})['params']
        else:
            logging.info('Random init params')
            if self.args.start_session > 0:
                logging.info('WARNING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def train(self, ):
        args = self.args
        t_start_time = time.time()
        result_list = [args]

        final_avgac = 0

        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = get_dataloader(args, session)
            self.model.load_state_dict(self.best_model_dict)
            if session == 0:  # load base class train img label
                if not args.only_do_incre:
                    logging.info(f'new classes for this session:{np.unique(train_set.targets)}')
                    optimizer, scheduler = get_optimizer(args, self.model)

                    # ================ no training used in FDR, set epochs_base = 0 ====================
                    for epoch in range(args.epochs_base):
                        start_time = time.time()

                        tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                        tsl, tsa = test(self.model, testloader, epoch, args, session, result_list=result_list)

                        # save better model
                        if (tsa * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                            torch.save(dict(params=self.model.state_dict()), save_model_dir)
                            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                            self.best_model_dict = deepcopy(self.model.state_dict())
                            logging.info('********A better model is found!!**********')
                            logging.info('Saving model to :%s' % save_model_dir)
                        logging.info('best epoch {}, best test acc={:.3f}'.format(
                            self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

                        self.trlog['train_loss'].append(tl)
                        self.trlog['train_acc'].append(ta)
                        self.trlog['test_loss'].append(tsl)
                        self.trlog['test_acc'].append(tsa)
                        lrc = scheduler.get_last_lr()[0]

                        logging.info(
                            'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, lrc, tl, ta, tsl, tsa))
                        print('This epoch takes %d seconds' % (time.time() - start_time),
                              '\n still need around %.2f mins to finish this session' % (
                                      (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                        scheduler.step()

                    # Finish base train
                    logging.info('>>> Finish Base Train <<<')
                    result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                        session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))
                else:
                    logging.info('>>> Load Model &&& Finish base train...')
                    assert args.model_dir is not None

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)

                    # ==== Train keyword attention, split base class samples into two: one for base class and the other for new class ====
                    if args.soft_mode == 'keyword_seg_with_training':
                        optimizer, scheduler = get_optimizer(args, self.model)

                        # 1. Split base class samples, use the first half to construct base class prototypes
                        args.split_idx = int(args.base_class * args.split_ratio)
                        self.model, embeddings, labels = replace_base_fc_split(train_set, testloader.dataset.transform,
                                                                               self.model,
                                                                               args)
                        # Update class prototypes for each epoch in the training loop
                        for epoch in range(args.epochs_split):
                            # 2. Calibration
                            current_novel_prototypes = self.model.module.keyword_seg_calibration_training(args)
                            # 3. Compute loss and train keyword attention based on the calibration results
                            # Get current new class prototypes and GT
                            # current_novel_prototypes = self.model.module.fc.weight[args.split_idx:args.base_class]
                            gt_prototypes = self.model.module.gt_novel_prototypes

                            # Design dual loss function TODO
                            mse_loss = F.mse_loss(current_novel_prototypes, gt_prototypes)
                            # cosine_loss = 1 - F.cosine_similarity(current_novel_prototypes, gt_prototypes).mean()
                            # calibration_loss = mse_loss + cosine_loss
                            calibration_loss = mse_loss

                            # Combine other necessary losses (e.g., classification loss) TODO
                            # # 1. Get current batch data
                            for batch in trainloader:  # Assume trainloader contains base class samples
                                _inputs, _labels = batch
                                _inputs = _inputs.cuda()
                                _labels = _labels.cuda()

                                # 2. Extract features
                                self.model.module.mode = 'encoder'
                                features = self.model(_inputs)  # [B, embed_dim]

                                # 3. Calculate classification logits (based on all base class prototypes)
                                prototypes = self.model.module.fc.weight[:args.base_class]  # [base_class, embed_dim]
                                logits = torch.mm(features, prototypes.T)  # [B, base_class]

                                # 4. Calculate cross-entropy loss
                                classification_loss = F.cross_entropy(logits / args.temperature, _labels)

                            # total_loss = calibration_loss  + classification_loss
                            # total_loss = calibration_loss
                            total_loss = calibration_loss * args.lambda_weight + classification_loss

                            # print(self.model.module.fc.weight)

                            # **** Save original prototype parameters
                            original_protos = self.model.module.fc.weight.data.clone()

                            optimizer.zero_grad()
                            # calibration_loss.backward()
                            total_loss.backward()

                            # Save parameters before optimization
                            # old_params = {name: param.clone().detach() for name, param in self.model.module.named_parameters()}

                            # Perform parameter update
                            optimizer.step()

                            # Check if parameters are updated
                            # print("\n=== Parameter Update Check ===")
                            # for name, param in self.model.module.named_parameters():
                            #     old_param = old_params[name]
                            #     if torch.equal(param.data, old_param):
                            #         # print(f"Parameter '{name}' was not updated")
                            #         pass
                            #     else:
                            #         print(f"Parameter '{name}' was updated")

                            # **** Restore original prototype parameters
                            self.model.module.fc.weight.data.copy_(original_protos)

                            # Clear possible remaining gradients
                            self.model.module.fc.weight.grad = None

                            lrc = scheduler.get_last_lr()[0]
                            print('epoch:%03d,  lr:%.4f,  loss::%.4f,  cali loss::%.4f,  cls loss::%.4f' % (
                            epoch, lrc, total_loss, calibration_loss, classification_loss))
                            scheduler.step()
                            # 4. For each epoch, construct new class prototypes using small samples
                            self.model = update_novel_prototypes(self.model, embeddings, labels, args, epoch)

                            # self.model.module.fc.weight.data[args.split_idx:args.base_class] = self.model.module.gt_novel_prototypes
                            self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model,
                                                         args)  # 不论是否训练keyword attention，用基类样本重置原型

                            #
                            if args.soft_mode == 'keyword_seg':
                                self.model.module.build_knowledge_base(args)

                            best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                            logging.info('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                            self.best_model_dict = deepcopy(self.model.state_dict())
                            torch.save(dict(params=self.model.state_dict()), best_model_dir)

                            self.model.module.mode = 'avg_cos'
                            tsl, tsa = test(self.model, testloader, 0, args, session, result_list=result_list)
                            if (tsa * 100) >= self.trlog['max_acc'][session]:
                                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                                logging.info('The new best test acc of base session={:.3f}'.format(
                                    self.trlog['max_acc'][session]))

                    # incremental learning sessions
                    else:
                        logging.info("training session: [%d]" % session)
                        self.model.module.mode = self.args.new_mode
                        self.model.eval()
                        trainloader.dataset.transform = testloader.dataset.transform

                        self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                        if args.soft_mode == 'soft_proto':
                            self.model.module.soft_calibration(args, session)
                        elif args.soft_mode == 'soft_proto_txt':
                            self.model.module.soft_calibration_txt(args, session, self.text_features)  # 传递缓存的文本特征
                        elif args.soft_mode == 'no_calibration':
                            pass
                        elif args.soft_mode == 'random_seg':
                            self.model.module.random_seg_calibration(args, session)
                        elif args.soft_mode == 'keyword_seg':
                            self.model.module.keyword_seg_calibration(args, session)
                        elif args.soft_mode == 'keyword_seg_with_training':
                            self.model.module.keyword_seg_calibration(args, session)
                        else:
                            raise NotImplementedError

                        tsl, (seenac, unseenac, avgac) = test(self.model, testloader, 0, args, session,
                                                              result_list=result_list)
                        final_avgac = avgac
                        # update results and save model
                        self.trlog['seen_acc'].append(float('%.3f' % (seenac * 100)))
                        self.trlog['unseen_acc'].append(float('%.3f' % (unseenac * 100)))
                        self.trlog['max_acc'][session] = float('%.3f' % (avgac * 100))
                        self.best_model_dict = deepcopy(self.model.state_dict())

                        logging.info(f"Session {session} ==> Seen Acc:{self.trlog['seen_acc'][-1]} "
                                     f"Unseen Acc:{self.trlog['unseen_acc'][-1]} Avg Acc:{self.trlog['max_acc'][session]}")
                        result_list.append(
                            'Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

                # Finish all incremental sessions, save results.
                result_list, hmeans = postprocess_results(result_list, self.trlog)
                save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)
                if not self.args.debug:
                    save_result(args, self.trlog, hmeans)

                t_end_time = time.time()
                total_time = (t_end_time - t_start_time) / 60
                logging.info(f"Base Session Best epoch:{self.trlog['max_acc_epoch']}")
                logging.info('Total time used %.2f mins' % total_time)
                logging.info(self.args.time_str)

                return final_avgac
