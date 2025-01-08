# coding: utf-8

from logging import getLogger
from itertools import product

import torch

from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))
    # total=0
    # for u in range(39387):
    #     total +=len(set(train_data.hidden_item_per_user[u]) & set(test_data.eval_items_per_u[u]))
         # len(train_data.history_items_per_u[u].intersection(test_data.eval_items_per_u[u]))
    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')
    # meta_weight: [1, 0.7, 0.5, 0.3, 0.1]
    # n_meta_layer: [8, 7, 6, 5, 4, 3]
    # att_weight: [0.15, 0.1, 0.05, 0.2, 0.3, 0.5]
    # att_init: [0.3, 0.1, 0.2, 0.4, 0.5]
    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx + 1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer('Trainer')(config, model)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid= trainer.fit(train_data, valid_data=valid_data,
                                                                                test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))
        # torch.save((h[0],h[1],h[2]),'teacher'+config['dataset']+'.pth')
        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
                                                        hyper_ret[best_test_idx][0],
                                                        dict2str(hyper_ret[best_test_idx][1]),
                                                        dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                        hyper_ret[best_test_idx][0],
                                                                        dict2str(hyper_ret[best_test_idx][1]),
                                                                        dict2str(hyper_ret[best_test_idx][2])))
    return hyper_ret
    # import numpy as np
    # _, i_v_feats = model.agg_mm_neighbors('v')
    # np.save('log/'+config['model'] + '-' + config['dataset']+'-i_v_feats', i_v_feats.detach().cpu().numpy())
    # _, i_t_feats = model.agg_mm_neighbors('t')
    # np.save('log/'+config['model'] + '-' + config['dataset']+'-i_t_feats', i_t_feats.detach().cpu().numpy())

# def quick_start2(model, dataset, config_dict, save_model=True):
#     # merge config dict
#     config = Config(model, dataset, config_dict)
#     init_logger(config)
#     logger = getLogger()
#     # print config infor
#     logger.info('██Server: \t' + platform.node())
#     logger.info('██Dir: \t' + os.getcwd() + '\n')
#     logger.info(config)
#
#     # load data
#     dataset = RecDataset(config)
#     # print dataset statistics
#     logger.info(str(dataset))
#
#     train_dataset, valid_dataset, test_dataset = dataset.split()
#     logger.info('\n====Training====\n' + str(train_dataset))
#     logger.info('\n====Validation====\n' + str(valid_dataset))
#     logger.info('\n====Testing====\n' + str(test_dataset))
#
#     # wrap into dataloader
#     train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
#     (valid_data, test_data) = (
#         EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
#         EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))
#
#     ############ Dataset loadded, run model
#     hyper_ret = []
#     val_metric = config['valid_metric'].lower()
#     best_test_value = 0.0
#     idx = best_test_idx = 0
#
#     logger.info('\n\n=================================\n\n')
#     # meta_weight: [1, 0.7, 0.5, 0.3, 0.1]
#     # n_meta_layer: [8, 7, 6, 5, 4, 3]
#     # att_weight: [0.15, 0.1, 0.05, 0.2, 0.3, 0.5]
#     # att_init: [0.3, 0.1, 0.2, 0.4, 0.5]
#     # hyper-parameters
#     hyper_ls = []
#     if "seed" not in config['hyper_parameters']:
#         config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
#     for i in config['hyper_parameters']:
#         hyper_ls.append(config[i] or [None])
#     # combinations
#     combinators = list(product(*hyper_ls))
#     total_loops = len(combinators)
#     for hyper_tuple in combinators:
#         # random seed reset
#         for j, k in zip(config['hyper_parameters'], hyper_tuple):
#             config[j] = k
#         init_seed(config['seed'])
#
#         logger.info('========={}/{}: Parameters:{}={}======='.format(
#             idx + 1, total_loops, config['hyper_parameters'], hyper_tuple))
#
#         # set random state of dataloader
#         train_data.pretrain_setup()
#         # model loading and initialization
#         model = get_model(config['model'])(config, train_data).to(config['device'])
#         logger.info(model)
#
#         # trainer loading and initialization
#         trainer = get_trainer('Trainer2')(config, model)
#         # debug
#         # model training
#         best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data,
#                                                                                 test_data=test_data, saved=save_model)
#         #########
#         hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))
#
#         # save best test
#         if best_test_upon_valid[val_metric] > best_test_value:
#             best_test_value = best_test_upon_valid[val_metric]
#             best_test_idx = idx
#         idx += 1
#
#         logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
#         logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
#         logger.info('████Current BEST████:\nParameters: {}={},\n'
#                     'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
#                                                         hyper_ret[best_test_idx][0],
#                                                         dict2str(hyper_ret[best_test_idx][1]),
#                                                         dict2str(hyper_ret[best_test_idx][2])))
#
#     # log info
#     logger.info('\n============All Over=====================')
#     for (p, k, v) in hyper_ret:
#         logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
#                                                                                   p, dict2str(k), dict2str(v)))
#
#     logger.info('\n\n█████████████ BEST ████████████████')
#     logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
#                                                                         hyper_ret[best_test_idx][0],
#                                                                         dict2str(hyper_ret[best_test_idx][1]),
#                                                                         dict2str(hyper_ret[best_test_idx][2])))
#
#     # import numpy as np
#     # _, i_v_feats = model.agg_mm_neighbors('v')
#     # np.save('log/'+config['model'] + '-' + config['dataset']+'-i_v_feats', i_v_feats.detach().cpu().numpy())
#     # _, i_t_feats = model.agg_mm_neighbors('t')
#     # np.save('log/'+config['model'] + '-' + config['dataset']+'-i_t_feats', i_t_feats.detach().cpu().numpy())