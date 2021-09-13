import os
import logging
import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import gather
import cv2

from common.speedometer import BatchEndParam
from common_pytorch.dataset.all_dataset import *
from common_pytorch.common_loss.loss_recorder import LossRecorder
from common.utility.image_processing_cv import flip
from common_pytorch.group.tag_group import HeatmapParser, group_corners_on_tags, mapInputForPostProcessing
from post_processing.postProcess import PostProcess
from calculateTotalParams.calculateTotalParams import CalculateTotalParams
import numpy as np

def trainNet(nth_epoch, train_data_loader, network, optimizer, loss_config, loss_func, speedometer=None):
    """
    :param nth_epoch:
    :param train_data_loader: batch_size, dataset.db_length
    :param network:
    :param optimizer:
    :param loss_config:
    :param loss_func:
    :param speedometer:
    :param tensor_board:
    :return:
    """
    network.train()

    loss_recorder = LossRecorder()
    for idx, _data in enumerate(train_data_loader):
        batch_data = _data[0].cuda()
        batch_hm_label = _data[1].cuda()
        batch_gt_loc = _data[2]

        optimizer.zero_grad()

        heatmaps, tagmaps = network(batch_data)
        del batch_data

        loss = loss_func(heatmaps, tagmaps, batch_hm_label, batch_gt_loc)
        del batch_hm_label, batch_gt_loc
        del heatmaps, tagmaps

        loss.backward()

        optimizer.step()

        loss_recorder.update(loss.detach(), train_data_loader.batch_size)
        del loss

        if speedometer != None:
            speedometer(BatchEndParam(epoch=nth_epoch, nbatch=idx,
                                      total_batch=len(train_data_loader.dataset) // train_data_loader.batch_size,
                                      add_step=True, eval_metric=None, loss_metric=loss_recorder, locals=locals()))

    return loss_recorder.get_avg()

def validNet(valid_data_loader, network, loss_func, merge_hm_flip_func, merge_tag_flip_func,
             devices, flip_pair, flip_test):
    """

    :param nth_epoch:
    :param valid_data_loader:
    :param network:
    :param loss_config:
    :param result_func:
    :param loss_func:
    :param patch_size:
    :param devices:
    :param tensor_board:
    :return:
    """
    print('in valid')
    network.eval()

    loss_recorder = LossRecorder()

    heatmaps_list = []
    tagmaps_list = []
    with torch.no_grad():
        for idx, _data in enumerate(valid_data_loader):
            batch_data = _data[0].cuda()
            batch_hm_label = _data[1].cuda()
            batch_gt_loc = _data[2]

            heatmaps, tagmaps = network(batch_data)

            if flip_test:
                batch_data_flip = flip(batch_data, dims=3)
                heatmaps_flip, tagmaps_flip = network(batch_data_flip)
                del batch_data_flip

            del batch_data

            if len(heatmaps.shape) == 5:
                heatmaps = heatmaps[:, -1]
                tagmaps = tagmaps[:, -1]
                if flip_test:
                    heatmaps_flip = heatmaps_flip[:, -1]
                    tagmaps_flip = tagmaps_flip[:, -1]

            loss = loss_func(heatmaps, tagmaps, batch_hm_label, batch_gt_loc)
            del batch_hm_label

            loss_recorder.update(loss.detach(), valid_data_loader.batch_size)
            del loss

            # because we are using new DataParallel, so need to gather prediction from GPUs
            if len(devices) > 1: # TODO: check
                heatmaps = gather(heatmaps, 0)
                tagmaps = gather(tagmaps, 0)

            if flip_test:
                if len(devices) > 1:  # TODO: check
                    heatmaps_flip = gather(heatmaps_flip, 0)
                    tagmaps_flip = gather(tagmaps_flip, 0)
                heatmaps = merge_hm_flip_func(heatmaps, heatmaps_flip, flip_pair)
                tagmaps = merge_tag_flip_func(tagmaps, tagmaps_flip, flip_pair)
                del heatmaps_flip, tagmaps_flip

            heatmaps_list.append(heatmaps)
            tagmaps_list.append(tagmaps)

            del heatmaps, tagmaps

    return torch.cat(heatmaps_list), torch.cat(tagmaps_list), loss_recorder.get_avg()

def evalNet(nth_epoch, heatmaps, tagmaps, valid_data_loader, loss_config, test_config,
            patch_width, patch_height, final_output_path):

    print("in eval")

    heatmaps = nn.UpsamplingBilinear2d((patch_height, patch_width)).cuda()(heatmaps)
    tagmaps = nn.UpsamplingBilinear2d((patch_height, patch_width)).cuda()(tagmaps)
    heatmaps = heatmaps.cpu().numpy().astype(float)
    tagmaps = tagmaps.cpu().numpy().astype(float)

    imdb_list = valid_data_loader.dataset.db

    num_samples = heatmaps.shape[0]
    assert num_samples == tagmaps.shape[0]
    assert num_samples == len(imdb_list)

    windows_list_with_score = list()
    ############################ Group corners on TAG ##############################
    # 1. Group Corners
    parser = HeatmapParser(loss_config, test_config.useCenter, test_config.centerT, imdb_list)
    for n_s in range(num_samples):
        try:
            group_corners_wz_score = \
                group_corners_on_tags(n_s, parser, heatmaps[n_s], tagmaps[n_s], patch_width, patch_height,
                                      imdb_list[n_s]['im_width'], imdb_list[n_s]['im_height'],
                                      rectify = test_config.rectify, winScoreThres = test_config.windowT)
            windows_list_with_score.append(group_corners_wz_score)
        except Exception as e:
            assert 0, (n_s, e, os.path.basename(imdb_list[n_s]['image']))

    # 2. Evaluate
    name_value = facade.evaluate(windows_list_with_score, imdb_list, final_output_path,
                                 test_config.fullEval, test_config.plot)
    for name, value in name_value:
        logging.info('Epoch[%d] - Validation %s=%.3f', nth_epoch, name, value)

def inferNet(infer_data_loader, network, merge_hm_flip_func, merge_tag_flip_func, flip_pairs,
             patch_width, patch_height, loss_config, test_config, final_output_path, flip_test=True):

    print('in valid')
    network.eval()

    heatmaps_list = []
    tagmaps_list = []
    with torch.no_grad():
        for idx, _data in enumerate(infer_data_loader):
            batch_data = _data.cuda()

            heatmaps, tagmaps = network(batch_data)

            if flip_test:
                batch_data_flip = flip(batch_data, dims=3)
                heatmaps_flip, tagmaps_flip = network(batch_data_flip)

            if flip_test:
                heatmaps = merge_hm_flip_func(heatmaps, heatmaps_flip, flip_pairs)
                tagmaps = merge_tag_flip_func(tagmaps, tagmaps_flip, flip_pairs)

            heatmaps_list.append(heatmaps)
            tagmaps_list.append(tagmaps)

    heatmaps = torch.cat(heatmaps_list)
    tagmaps = torch.cat(tagmaps_list)

    print("in infer")
    heatmaps = nn.UpsamplingBilinear2d((patch_height, patch_width)).cuda()(heatmaps)
    tagmaps = nn.UpsamplingBilinear2d((patch_height, patch_width)).cuda()(tagmaps)
    heatmaps = heatmaps.cpu().numpy().astype(float)
    tagmaps = tagmaps.cpu().numpy().astype(float)

    imdb_list = infer_data_loader.dataset.db

    num_samples = heatmaps.shape[0]
    assert num_samples == tagmaps.shape[0]
    assert num_samples == len(imdb_list)

    ### PostProcess Path ###
    infer_result_path = os.path.dirname(final_output_path)
    print(infer_result_path)
    postProcessOutputPath = infer_result_path + '\post_process_result'
    if not os.path.exists(postProcessOutputPath):
        os.makedirs(postProcessOutputPath)

    windows_list_with_score = list()
    ############################ Group corners on TAG ##############################
    # 1. Group Corners
    parser = HeatmapParser(loss_config, test_config.useCenter, test_config.centerT, imdb_list)
    # print('imdb list[2]:')
    # print(imdb_list[2])

    ## Final Recorded coords of bounding boxes 
    recordedCoordsOfEntireSeq = []

    for n_s in range(num_samples):
        try:
            group_corners_wz_score = \
                group_corners_on_tags(n_s, parser, heatmaps[n_s], tagmaps[n_s], patch_width, patch_height,
                                      imdb_list[n_s]['im_width'], imdb_list[n_s]['im_height'],
                                      rectify = test_config.rectify, winScoreThres = test_config.windowT)
            # print('windowScores: ' + str(group_corners_wz_score))
            windows_list_with_score.append(group_corners_wz_score)
            inputImg = cv2.imread(imdb_list[n_s]['image'], 1)
            modelInferredImg = facade.plotSingleImage(inputImg, 
                group_corners_wz_score,  final_output_path, n_s)

            final_postProcess_path = os.path.join(postProcessOutputPath, str(n_s) + '.png') 
            print(final_postProcess_path)
            postProcess = PostProcess(modelInferredImg, group_corners_wz_score, final_postProcess_path)
            boundingBoxes = postProcess.runPostProcessingModule()
            print(boundingBoxes.shape)
            recordedCoordsOfEntireSeq.append(boundingBoxes)

            # print("windowCount: " + str(len(group_corners_wz_score)))
            # postProcess = PostProcess(cv2.imread(imdb_list[n_s]['image'], 1), group_corners_wz_score)
            # postProcess.runPostProcessingModule()
        except Exception as e:
            assert 0, (n_s, e, os.path.basename(imdb_list[n_s]['image']))
            print(e, '  ', os.path.basename(imdb_list[n_s]['image']))

    # 2. Infer or Evaluate
    # print("windows list:")
    # print(windows_list_with_score)
    # imdb_list_1 = facade.plot(windows_list_with_score, imdb_list, final_output_path)
    
    # print('DEBUG INPUT PP:', imdb_list)
    # print('DEBUG imdb_list_1:', imdb_list_1)
    # print(type(imdb_list))

    # for n_s in range(num_samples):
    #     group_corners_wz_score = windows_list_with_score[n_s]
    #     print('windowScores: ' + str(group_corners_wz_score))
    #     print("windowCount: " + str(len(group_corners_wz_score)))
    #     print(mapInputForPostProcessing(group_corners_wz_score))
    #     fileName = str(n_s) + '.png'
    #     final_postProcess_path = os.path.join(postProcessOutputPath, fileName) 
    #     print(final_postProcess_path)
    #     postProcess = PostProcess(imdb_list_1[n_s]['image'], group_corners_wz_score, final_postProcess_path)
    #     boundingBoxes = postProcess.runPostProcessingModule()
    #     print(boundingBoxes.shape)
    #     recordedCoordsOfEntireSeq.append(boundingBoxes)

    print(recordedCoordsOfEntireSeq)
    ## Height info of sequence in cm
    height_info = [60, 110, 230, 330, 380, 420, 450, 480, 510, 530, 560, 580, 590, 600, 610, 670, 750, 780, 800, 850, 910, 950, 1000]
    calculateTotalParams = CalculateTotalParams(recordedCoordsOfEntireSeq, height_info)
    calculateTotalParams.runCalculateTotalParamsModule()
    
    # with open('coordinatesFromPostProcessing-1-shufflenet.csv', 'ab') as f:
    #     for array in recordedCoordsOfEntireSeq:
    #         array = array.ravel()
    #         array = array.reshape(-1, array.shape[0])
    #         np.savetxt(f, array, delimiter=',')