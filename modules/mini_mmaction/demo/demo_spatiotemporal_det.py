# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import tempfile

import os
import sys
from common.is_aarch_64 import is_aarch64

# if is_aarch64():
#     sys.path.append('/home/agx123/DS_pipeline_new/mini_mmaction')

sys.path.append('./modules/mini_mmaction')

import cv2
import mmcv
import mmengine
import numpy as np
import torch
from mmengine import DictAction
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData

from mmaction.apis import detection_inference
from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.utils import frame_extract

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

cur_dir=os.getcwd()
sys.path.append(cur_dir)


# args_video = cur_dir + "/demo/demo.mp4"
args_short_side = 256
args_config = cur_dir + '/modules/mini_mmaction/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py'
args_cfg_options = {}
args_predict_stepsize = 8
args_label_map = cur_dir + "/modules/mini_mmaction/tools/label_map.txt"
args_det_config = cur_dir + "/modules/mini_mmaction/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py"
# args_det_checkpoint =  "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
args_det_score_thr = 0.9
args_det_cat_id = 0
args_device =  "cuda:0"
args_checkpoint = cur_dir + "/models_deepstream/best_mAP_overall_epoch_12.pth"
args_action_score_thr = 0.5 
args_output_stepsize = 4 
tmp_dir = tempfile.TemporaryDirectory()
config = mmengine.Config.fromfile(args_config)
config.model.backbone.pretrained = None
model = MODELS.build(config.model)

load_checkpoint(model, args_checkpoint, map_location='cpu')
model.to(args_device)
model.eval()


FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = '004b23-006400-007200-008000-38b000-70e000'
plate_green = plate_green.split('-')
plate_green = [hex2color(h) for h in plate_green]

#selecting 56 frames from entire video(280 frames), results of all batches(28) , list of listss of ids(28)  
def visualize(frames, annotations, obj_id_ref, plate=plate_blue, max_num=5):
    act_batch_res = {}
    # print(obj_id_ref)
    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_out = cp.deepcopy(frames)
    nf, na ,ni= len(frames), len(annotations), len(obj_id_ref)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])
    for i in range(na):
        anno = annotations[i]
        idds = obj_id_ref[i]

        if anno is None:
            continue

        for j in range(nfpa):
            ind = i * nfpa + j
            # print(i , nfpa , j)
            # print(ind)
            frame = frames_out[ind]
            # print(len(anno),len(idds))
            for iddx,ann in zip(idds,anno):
                box = ann[0]
                label = ann[1]
                
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(frame, st, ed, plate[0], 2)
                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = abbrev(lb)
                    text = ': '.join([text, str(score[k])])
                    text = str(iddx)+" " + text 
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                            THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)
                    act_batch_res[iddx] = lb
                    
                # cv2.imwrite("out.jpg",frame)
    print("ACTIVITY RESULT: ", act_batch_res)
    return frames_out,act_batch_res

def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.
    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}


def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name

#process a batch on a call
def pack_result(human_detection, result, img_h, img_w):

    """Short summary.
    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.
    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    # print("len(result) ",len(result))
    # print("len(human_detection) ",len(human_detection))
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None


    for prop, res in zip(human_detection, result):
        # print("prop",prop)
        # print("res",res)
        # print("len(res)",len(res))
        # print("___________________________")

        res.sort(key=lambda x: -x[1])
        # print((prop.data.cpu().numpy(), [x[0] for x in res], [x[1] for x in res]))
        #bbox,activities,confidence
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1] for x in res]))
        # print("___________________________")
    
    return results



#python3 demo/demo_spatiotemporal_det.py demo/demo.mp4 demo/demo_spatiotemporal_det.mp4
#python3 demo/demo_spatiotemporal_det.py demo/demo.mp4 demo/demo_spatiotemporal_det.mp4 --config configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py --checkpoint https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py  --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth --det-score-thr 0.9 --action-score-thr 0.5 --label-map tools/label_map.txt --predict-stepsize 8 --output-stepsize 4 --output-fps 6 
def activity_main(original_frames, human_detections_, obj_id_ref):

    # args = parse_args()



    # frame_paths, original_frames = frame_extract(
    #     args_video, out_dir=tmp_dir.name)
    num_frame = len(original_frames)
    h, w, _ = original_frames[0].shape
    
    # resize frames to shortside
    new_w, new_h = mmcv.rescale_size((w, h), (args_short_side, np.Inf))
    frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
    w_ratio, h_ratio = new_w / w, new_h / h

    config.merge_from_dict(args_cfg_options)
    val_pipeline = config.val_pipeline
    # print(val_pipeline)
    sampler = [x for x in val_pipeline if x['type'] =="SampleAVAFrames"][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    # Note that it's 1 based here
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args_predict_stepsize)
    # print(timestamps)
    # Load label_map
    label_map = load_label_map(args_label_map)
    try:
        if config['data']['train']['custom_classes'] is not None:
            label_map = {
                id + 1: label_map[cls]
                for id, cls in enumerate(config['data']['train']
                                         ['custom_classes'])
            }
    except KeyError:
        pass

    # Get Human detection results
    # center_frames = [original_frames[ind - 1] for ind in timestamps]

    # human_detections, _ = detection_inference(args_det_config,
    #                                           args_det_checkpoint,
    #                                           center_frames,
    #                                           args_det_score_thr,
    #                                           args_det_cat_id, args_device)

    torch.cuda.empty_cache()
    for i in range(len(human_detections_)):
        det = human_detections_[i]
        det[:, 0:4:2] *= w_ratio
        det[:, 1:4:2] *= h_ratio
        human_detections_[i] = torch.from_numpy(det[:, :4]).to(args_device)

    # Build STDET model
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        config['model']['test_cfg']['rcnn'] = dict(action_thr=0)
        # print('config:', config)
    except KeyError:
        pass


    predictions = []

    img_norm_cfg = dict(
        mean=np.array(config.model.data_preprocessor.mean),
        std=np.array(config.model.data_preprocessor.std),
        to_rgb=False)

    print('Performing SpatioTemporal Action Detection for each clip')
    #select specific frames for inference from list of frame's bbox[[frame1 bboxs],[frame2 bboxs]...](central frame from each batch)
    human_detections = [human_detections_[every-1] for every in timestamps]
    #selecting ids from sll the frames based on selected frames 
    final_id_lst = [obj_id_ref[every-1] for every in timestamps]
    # print(len(human_detections),len(final_id_lst) )
    # print(len(human_detections[0]),len(final_id_lst[0]) )


    # print(final_id_lst)
    # print(len(final_id_lst), len(human_detections))
    assert len(timestamps) == len(human_detections)    
    prog_bar = mmengine.ProgressBar(len(timestamps))
    loop = 0
    full_idxs = []
    #loop over short video clips
    for timestamp, proposal,idd in zip(timestamps, human_detections,final_id_lst):
    #len of model output depends on number of proposal
        idxs = []
        # print("------------------------------------------------------------------------------")
        # print("PROPOSAL:",proposal)
        # print(len(proposal))
        loop = loop +1
        if proposal.shape[0] == 0:
            # print("if")
            predictions.append(None)
            full_idxs.append(None)
            # print(full_idxs)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval

        frame_inds = start_frame + np.arange(0, window_size, frame_interval)

        frame_inds = list(frame_inds - 1)

        imgs = [frames[ind].astype(np.float32) for ind in frame_inds]

        for i in range(start_frame-1, frame_inds[-1]):
            idxs.append(obj_id_ref[i])
        full_idxs.append(idxs)
        # print(len(idxs))
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        # print("input_array ",input_array)

        input_tensor = torch.from_numpy(input_array).to(args_device)
        # print("input_tensor ",input_tensor)

        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=proposal)
        # print("DATASAMPLE PROPOSALS:",datasample.proposals)
        datasample.set_metainfo(dict(img_shape=(new_h, new_w)))
        with torch.no_grad():
            result = model(input_tensor, [datasample], mode='predict')
            # print(result)
            scores = result[0].pred_instances.scores
            # print(len(scores))
            # print(len(proposal))

            prediction = []
            # N proposals
            for i in range(proposal.shape[0]):
                prediction.append([])
            # Perform action score thr
            for i in range(scores.shape[1]):
                if i not in label_map:
                    continue
                for j in range(proposal.shape[0]):
                    if scores[j, i] > args_action_score_thr:
                        prediction[j].append((label_map[i], scores[j,
                                                                   i].item()))
            # print("\n",len(idxs),len(prediction))
            # print(prediction)
            # print(len(prediction))
        


            predictions.append(prediction)
            print("------------------------------------------------------------------------------")
            
        prog_bar.update()

    #activity results for entire video is ready at this stage(28 batches for 280 frame video)
    # print("\nFULL ID",len(full_idxs))
            
    # print("len(predictions)",len(predictions))
    # print('human_detections', len(human_detections))

    results = []

    #looping over activity results of each batch
    for human_detection, prediction in zip(human_detections, predictions):
        # print(human_detection)
        # if len(human_detection) > 0:
        #     print("len(human_detection) ",len(human_detection))
        #     print("len(prediction) ",len(prediction))
        # else:
        #     print("no face detected")

        #bbox,activities,confidence
        results.append(pack_result(human_detection, prediction, new_h, new_w))
    
    #at this point results of all batched are organised as bbox,activities,confidence and appended into a list
    # print('results', len(results))

    def dense_timestamps(timestamps, n):
        """Make it nx frames."""
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int64)

    dense_n = int(args_predict_stepsize / args_output_stepsize)
    frames = [
        original_frames[i - 1]
        for i in dense_timestamps(timestamps, dense_n)
    ]
    # print(frames)
    # print(len(frames))

    #selecting 56 frames from entire video(280 frames), results of all batches(28) , list of listss of ids(28)  
    vis_frames,act_batch_res = visualize(frames, results, final_id_lst)

    # # print(vis_frames)
    # # cv2.imwrite("out.jpg",vis_frames)
    # vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
    #                             fps=25)
    # vid.write_videofile(args_out_filename, fps=25)
    tmp_dir.cleanup()
    return act_batch_res

# # if __name__ == '__main__':
# #     activity_main()
