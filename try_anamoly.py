from ava_action_list import data , object_score, anamoly_rank


# frame_info = [{2: {'type': 'Person', 'activity': 'run', 'confidence': 0.5413001179695129}}]

#sample input {1: {'type': 'Vehicle', 'activity': 'sit', 'confidence': 0.7908163070678711}

def frame_weighted_avg(frame_info_anamoly):
    activity_score_sum = 0

    weigthed_avg_numerator = 0

    for each in frame_info_anamoly:
        if [each[every] for every in each][0]['activity_score']:
            activity_score_sum = activity_score_sum + [each[every] for every in each][0]['activity_score']
            weigthed_avg_numerator = weigthed_avg_numerator + [each[every] for every in each][0]['anamoly_score'] * [each[every] for every in each][0]['activity_score']
    if weigthed_avg_numerator != 0 and activity_score_sum != 0:
        weighted_avg = weigthed_avg_numerator / activity_score_sum
    else:
        weighted_avg = 0

    return weighted_avg


def get_obj_act_score(detection):

    #get object score
    if [detection[each] for each in detection][0]['type'] in object_score:
        object_score_value = object_score[[detection[each] for each in detection][0]['type']]
        object_score_value = object_score_value*10
    else:
        object_score_value = None

    # print(object_score_value)


    #get activity score
    avail_activity_lst = [each['name'] for each in data]
    yolo_activity = [detection[each] for each in detection][0]['activity']
    activity_score_val = None
    if yolo_activity != '':
        if yolo_activity != "Unknow":
            for each in data:
                # print(each)
                if [detection[each] for each in detection][0]["activity"] in each['name']:
                    activity_score_val = anamoly_rank[str(each['id'])]
                    activity_score_val = activity_score_val*10
                    break
    return object_score_value, activity_score_val


# for detection in frame_info:
def anamoly_score_calculator(frame_info):
    # print(" ")
    # print("******************************************************************")
    frame_info_anamoly = []
    # print(frame_info)
    if len(frame_info)>0:
        for detection in frame_info:
            #get object score and activity score for a detection
            # print(detection)
            
            object_score_value, activity_score_val = get_obj_act_score(detection)
            re_id = [each for each in detection][0]
            # print(object_score_value, activity_score_val)
            # if object_score_value == 100:
            #     [detection[each] for each in detection][0]['anamoly_score'] = 100
            #     [detection[each] for each in detection][0]['activity_score'] = 100
            # else:
            if object_score_value and activity_score_val:
                #after applying confidence score to object
                # print([detection[each] for each in detection][0]['confidence'])
                # print(type([detection[each] for each in detection][0]['confidence']))
                # print(object_score_value)
                # print(type(object_score_value))
                anamoly_score = (float("{:.2f}".format(object_score_value * [detection[each] for each in detection][0]['confidence'])) + activity_score_val)  /   2
                detection_anamoly = {re_id:float("{:.2f}".format(anamoly_score))}
                [detection[each] for each in detection][0]['anamoly_score'] = float("{:.2f}".format(anamoly_score))
                [detection[each] for each in detection][0]['activity_score'] = activity_score_val


            else:
                anamoly_score = None
                [detection[each] for each in detection][0]['anamoly_score'] = anamoly_score
                [detection[each] for each in detection][0]['activity_score'] = activity_score_val
                detection_anamoly = {re_id:0}
            # if "did" in detection:
            #     [detection[each] for each in detection][0]['did'] = detection["did"]
            # detection["track_type"]
            # detection["did"]

            # "did":did,"track_type":track_type,"crops":cidd
            frame_info_anamoly.append(detection)
            
    # print(frame_info_anamoly)
    # print("******************************************************************")
    return frame_info_anamoly
        

# print(anamoly_score_calculator(frame_info))



#[{1: {'type': 'Person', 'activity': 'carry/hold', 'confidence': 0.4854480028152466, 'anamoly score': None}}, {1: {'type': 'Person', 'activity': 'carry/hold', 'confidence': 0.4854480028152466}}, {1: {'type': 'Person', 'activity': 'carry/hold', 'confidence': 0.4854480028152466}}, {1: {'type': 'Person', 'activity': 'carry/hold', 'confidence': 0.4854480028152466}}, {1: {'type': 'Person', 'activity': 'carry/hold', 'confidence': 0.4854480028152466}}, {1: {'type': 'Person', 'activity': 'carry/hold', 'confidence': 0.4854480028152466}}, {1: {'type': 'Person', 'activity': 'carry/hold', 'confidence': 0.4854480028152466}}]
#{1: {'type': 'Person', 'activity': 'sit', 'confidence': 0.8481906652450562}}

