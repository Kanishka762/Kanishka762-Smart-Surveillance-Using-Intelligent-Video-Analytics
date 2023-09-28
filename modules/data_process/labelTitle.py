from modules.components.load_paths import *
import ast
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path)

anomaly_objects = ast.literal_eval(os.getenv("anamoly_object"))
anomaly_activities = ast.literal_eval(os.getenv("anamoly"))
anomaly_members = ast.literal_eval(os.getenv('anamolyMemberCategory'))
track_type = ast.literal_eval(os.getenv('track_type'))

def create_title_dict():
    try:
        return {
                'anomalyActivity': set(),
                'detectionActivity': set(),
                'anomalyObject': set(),
                'detectionObject': set(),
                'anomalyMember': set(),
                'detectionMember': set(),
            }
    except Exception as e:
        print("An error occurred while creating title dictionary:", e)

def update_title_dict(initial_dict, output_json):
    try:
        for obj in output_json['metaData']['object']:
            activity = obj['activity']
            object_class = obj['class']
            member_type = obj.get('track')

            if activity != 'No Activity':
                key = 'anomalyActivity' if activity in anomaly_activities else 'detectionActivity'
                initial_dict[key].add(activity)

            key = 'anomalyObject' if object_class in anomaly_objects else 'detectionObject'
            initial_dict[key].add(object_class)

            if member_type is not None:
                key = 'anomalyMember' if member_type in anomaly_members else 'detectionMember'
                initial_dict[key].add(member_type)

        return initial_dict
    except Exception as e:
        print("An error occurred while updating title dictionary:", e)

def fetch_member_category(member_types):
    try:
        return [track_type[member_type] for member_type in member_types if member_type not in {'100', '10', '11'}]
    except Exception as e:
        print("An error occurred while fetching member category:", e)

def create_batch_title(result_dict):
    try:
        if result_dict['anomalyObject']:
            return " ".join(result_dict['anomalyObject']) + " detected"
        elif result_dict['anomalyActivity'] or result_dict['anomalyMember']:
            activities = result_dict['anomalyActivity']
            member_categories = fetch_member_category(result_dict['anomalyMember'])
            return f"{' '.join(member_categories)} member detected {' '.join(activities)}"
        elif result_dict['detectionMember']:
            member_categories = fetch_member_category(result_dict['detectionMember'])
            return f"{' '.join(member_categories)} member detected"
        elif result_dict['detectionObject']:
            return "People detected"
        else:
            return None
    except Exception as e:
        print("An error occurred while creating batch title:", e)

def fetch_batch_title(output_json):
    try:
        initial_dict = create_title_dict()
        update_dict = update_title_dict(initial_dict, output_json)
        result_dict = {key: list(value) for key, value in update_dict.items()}
        title = create_batch_title(result_dict)
        output_json['metaData']['title'] = title
        return output_json
    except Exception as e:
        print("An error occurred while fetching batch title:", e)