from modules.components.load_paths import *
from dotenv import load_dotenv
load_dotenv(dotenv_path)

import ast
import os

classDict = ast.literal_eval(os.getenv("classDict"))
obj_det_labels = ast.literal_eval(os.getenv("obj_det_labels"))


def findClassList(subscriptions):
    subscriptions_class_list = [item for sublist in [classDict[each] for each in subscriptions if each in classDict] for item in sublist]
    subscriptions_class_list.extend(iter(obj_det_labels))
    return subscriptions_class_list
    