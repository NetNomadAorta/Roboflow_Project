import os
import time
import torch
from pycocotools.coco import COCO


# User parameters
DATASET_PATH   = "./Dataset/"



def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )



# Main()
# =============================================================================

# Starting stopwatch to see how long process takes
start_time = time.time()

torch.cuda.empty_cache()


dataset_path = DATASET_PATH



#load classes
coco = COCO(os.path.join(DATASET_PATH, "test", "_annotations.coco.json"))
categories = coco.cats
data = coco.dataset
annotations = data['annotations']
n_classes = len(categories.keys())
categories

previous_image_id = 0
count_rbc = 0
count_rbc_list = []

for id_index, info in enumerate(annotations):
    image_id = info['image_id']
    category_id = info['category_id']
    
    if image_id != previous_image_id:
        count_rbc_list.append(count_rbc)
        count_rbc = 0
    
    if category_id == 2:
        count_rbc += 1
    
    previous_image_id = image_id

count_rbc_list.append(count_rbc)

print(len(count_rbc_list))
print(count_rbc_list)
# =============================================================================





print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)
