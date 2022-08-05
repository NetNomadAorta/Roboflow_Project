import os
import time
import torch
# remove warnings (optional)
import warnings
warnings.filterwarnings("ignore")
from pycocotools.coco import COCO
import io
import cv2
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder


# User parameters
MIN_CONFIDENCE_SCORE = 0.30 # Confidence score ranging from 0 to 1 (THIS IS WHAT TO START WITH)
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

# Test Dataset Section
# -----------------------------------------------------------------------------
previous_image_id = 0
count_rbc_1 = 0
count_rbc_list_1 = []

for id_index, info in enumerate(annotations):
    image_id = info['image_id']
    category_id = info['category_id']
    
    if image_id != previous_image_id:
        count_rbc_list_1.append(count_rbc_1)
        count_rbc_1 = 0
    
    if category_id == 2:
        count_rbc_1 += 1
    
    previous_image_id = image_id

count_rbc_list_1.append(count_rbc_1)
# -----------------------------------------------------------------------------

most_accurate_confidence_score = 0
previous_accuracy = 0

# Confidence Change Section
for confidence_score_addition in range(100-int(MIN_CONFIDENCE_SCORE*100)+1):

    # Inference Section
    # -----------------------------------------------------------------------------
    test_folder_path = os.path.join(DATASET_PATH, "test")
    
    count_rbc_list_2 = []
    
    for image_name in os.listdir(test_folder_path):
        image_path = os.path.join(test_folder_path, image_name)
        
        if "annotation" in image_path:
            continue
        
        count_rbc_2 = 0
        
        # Load Image with PIL
        img = cv2.imread(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(image)
        
        # Convert to JPEG Buffer
        buffered = io.BytesIO()
        pilImage.save(buffered, quality=100, format="JPEG")
        
        # Build multipart form and post request
        m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
        
        response = requests.post("https://detect.roboflow.com/blood-cell-detection-1ekwu/1?api_key=umichXAeCyw6nlBsDZIt", data=m, headers={'Content-Type': m.content_type})
        
        predictions = response.json()['predictions']
        
        for prediction in predictions:
            label = prediction['class']
            confidence_score = prediction['confidence']
            
            if "RBC" in label and confidence_score > (MIN_CONFIDENCE_SCORE+(confidence_score_addition/100)):
                count_rbc_2 += 1
        
        count_rbc_list_2.append(count_rbc_2)
    # -----------------------------------------------------------------------------
    
    
    # Accuracy Test
    # -----------------------------------------------------------------------------
    accuracy_list = []
    
    for inference_count_index, inference_count in enumerate(count_rbc_list_2):
        if inference_count == count_rbc_list_1[inference_count_index]:
            accuracy_list.append(True)
        else:
            accuracy_list.append(False)
    
    accuracy = accuracy_list.count(True)/len(accuracy_list)
    # -----------------------------------------------------------------------------
    
    
    if accuracy > previous_accuracy:
        most_accurate_confidence_score = (MIN_CONFIDENCE_SCORE+(confidence_score_addition/100) )
        max_accuracy = accuracy
    
    if confidence_score_addition % 5 == 0:
        print(confidence_score_addition)
    

print("Confidence score with highest accuracy between test dataset and inference:", most_accurate_confidence_score)
print("Accuracy with confidence score of {}:".format(most_accurate_confidence_score), round(max_accuracy*100,2), "%")
# =============================================================================


print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)
