import os
import time
import torch
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
data = coco.dataset
annotations = data['annotations']

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


# Grabs inference data
# -----------------------------------------------------------------------------
test_folder_path = os.path.join(DATASET_PATH, "test")
infererence_data = []

for image_name in os.listdir(test_folder_path):
    image_path = os.path.join(test_folder_path, image_name)
    
    if "annotation" in image_path:
        continue
    
    # Load Image with PIL
    img = cv2.imread(image_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(image)
    
    # Convert to JPEG Buffer
    buffered = io.BytesIO()
    pilImage.save(buffered, quality=100, format="JPEG")
    
    # Construct the URL
    upload_url = "".join([
        "https://detect.roboflow.com/blood-cell-detection-1ekwu/1",
        "?api_key=umichXAeCyw6nlBsDZIt",
        "&confidence=0"
    ])
    
    # Build multipart form and post request
    m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
    
    response = requests.post(upload_url, 
                             data=m, 
                             headers={'Content-Type': m.content_type},
                             )
    
    predictions = response.json()['predictions']
    
    for prediction in predictions:
        prediction.update({'image_name': image_name})
        infererence_data.append(prediction)
# -----------------------------------------------------------------------------



# Confidence Change Section
# -----------------------------------------------------------------------------
most_accurate_confidence_score = 0
previous_accuracy = 0

for confidence_score_addition in range(100-int(MIN_CONFIDENCE_SCORE*100)+1):
    
    new_confidence_score = round((MIN_CONFIDENCE_SCORE+(confidence_score_addition/100)), 2)
    
    # filtered_infererence_data = [data_point for data_point in infererence_data 
    #                              if data_point['confidence'] >= new_confidence_score]

    # Inference Section
    # -----------------------------------------------------------------------------
    inference_count_rbc_list = []
    inference_count_rbc = 0
    previous_inference_image_name = infererence_data[0]['image_name']
    
    for infererence_data_point in infererence_data:
        
        inference_image_name = infererence_data_point['image_name']
        inference_label = infererence_data_point['class']
        inference_score = infererence_data_point['confidence']
        
        if inference_image_name != previous_inference_image_name:
            inference_count_rbc_list.append(inference_count_rbc)
            inference_count_rbc = 0
        
        if inference_label == 'RBC' and inference_score > new_confidence_score:
            inference_count_rbc += 1
        
        previous_inference_image_name = inference_image_name
    
    
    inference_count_rbc_list.append(inference_count_rbc)
    # -----------------------------------------------------------------------------
    
    
    # Accuracy Test
    # -----------------------------------------------------------------------------
    accuracy_list = []
    
    for inference_count_index, inference_count in enumerate(inference_count_rbc_list):
        if inference_count == count_rbc_list_1[inference_count_index]:
            accuracy_list.append(True)
        else:
            accuracy_list.append(False)
    
    accuracy = accuracy_list.count(True)/len(accuracy_list)
    # -----------------------------------------------------------------------------
    
    
    if accuracy > previous_accuracy:
        most_accurate_confidence_score = new_confidence_score
        max_accuracy = accuracy
    
    # if confidence_score_addition % 5 == 0:
    #     print(confidence_score_addition)
    

print("Confidence score with highest accuracy between test dataset and inference:", most_accurate_confidence_score)
print("Accuracy with confidence score of {}:".format(most_accurate_confidence_score), round(max_accuracy*100,2), "%")
# -----------------------------------------------------------------------------

# =============================================================================


print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)
