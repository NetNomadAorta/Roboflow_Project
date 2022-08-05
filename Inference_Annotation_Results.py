import io
import cv2
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import os
import time


# User parameters
MIN_CONFIDENCE_SCORE = 0.40 # Confidence score ranging from 0 to 1
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

test_folder_path = os.path.join(DATASET_PATH, "test")

count_rbc_list = []

for image_name in os.listdir(test_folder_path):
    image_path = os.path.join(test_folder_path, image_name)
    
    if "annotation" in image_path:
        continue
    
    count_rbc = 0
    
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
        
        if "RBC" in label and confidence_score > MIN_CONFIDENCE_SCORE:
            count_rbc += 1
    
    count_rbc_list.append(count_rbc)

# =============================================================================


print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)