import requests
url = 'http://localhost:5000/api'
r = requests.post(url,json={'vid_path':'data/UCF101/v_ApplyEyeMakeup_g01_c01.avi','k':10})
print(r.json())
