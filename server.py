import merge_similarity_inference
from flask import Flask, request, jsonify
app = Flask(__name__)
import json

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = merge_similarity_inference.merge_similarity_ucf_video(data['vid_path'], k=data['k'], verbose=True)
    # print(jsonify(similar_videos=prediction))
    return jsonify(similar_videos=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=False)
