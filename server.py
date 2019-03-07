import merge_similarity_inference
from flask import Flask, flash, request, jsonify, render_template, send_from_directory, redirect, url_for
import os
import glob
from werkzeug import secure_filename
app = Flask(__name__, static_folder='',template_folder='demo')

UPLOAD_FOLDER = "data/uploads/"

root = ''

@app.route('/<path:path>', methods=['GET'])
def static_proxy(path):
    return send_from_directory(root, path)

@app.route('/',methods=['GET'])
def query_file():
    vid = request.args.get('filename', None)
    k = request.args.get('k', None)
    assert int(k), "k is not an integer"
    if vid:
        if k:
            search_results = predict(int(k),"data/UCF101/"+vid)
        else:
            search_results = predict("data/UCF101/"+vid)
        print(vid,k)
        print(search_results)
        return render_template("index.html",vid_input=vid, similar_vids=search_results)
    else:
        print("Error")
        return render_template("index.html")

def predict(k=5,video_path=None):
    if not video_path:
        print("Oops, no video path.")
    prediction = merge_similarity_inference.merge_similarity_ucf_video(video_path, k=k, verbose=True ,newVid=False)
    return prediction

@app.route('/uploader',methods=['GET','POST'])
def upload_vid():
    if request.method == 'POST':
        try:
            if request.form:
                pass
        except:
            flash('No k value given. Assuming 5')
        try:
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            for fname in glob.iglob(UPLOAD_FOLDER+'/*'):
                os.remove(fname)
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))
            return redirect(url_for('query_file',
                                filename=file.filename, k=int(request.form['k'])))
        except:
            return "Error, You didn't choose a file"
            redirect(request.url.split("/")[0])
    return "Error, You didn't choose a file"

if __name__ == '__main__':
    app.run(port=5000, debug=False)
