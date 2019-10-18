import face_recognition
import json
import base64
import io
import pickle
import requests
import time
import numpy as np
from flask import Flask, jsonify, request, redirect
from PIL import Image

app = Flask(__name__)

encodingpath = 'encoding_faces.pickle'
data = pickle.loads(open(encodingpath, 'rb').read())

@app.route('/process', methods=['POST'])
def process_faces():
	faces_names = []
	content = request.json
	list_faces = json.loads(content)
	counts = {}
	for face in list_faces:
		bytes_face = bytes(face, 'utf-8')
		imgdecode = base64.decodebytes(bytes_face)
		img = Image.open(io.BytesIO(imgdecode))
		image = np.array(img)
		h,w,_ = image.shape
		box = [(0, w, h, 0)]
		#print(box)
		unknow_face_encodings = face_recognition.face_encodings(image,box)
		#print(len(unknow_face_encodings))
		match_results = face_recognition.compare_faces(data["encodings"], unknow_face_encodings[0])
		print(match_results)
		name = ""
		if all(match_results) is True:
			continue
		if True in match_results:
			matchedId = [i for (i,b) in enumerate(match_results) if b]
			#counts = {}
			for i in matchedId:
				name = data["names"][i]
				#print(name)
				counts[name] = counts.get(name, 0) + 1
			name = max(counts, key=counts.get)
		else:
			name = "unknown"
			counts[name] = counts.get(name, 0) + len(match_results)
		faces_names.append(name)
		print(counts)

	print(faces_names)
	facename = max(counts, key=counts.get)
	print(facename)
	return jsonify({"name":str(facename)})

@app.route('/train', methods=['POST'])
def train_data():
	content = request.json
	list_faces = json.loads(content)


if __name__=='__main__':
	app.debug = True
	app.run(host='0.0.0.0', port=5001)
