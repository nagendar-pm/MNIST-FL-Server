from flask import Flask, request, send_file
from Model_handling_CNN import CNN_2, Model
import json, pyrebase

app = Flask(__name__)
config = {
  "apiKey": "AIzaSyCzct6mzQ0wc8-8iR6tYksMKnFD2JuUrYU",
  "authDomain": "test-4acc6.firebaseapp.com",
  "databaseURL": "https://test-4acc6.firebaseio.com",
  "projectId": "test-4acc6",
  "storageBucket": "test-4acc6.appspot.com",
  "messagingSenderId": "435925488227",
  "appId": "1:435925488227:web:3c9792d7e86412db2cde56"
}
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

model = Model()


@app.route('/')
def home():
    return '<h1>Federated Learning is an interesting area!</h1>'


@app.route('/update', methods=['GET'])
def update():
    if request.method == 'GET':
        model.save()
        print('Sending the current version of the model!')
        result = send_file('Models/updated_model.pt', as_attachment=True)
        return result


@app.route('/train', methods=['POST', 'GET'])
def train():
    if request.method == 'POST':
        request_data = request.get_json()
        # print(str(request_data))
        input = request_data['prediction']
        label = request_data['label']
        uuid = request_data['UUID']
        print(uuid)
        prediction, outputs = model.predict_img(input)
        if int(label) != int(prediction):
            model.backward(outputs, int(label))
        return str(prediction)
    else:
        return '<h2>Sumi masen!!</h2>'


if __name__ == '__main__':
    app.run(debug=True, port=5000)



# import pyrebase
#
# config = {
#   "apiKey": "AIzaSyCzct6mzQ0wc8-8iR6tYksMKnFD2JuUrYU",
#   "authDomain": "test-4acc6.firebaseapp.com",
#   "databaseURL": "https://test-4acc6.firebaseio.com",
#   "projectId": "test-4acc6",
#   "storageBucket": "test-4acc6.appspot.com",
#   "messagingSenderId": "435925488227",
#   "appId": "1:435925488227:web:3c9792d7e86412db2cde56"
# }
#
# firebase = pyrebase.initialize_app(config)
# storage = firebase.storage()
#
# cloud_path = 'Models/cnn_2.pt'
# local_path = 'Models/modelB.pt'
# storage.child(cloud_path).put(local_path)
