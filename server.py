# importing modules
from flask import Flask, render_template, jsonify, request
import numpy as np
import cv2
import tensorflow as tf


# configs and vars
app = Flask(__name__)
disease_type = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]



# routes from here -=====================-===============------=======

# bad request page error handeler --------------------------->>>>>>>>>>>>>>>>----------
@app.errorhandler(404) 
def not_found(e): 
  return jsonify({'error' : str(e)})
# ------------------------------>>>>>>>>>>>>>>>>>>-------------------------





# static --------------------->>>>>>>>>>>>>>--------------------
@app.route("/")
def welcome():
    return render_template("welcomepage.html")                 # home page

@app.route("/detectdiseasepage")
def detectdiseasepage():        
    return render_template("diseasedetection.html")            # high priority page



@app.route("/cropprediction1")
def cropprediction1():
    return render_template("cropprediction1.html")        

@app.route("/cropprediction2")
def cropprediction2():
    return render_template("cropprediction2.html") 

@app.route("/cropprediction3")
def cropprediction3():
    return render_template("cropprediction3.html") 

@app.route("/cropprediction4")
def cropprediction4():
    return render_template("cropprediction4.html") 

@app.route("/cropprediction5")
def cropprediction5():
    return render_template("cropprediction5.html")   



@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html") 

@app.route("/studyarea")
def studyarea():
    return render_template("studyarea.html") 

@app.route("/technologyused")
def technologyused():
    return render_template("technologyused.html") 

@app.route("/whatwedo")
def whatwedo():
    return render_template("whatwedo.html") 

# ---------------------------->>>>>>>>>>>>>>>>>>>>--------------





#ML -------------------------->>>>>>>>>>>>>>>>>>>>----------------------------

# detect disease of the crop
@app.route("/detectdisease", methods=["GET", "POST"])
def detect_disease():
    print(request)
    img = request.files["image"].read()                         # getting the image file
    npimg = np.fromstring(img, np.uint8)                        # convert string file obj. to np array
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)             # convert np array to image
    re_img = cv2.resize(img, (200, 200)) / 255.0                        # resize the image to desired size
    re_tensor = re_img.reshape(1, 200, 200, 3)                  # reshape the image

    model = tf.keras.models.load_model("CNN_model.h5")     # loads the model
    one_hot = model.predict(re_tensor)                              # predictions predictions predictions
    print(one_hot)
    return jsonify({"Disease Detected" : disease_type[int(np.argmax(one_hot, axis=1))]})      # returned the fucking disease
    


# predict crops
@app.route("/croppredict")
def croppredict():
    return "Nothing to show here"
# ---------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>----------------------------------------


# main
if __name__ == "__main__": app.run(debug = True)