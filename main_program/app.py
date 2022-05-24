import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt 
import pandas as pd
import csv
from datetime import datetime
from tensorflow import keras
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from keras.applications.mobilenet import preprocess_input

menu = ['Login', 'Sign Up']
menu_choice = st.sidebar.selectbox('Menu', menu)

def attendance(user_filename, user):
    choice = st.selectbox("Select Option",[
        "Face Verification"
    ])

    fig = plt.figure()

    if choice == "Face Verification":
        column1, column2 = st.columns(2)
    
        with column1:
            image1 = Image.open('./database_images/' + user_filename)
            st.markdown('**User Image:**')
            column1.image(image1)
        
        with column2:
            image2 = st.file_uploader("Select File", type=["jpg","png", "jpeg"])
        if (image1 is not None) & (image2  is not None):
            st.markdown("""---""")
            st.markdown('**Results:**')

            col1, col2 = st.columns(2)
            image2 =  Image.open(image2)
            with col1:
                st.image(image1)
            with col2:
                st.image(image2)

            filenames = [image1,image2]

            faces = [extract_face(f) for f in filenames]
            samples = asarray(faces, "float32")
            samples = preprocess_input(samples)
            # load the mobilenet model
            model = keras.models.load_model('./retrained_mobilenet.h5')
            # perform prediction
            embeddings = model.predict(samples)
            thresh = 0.5
            score = cosine(embeddings[0], embeddings[1])
            if score <= thresh:
                st.success( " >face is a match (%.3f <= %.3f) " % (score, thresh))

                now = datetime.now()
                attendance = [user, now.strftime("%m/%d/%Y, %H:%M:%S")]

                with open('attendance.csv', 'a', newline='') as f:
                    fileOut = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
                    fileOut.writerow(attendance)

                st.success('Attendance Taken!')
            else:
                st.error(" >face is NOT a match (%.3f > %.3f)" % (score, thresh))
                st.error('Could not verify user! Check uploaded image...')

def main():

    if menu_choice == 'Login':
        st.sidebar.title('Face Attendance')
        st.sidebar.subheader('Login')

        username = st.sidebar.text_input('Username')
        password = st.sidebar.text_input('Password')

        if st.sidebar.checkbox('Log In'):
            df = pd.read_csv('users.csv', sep=',')

            if username in df['Username'].tolist():
                userFilename = df[df['Username'] == username]['Filename'].values[0]
                attendance(userFilename, username)
            else:
                st.sidebar.warning('Incorrect Username/Password. Check credentials or Sign Up')


    elif menu_choice == 'Sign Up':
        st.sidebar.subheader('Create New Account')
        new_user = st.sidebar.text_input('Username')
        new_password = st.sidebar.text_input('Password')
        new_image = st.sidebar.file_uploader("Upload User Image", type=["jpg","png", "jpeg"])


        if st.sidebar.button('Sign Up'):
            if new_image is not None:
                with open(('./database_images/'+ new_image.name), 'wb') as f:
                    f.write((new_image).getbuffer())



            df = pd.read_csv('users.csv', sep=',')

            if new_user in df['Username'].tolist():
                st.sidebar.warning('Username not available')
            else:
                data = {
                    'Username': [new_user],
                    'Password': [new_password],
                    'Filename': [new_image.name]
                }    

                df = pd.DataFrame(data)
                df.to_csv('users.csv', mode='a', index=False, header=False)

                st.sidebar.success('You have successfully created an account!')
                st.sidebar.info('Go to Login menu to sign in')
    

    

def extract_face(file):
    pixels = asarray(file)
    plt.axis("off")
    plt.imshow(pixels)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]["box"]
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = asarray(image)
    return face_array

if __name__ == "__main__":
    main()