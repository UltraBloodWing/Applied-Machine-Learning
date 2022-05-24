# ML-Project
Machine learning project for creating a face recognition based attendance system.

# Methods used
## Classification
- Mobilenet model used for face recognition
- Multiclass Classification Model

## Deep Metric Learning
- Siamese Network with contrastive loss used for face recognition

## Program Preview
### Login area
![plot](./screenshots/login_area.png)
### Face verification area after user is logged in
![plot](./screenshots/face_verification_area.png)
### Deteced unknown face, prompts user to check uploaded image
![plot](./screenshots/unknown_face.png)
### Face identified and verified with attendance being taken
![plot](./screenshots/face_verified_attendance_taken.png)

## How to use:
To use the main program, read the README file in the main_program folder and follow the steps inside.

Ensure that the requirements are installed using: pip install -r requirements.

And to run the program, open a terminal to the folder directory and run this command: streamlit run app.py
