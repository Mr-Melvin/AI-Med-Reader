# AI-Med-Reader
AI-Med-Reader is a personal medication assistant that utilizes computer vision and deep learning to identify brand names of medicines, providing instant identification and explanation of their common use.


# AI-Med-Reader Technology Stacks:
  1) YOLOv8 :
               Utilising a customised dataset, a YOLOv8 model was effectively trained to identify the pharmaceutical brand name section on medication strips. In order to provide the groundwork                 for additional processing and analysis, this model is especially made to recognise and find the region of interest on the strip that contains the brand name of the medication.
     
  2) OpenCV :
               The OpenCV package is used to make jobs involving image processing and computer vision easier. worked with OpenCV to accomplish tasks like processing webcam footage,                       cropping, and thresholding.

  3) EasyOCR :
               Text from cropped pictures of medication strips can be extracted with EasyOCR.

  4) Generative AI :
Gemini is used by AI-Med-Reader. The extracted words are given to the Gemini after the medicine brand name has been taken out of the picture. It will produce an explanation that is understandable, succinct, and easy to use.

5) Google Translate :
Google Translate can be used to effortlessly convert Gemini's results into the user's preferred language. Better understanding is promoted by the integrated translation capacity, which guarantees that consumers can obtain information about commom use of medicine in their own language.

Created an interactive and user-friendly interface for this project using Streamlit as the frontend framework.


# How it Works :

Step 1:
    Homepage,the AI-Med-Reader begins scanning the medicine name as soon as the allow button is pressed.
    <img width="1460" alt="s1" src="https://github.com/user-attachments/assets/e5ddf3ca-0c04-407f-a4fd-3461d5defb45">


Step 2:
    AI-Med-Reader found the medication name after scanning it.
    <img width="1458" alt="s2" src="https://github.com/user-attachments/assets/51d34a33-969e-4864-a8f3-66bfa3cab9b3">


Step 3: 
    Here,AI-Med-Reader provided an answer.
    <img width="1457" alt="s3" src="https://github.com/user-attachments/assets/9a335e64-6223-46b4-8652-1755559cf62d">


Step 4:
    Language is translated by AI-Med-Reader to the user's preference.
    <img width="1458" alt="s4" src="https://github.com/user-attachments/assets/c30ee4d7-c932-4299-ba92-6cf40cd7e8b7">



