import streamlit as st
import cv2
import textwrap
import easyocr
import pandas as pd
import google.generativeai as genai

from google_key import Google_API_Key  # Create your own Gemini API
from IPython.display import Markdown

from PIL import Image
from ultralytics import YOLO
from thefuzz import process

from googletrans import Translator,LANGUAGES
from streamlit_js_eval import streamlit_js_eval


def remove_special_chars(text):
    remove_chars = [';',',','-',' ','!','@','#','$','%','^','&','*','(',')','?']
    for c in remove_chars:
        text=text.replace(c,'')
    return text 


def fuzzy_matching(p_text,choices):
    new_text=remove_special_chars(p_text)
    result = process.extract(new_text,choices,limit=3)
    for i in range(len(result)):
        new_result = remove_special_chars(result[i][0])
        if new_text.lower() == new_result.lower():
            return result[i][0],1
        else:
            return 'Failed',0


# Convert plain text to HTML after textwrap      
def to_markdown(text):
    return Markdown(textwrap.indent(text,'>'))


# Convert first letter of each words to uppercase.
def set_of_languages():
    l = list(LANGUAGES.values())
    lang_list  = []
    for i in l:
        lang_list.append(i.capitalize())

    return lang_list


def result_translate(gmini_result,target_lang):
    translater = Translator()
    out = translater.translate(text=gmini_result,dest=target_lang)
    return to_markdown(out.text)


def language_translate(raw_text,target_lang):
    translater = Translator()
    out = translater.translate(text=raw_text,dest=target_lang)
    return out.text


def camera_starts_detecting():
    frame_placeholder = st.empty()
    stop_button = st.button("Cancel")
    video = cv2.VideoCapture(0)
    
    while True:    
        success,frame = video.read()
        if not success:
            break
        # Detect medicine name using Yolov8 model
        pred = model.predict(source=frame,
                            save=False,
                            conf=0.40)
        # Crop the detected medicine name from the frame
        for x1,y1,x2,y2 in pred[0].boxes.xyxy:
            crop_img = frame[int(y1):int(y2),int(x1):int(x2)]
            img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
            img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,10) 
            # Pass cropped medicine name to EasyOCR 
            ocr_data = reader.readtext(img,detail=1,paragraph=False)
            for box,text,prob in ocr_data:
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                # Do fuzzy matching with orginal medicine names.
                r,f = fuzzy_matching(text,orginal_drug_names)
                if f==1:
                    video.release() 
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    return r
                           
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame,channels="RGB",width=600)

        if stop_button:
            break
        
    video.release()  
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def generativeAI_gemini(detected_medicine_name):
    genai.configure(api_key=Google_API_Key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f'What are the common uses of {detected_medicine_name}?')
    res=response.text
    return res
        

med_name = pd.read_csv('csv_file/med_name.csv')
orginal_drug_names = med_name['Medicine Names'].tolist()

model = YOLO('yoloV8_model_Medicine_Name_Detection/best.pt')

reader = easyocr.Reader(['en'],gpu=True)


st.set_page_config(page_title='AI-Med-Reader',layout='wide',initial_sidebar_state="collapsed")

st.markdown("""
                <h1 style='text-align: center; color: yellow;'>
                    <u style='text-decoration: underline;text-decoration-color: silver;'>
                        AI-Med-Reader
                    </u>
                </h1>
            """, unsafe_allow_html=True)

col1,col2 = st.columns([1,6])
with col1:
    img = Image.open('images/med_img.png')
    st.image(img,width=200)
with col2:
    st.header("Hi, I'm AI-Med-Reader. Please show me your medicine, and I will answer for what disease this medicine is used for.")

with st.container():
    col3,col4,col5 =  st.columns([4,6,4])
    with col4:
        st.markdown("""
                        <h3 style='text-align: center; color: hotpink;font-family:Optima;'>
                            Let's allow AI-Med-Reader to scan your medicine? 
                            <p style='color:white;'>( Warning : AI-Med-Reader will use your camera.)</p>
                        </h3> 
                    """,unsafe_allow_html=True) 
    with col5:    
        butn = st.button("Yes, allow")

if 'gemini_result' not in st.session_state:
    st.session_state.gemini_result = None

if 'det_med_name' not in st.session_state:
    st.session_state.det_med_name = None    

with st.container():
    col6,col7,col8 = st.columns([3,4,4])
    
    try:
        if butn:
            # Detect medicine name using camera input
            with col7:
                detected_medicine_name = camera_starts_detecting()
                if detected_medicine_name is not None:
                    st.markdown(
                        f"""
                        <h7>
                            Name Detected : <h8 style='color: yellow;'>{detected_medicine_name}</h8>.&emsp; Please wait for the result.
                        </h7>
                        """,
                        unsafe_allow_html=True,
                    )

            # Get result from generative AI model
            st.session_state.gemini_result = generativeAI_gemini(detected_medicine_name)
            st.session_state.det_med_name = detected_medicine_name

        if st.session_state.gemini_result is not None:
            # Display result
            st.markdown(
                """
                <h1 style='color: salmon;'>
                    AI-Med-Reader says :
                </h1>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                        f"""
                        <h7>
                            Medicine Name : <h8 style='color: yellow;'>{st.session_state.det_med_name}</h8>
                        </h7>
                        """,
                        unsafe_allow_html=True,
                    )        
            
            st.markdown(to_markdown(st.session_state.gemini_result)._repr_markdown_())
            st.write(" ")
            st.warning('WARNING : Self-medication can lead to serious health risks. Always consult a doctor before using any medicine.', icon="⚠️")
            st.write(" ")
            st.write(" ")
            st.write(" ")            

            # Offer translation option
            st.markdown(
                """
                <h5 style='text-align: center; color: turquoise;'>
                    Do you want to translate result language to your choice? Then please select your language below.
                </h5> 
                """,
                unsafe_allow_html=True,
            )

            target_lang = st.selectbox(
                "Choose your language:",
                set_of_languages(),
                placeholder="Select languages...",
                key="key",
                index=None,
            )
            st.write(" ")
            st.write(" ")
            st.write(" ")
            if target_lang is not None:
                # Display translated result
                st.markdown(
                    f"""
                    <h1 style='color: salmon;'>
                        AI.Med-Reader {language_translate('says',target_lang)} :
                    </h1>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                        f"""
                            <h8 style='color: yellow;'>
                                {st.session_state.det_med_name}
                            </h8>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown(result_translate(st.session_state.gemini_result, target_lang)._repr_markdown_())
                st.write(" ")
                warning_text = 'WARNING : Self-medication can lead to serious health risks. Always consult a doctor before using any medicine.'
                st.warning(language_translate(warning_text,target_lang), icon="⚠️")

            st.write(" ")
            st.write(" ")
            st.write(" ")
            if st.button("Close Result",type="primary"):
                streamlit_js_eval(js_expressions="parent.window.location.reload()") 

    except ValueError:
        st.info("Something went wrong.")
        if st.button("Try again",type="primary"):
            streamlit_js_eval(js_expressions="parent.window.location.reload()") 
