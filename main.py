import streamlit as st
import numpy as np
import io
import requests
from PIL import Image, ImageDraw, ImageFont, ImageColor
from requests_toolbelt.multipart.encoder import MultipartEncoder

# Set page configs.
st.set_page_config(page_title="Object Detection", layout="centered")

# -------------Header Section------------------------------------------------

title = '<p style="text-align: center;font-size: 40px;font-weight: 550; "> Vehicle & Numberplate Detection </p>'
st.markdown(title, unsafe_allow_html=True)

st.markdown(
    "&nbsp; Upload your Image and our ML model will "
    "find all the Vehicles/Numberplates inside the Image", unsafe_allow_html=True
)

# -------------Sidebar Section------------------------------------------------

detection_mode = None

with st.sidebar:
    title = '<p style="font-size: 25px;font-weight:450;">Detection settings</p>'
    st.markdown(title, unsafe_allow_html=True)

    # choose what you want to detect
    mode = st.radio("What do you want to detect ?", ('Vehicles Only',
                                                     'Numberplates Only'), index=1)
    if mode == 'Numberplates Only':
        detection_mode = mode
    else:
        detection_mode = 'Vehicles Only'

    # Get bbox color and convert from hex to rgb
    bbox_color = ImageColor.getcolor(str(st.color_picker(label="Bounding Box Color", value="#00ffce")), "RGB")

    # bbox thickness
    bbox_thickness = st.slider("Bounding Box Thickness", min_value=2, max_value=10,
                               help="Sets the thickness of bounding boxes",
                               value=2)

    st.info("NOTE : All Images are automatically resized to 416x416. "
            "Some Images which are unsuitable or corrupted are automatically rejected by the model.")

    # line break
    st.markdown(" ")
    # About the programmer
    st.markdown("## Product by **Midhawy Alabri** \U0001F609")

# -------------Body Section------------------------------------------------

if detection_mode == "Numberplates Only":

    # Example Images
    st.image(image="./assets/collage.jpg")
    st.markdown("</br>", unsafe_allow_html=True)

    # Upload the Image
    content_image = st.file_uploader(
        "Upload Image (PNG & JPG images only). All Images are resized to (416x416) for performance.",
        type=['png', 'jpg', 'jpeg'])

    st.markdown("</br>", unsafe_allow_html=True)

    if content_image is not None:

        with st.spinner("Scanning the Image...will take about 10-15 secs"):

            upload_image = Image.open(content_image)
            upload_image = np.array(upload_image)
            pilImage = Image.fromarray(upload_image)

            # resize to (416,416)
            resized_pilImage = pilImage.resize(size=(416, 416))

            # ---------------Detection Phase-------------------------

            # Convert to JPEG Buffer
            buffered = io.BytesIO()

            try:
                resized_pilImage.save(buffered, quality=100, format="JPEG")

                # Build multipart form and post request
                m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})

                response = requests.post(
                    "https://detect.roboflow.com/license-plate-detection-s3g4g/1?api_key=vGEbPRlKg27qAG4N76W0", data=m,
                    headers={'Content-Type': m.content_type})

                print(response)
                preds = response.json()

                detections = preds['predictions']
                print(preds['predictions'])

                # ----------------Draw BBoxes--------------------------------

                image_with_detections = resized_pilImage
                draw = ImageDraw.Draw(image_with_detections)
                font = ImageFont.load_default()

                for box in detections:
                    x1 = box['x'] - box['width'] / 2
                    x2 = box['x'] + box['width'] / 2
                    y1 = box['y'] - box['height'] / 2
                    y2 = box['y'] + box['height'] / 2
                    conf_score = box['confidence']
                    draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=bbox_thickness)

                print('Done')
                if image_with_detections is not None:
                    # some baloons
                    st.balloons()

                col1, col2 = st.columns(2)
                with col1:
                    # Display the output
                    st.image(image_with_detections)
                with col2:
                    st.markdown("</br>", unsafe_allow_html=True)
                    st.markdown(f"<h5> Detected <u>{len(detections)}</u>  Numberplates </h5>", unsafe_allow_html=True)
                    st.markdown(
                        "<b> Your Image is Ready ! Click below to download it. </b> ", unsafe_allow_html=True)

                    # convert to pillow image
                    img = Image.fromarray(np.array(image_with_detections))
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    st.download_button(
                        label="Download image",
                        data=buffered.getvalue(),
                        file_name="output.png",
                        mime="image/png")

            except:
                title = '<p style="text-align: center;font-size: 20px;font-weight: 350; "> Image not suitable for detection, try some other Image.  </p>'
                st.markdown(title, unsafe_allow_html=True)

else:

    # Example Images
    st.image(image="./assets/collage2.jpg")
    st.markdown("</br>", unsafe_allow_html=True)

    # Upload the Image
    content_image = st.file_uploader(
        "Upload Image (PNG & JPG images only). All Images are resized to (416x416) for performance.",
        type=['png', 'jpg', 'jpeg'])

    st.markdown("</br>", unsafe_allow_html=True)

    if content_image is not None:

        with st.spinner("Scanning the Image...will take about 10-15 secs"):

            upload_image = Image.open(content_image)
            upload_image = np.array(upload_image)
            pilImage = Image.fromarray(upload_image)

            # resize to (416,416)
            resized_pilImage = pilImage.resize(size=(416, 416))

            # ---------------Detection Phase-------------------------

            # Convert to JPEG Buffer
            buffered = io.BytesIO()

            try:
                resized_pilImage.save(buffered, quality=100, format="JPEG")

                # Build multipart form and post request
                m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})

                response = requests.post(
                    "https://detect.roboflow.com/vehicle-detection-ptces/1?api_key=vGEbPRlKg27qAG4N76W0", data=m,
                    headers={'Content-Type': m.content_type})

                print(response)
                preds = response.json()

                detections = preds['predictions']
                print(preds['predictions'])

                # ----------------Draw BBoxes--------------------------------

                image_with_detections = resized_pilImage
                draw = ImageDraw.Draw(image_with_detections)
                font = ImageFont.load_default()

                for box in detections:
                    x1 = box['x'] - box['width'] / 2
                    x2 = box['x'] + box['width'] / 2
                    y1 = box['y'] - box['height'] / 2
                    y2 = box['y'] + box['height'] / 2
                    conf_score = box['confidence']
                    draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=bbox_thickness)

                print('Done')
                if image_with_detections is not None:
                    # some baloons
                    st.balloons()

                col1, col2 = st.columns(2)
                with col1:
                    # Display the output
                    st.image(image_with_detections)
                with col2:
                    st.markdown("</br>", unsafe_allow_html=True)
                    st.markdown(f"<h5> Detected <u>{len(detections)}</u>  Vehicles </h5>", unsafe_allow_html=True)
                    st.markdown(
                        "<b> Your Image is Ready ! Click below to download it. </b> ", unsafe_allow_html=True)

                    # convert to pillow image
                    img = Image.fromarray(np.array(image_with_detections))
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    st.download_button(
                        label="Download image",
                        data=buffered.getvalue(),
                        file_name="output.png",
                        mime="image/png")

            except:
                title = '<p style="text-align: center;font-size: 20px;font-weight: 350; "> Image not suitable for detection, try some other Image.  </p>'
                st.markdown(title, unsafe_allow_html=True)

# -------------Hide Streamlit Watermark------------------------------------------------
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
