import streamlit as st
import requests
import pandas as pd
st.set_page_config(layout="wide")
from PIL import Image
from io import BytesIO
import base64
from openai import OpenAI
# from transformers import pipeline
# import torch
# from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
#dr.Client()

client = OpenAI()

def vision(url):
  botResponse = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      # {"role": "system", "content": "You are a real estate photo expert. "},
      {"role": "user", "content": [
          {"type": "text", "text": """
          Respond with 3 sections with bold headers.
          1. Room Type: in as few words as possible, indicate the room type. I.e. Kitchen, Bathroom, Laundry Room, Finished Basement, etc.
          2. Features increasing value: A bullet list noting a few of the positive attributes that would make this property desirable or valuable.
          3. Features decreasing value: A bullet list noting a few of the negative attributes that could decrease this property's value.
          """},
          {"type": "image_url",
           "image_url": {"url": str(url)}
           }]
      }
    ]
  )
  return botResponse.choices[0].message.content

def image_to_base64(image: Image) -> str:
    img_bytes = BytesIO()
    image.save(img_bytes, 'jpeg', quality=90)
    image_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    return image_base64

def is_kitchen(df):
    # Get price prediction from DataRobot
    deployment_id = '637c167648411c671967b805'
    API_KEY = 'NjQ2MjRiMzNmNDBhNzViYTNjMjAyZDc4OlpyMW9MTGQ1Vms3MllCNDV3VlhVd05CaTZLVC9mRVBjZlFiazdzaTlqWG89'
    DATAROBOT_KEY = '544ec55f-61bf-f6ee-0caf-15c7f919a45d'
    API_URL = 'https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/predictions'
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
    url = API_URL.format(deployment_id=deployment_id)
    # Make API request for predictions
    predictions_response = requests.post(
        url,
        data=df.to_json(orient='records'),
        headers=headers
    )
    try:
        predsDF = pd.json_normalize(predictions_response.json()['data'])
        #predsDF = pd.json_normalize(predsDF['predictionValues'])
        decision = predsDF.iloc[0,1]
        #probability = predsDF.iloc[0,1]
    except Exception as e:
        print("Error:")
        print(e)
    return decision

def kitchenText(image):
    loc = "ydshieh/vit-gpt2-coco-en"
    feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
    tokenizer = AutoTokenizer.from_pretrained(loc)
    model = VisionEncoderDecoderModel.from_pretrained(loc)
    model.eval()

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=25, num_beams=5, return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds[0]

def kitchenQuality(df):
    # Get price prediction from DataRobot
    deployment_id = '637d9bdf60a074513567b64e'
    API_KEY = 'NjQ2MjRiMzNmNDBhNzViYTNjMjAyZDc4OlpyMW9MTGQ1Vms3MllCNDV3VlhVd05CaTZLVC9mRVBjZlFiazdzaTlqWG89'
    DATAROBOT_KEY = '544ec55f-61bf-f6ee-0caf-15c7f919a45d'
    API_URL = 'https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/predictions'
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
    url = API_URL.format(deployment_id=deployment_id)
    # Make API request for predictions
    predictions_response = requests.post(
        url,
        data=df.to_json(orient='records'),
        headers=headers
    )
    try:
        predsDF = pd.json_normalize(predictions_response.json()['data'])
        #predsDF = pd.json_normalize(predsDF['predictionValues'])
        quality = predsDF.iloc[0,1]
        quality = quality.replace("1.0", "Below Average")
        quality = quality.replace("2.0", "Average")
        quality = quality.replace("3.0", "Above Average")
        #probability = predsDF.iloc[0,1]
    except Exception as e:
        print("Error:")
        print(e)
    return quality

def kitchenDetectorPage():
    st.header("Kitchen Rating by URL")
    st.write("Some Examples to try. Copy and paste one of these into the field below or use your own!")
    # Layout
    container0 = st.container()
    col1, col2, col3, col4, col5, col6 = container0.columns([1,1,1,1,1,1])

    with col1:
        st.caption("https://ssl.cdn-redfin.com/photo/248/bigphoto/702/C8160702_16_0.jpg")
        st.image("https://ssl.cdn-redfin.com/photo/248/bigphoto/702/C8160702_16_0.jpg", width=75)
    with col2:
        st.caption("https://ssl.cdn-redfin.com/photo/248/bigphoto/082/C8161082_0.jpg")
        st.image("https://ssl.cdn-redfin.com/photo/248/bigphoto/082/C8161082_0.jpg", width=150)
    with col3:
        st.caption("https://ssl.cdn-redfin.com/photo/248/bigphoto/508/C8160508_0.jpg")
        st.image("https://ssl.cdn-redfin.com/photo/248/bigphoto/508/C8160508_0.jpg", width=150)
    with col4:
        st.caption("https://ssl.cdn-redfin.com/photo/248/bigphoto/508/C8160508_11_0.jpg")
        st.image("https://ssl.cdn-redfin.com/photo/248/bigphoto/508/C8160508_11_0.jpg", width=150)
    with col5:
        st.caption("https://ssl.cdn-redfin.com/photo/248/bigphoto/508/C8160508_27_0.jpg")
        st.image("https://ssl.cdn-redfin.com/photo/248/bigphoto/508/C8160508_27_0.jpg", width=150)
    with col6:
        st.caption("https://ssl.cdn-redfin.com/photo/248/bigphoto/790/W8156790_15_1.jpg")
        st.image("https://ssl.cdn-redfin.com/photo/248/bigphoto/790/W8156790_15_1.jpg", width=150)

    container1 = st.container()
    with container1:
        # imageURL = "https://ssl.cdn-redfin.com/photo/248/bigphoto/457/W5827457_8_0.jpg"

        imageURL = st.text_input("Enter an image URL to get a rating", "https://ssl.cdn-redfin.com/photo/248/bigphoto/457/W5827457_8_0.jpg")
        headers = {
            'User-Agent': ""'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
        img = Image.open(requests.get(imageURL, stream=True, headers=headers).raw)
        try:
            image_base64 = image_to_base64(img)
            st.image(imageURL)
            # st.write(image_base64)

            data = {"index": 1, "Photo": image_base64}
            df = pd.DataFrame(data, index=[0])

            # img.save(str("image1.jpg"))
            decision = is_kitchen(df=df)
            if decision > 0:
                quality = kitchenQuality(df=df)
                st.subheader("This is a kitchen.")
                st.subheader("Quality score is: " + str(quality))
            else:
                st.subheader("This is not a kitchen.")
        except Exception as e:
            st.write(e)
        # with Image.open(requests.get(imageURL, stream=True).raw) as image:
        with st.spinner(text="Processing image attributes..."):
            text = vision(imageURL)
        st.subheader(text)


def kitchenCameraPage():

    # Layout
    container1 = st.container()
    with container1:
        st.write("This demo works best from your smartphone. :camera:")
        try:
            cameraPic = st.camera_input(label="Take a picture")
            headers = {
                'User-Agent': ""'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}

            try:
                img = Image.open(cameraPic)
            except:
                pass
            try:
                image_base64 = image_to_base64(img)
                data = {"index": 1, "Photo": image_base64}
                df = pd.DataFrame(data, index=[0])
                decision = is_kitchen(df=df)
                if decision > 0:
                    quality = kitchenQuality(df=df)
                    st.subheader("This is a kitchen.")
                    st.subheader("Quality score is: " + str(quality))
                else:
                    st.subheader("This is not a kitchen.")
            except:
                pass


        except Exception as e:
            st.write(e)
        #st.image(imageURL)

def kitchenTextPage():
    # Layout
    container1 = st.container()
    with container1:
        # imageURL = "https://ssl.cdn-redfin.com/photo/248/bigphoto/457/W5827457_8_0.jpg"
        st.header("Kitchen Description by URL")
        imageURL = st.text_input("Enter an image URL to get a rating", "https://ssl.cdn-redfin.com/photo/248/bigphoto/457/W5827457_8_0.jpg")
        headers = {
            'User-Agent': ""'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
        try:
            with Image.open(requests.get(imageURL, stream=True).raw) as image:
                text = kitchenText(image)

            st.write(text)

        except Exception as e:
            st.write(e)

        st.image(imageURL)

def introPage():
    st.header("Welcome to KitchenRobot")
    st.subheader("   Choose a demo from the sidebar.")
    headers = {
        'User-Agent': ""'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    logo = Image.open(requests.get('https://dv-website.s3.amazonaws.com/uploads/2021/03/datarobot-logo-300x200.png', stream=True, headers=headers).raw)
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.image(logo, width=100, caption="AI by DataRobot")

def _main():
    ## MainMenu {visibility: hidden;}
    # header {visibility: hidden;}
    hide_streamlit_style = """
    <style>
    # MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>

    """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


    # Navigation
    page_names_to_funcs = {
        # "Welcome": introPage,
        "Kitchen Rating by URL": kitchenDetectorPage,
        # "Kitchen Rating Camera": kitchenCameraPage

    }

    page_name = st.sidebar.selectbox("Choose a Page", page_names_to_funcs.keys())
    page_names_to_funcs[page_name]()


if __name__ == "__main__":
    _main()



