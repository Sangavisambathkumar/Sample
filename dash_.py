import base64
import io

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO

# Load your trained model
model = tf.keras.models.load_model('/Users/ADMIN/Documents/sem_2/user_interface/model.h5')

# Define the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the dashboard
app.layout = dbc.Container([
    html.H1("Pneumonia Detection Dashboard", style={'textAlign': 'center', 'marginBottom': 50}),
    dbc.Card(
        [
            dbc.CardHeader("Upload your X-ray image below"),
            dbc.CardBody(
                [
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '50px',
                            'lineHeight': '50px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'marginBottom': '20px',
                            'color': '#3174AD',
                            'background-color': '#C3E6CB',
                            'font-size': '18px',
                            'cursor': 'pointer'
                        },
                        multiple=False
                    ),
                    html.Div(id='output-image-upload', style={'textAlign': 'center', 'border': '2px solid #3174AD', 'borderRadius': '5px', 'padding': '10px', 'maxWidth': '100%', 'overflow': 'hidden'}),
                ]
            ),
        ],
        style={'border': '2px solid #3174AD', 'borderRadius': '5px', 'boxShadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}
    ),
], style={'maxWidth': '800px', 'margin': 'auto', 'background-color': '#E9ECEF', 'padding': '20px'})


# Callback to process uploaded image and make predictions
@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')])
def update_output(content):
    if content is not None:
        # Decode the image from base64
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        image_bytes = BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        image = tf.keras.preprocessing.image.load_img(image_bytes,target_size=(224, 224))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        # Make prediction
        prediction = model.predict(input_arr)

        # Get the predicted class
        predicted_class = np.argmax(prediction)

        # Replace this with your own class labels
        class_labels = ['Bacteria Pneumonia', 'normal', 'Viral Pneumonia']  # Example class labels

        new_width = 300
        new_height = 300

        resized_base64_string = resize_base64_image(content_string, new_width, new_height)
        return html.Div([
            html.H5('Prediction: {}'.format(class_labels[predicted_class])),
            html.Img(src=content_type+","+resized_base64_string)
        ])

    else:
        return None


def resize_base64_image(base64_string, new_width, new_height):
    # Decode the Base64 string into bytes
    image_data = base64.b64decode(base64_string)
    
    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))
    
    # Resize the image
    resized_image = image.resize((new_width, new_height))
    
    # Convert the resized image to bytes
    with io.BytesIO() as buffer:
        resized_image.save(buffer, format="JPEG")  # You can specify the desired format here
        image_bytes = buffer.getvalue()
    
    # Encode the resized image bytes back to Base64
    resized_base64_string = base64.b64encode(image_bytes).decode('utf-8')
    
    return resized_base64_string

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

