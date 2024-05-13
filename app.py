import tensorflow as tf
import numpy as np
import gradio as gr

display_model = tf.keras.models.load_model('/revised_model.keras')

def check_flower(picture):
    resized = tf.keras.layers.Resizing(height=224, width=224)
    resized = resized(picture)
    preprocessed_image = tf.keras.preprocessing.image.img_to_array(resized)
    img_array = np.expand_dims(preprocessed_image, axis=0)
    class_labels = {0: 'Daisy', 1: 'Dandelion', 2: 'Rose', 3: 'Sunflower', 4: 'Tulip'}

    prediction = display_model.predict(img_array)

    predicted_class_index = np.argmax(prediction)

    return class_labels.get(predicted_class_index)

demo = gr.Interface(fn=check_flower,
                    inputs=gr.Image(np.ndarray(2,)),
                    title="Flower Classification",
                    description="""This tool will classify uploaded images as being either a Daisy, Dandelion, Rose, Sunflower or Tulip.""",
                    outputs=gr.Textbox(label="Predicted Flower Type:", lines=1, placeholder="Nothing uploaded yet!"),
                    allow_flagging="never",
                    examples=["data/dandelion.jpg",
                              "data/daisy.jpg",
                              "data/tulip.jpg",
                              "data/rose.jpg",
                              "data/sunflower.jpg"]
                    )

demo.launch()