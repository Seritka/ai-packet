import tensorflow as tf


d = tf.keras.models.load_model('./Chlee')
converter = tf.lite.TFLiteConverter.from_keras_model(d)
converter.target_spec.supported_types = [tf.int8]
# converter._experimental_lower_tensor_list_ops = False
#converter.inference_input_type = tf.int8
#converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)