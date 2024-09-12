import tensorflow as tf
from tensorflow import keras
import math

# Print available physical devices
print(tf.config.experimental.list_physical_devices())

def get_model():
    # Load VGG16 with pre-trained weights and without the top layer
    vgg16 = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    # Extract the output from the last layer of VGG16
    last_output = vgg16.output
    x = keras.layers.Flatten()(last_output)
    share_weight_net = keras.Model(inputs=vgg16.input, outputs=x)
    print(share_weight_net.summary())

    # Define inputs
    as_input = keras.Input(shape=(224, 224, 3), name="as_input")
    lpc_input_1 = keras.Input(shape=(224, 224, 3), name="lpc_input_1")
    lpc_input_2 = keras.Input(shape=(224, 224, 3), name="lpc_input_2")
    
    # Anti-spoofing branch
    as_flatten_1 = share_weight_net(as_input)
    as_fc1 = keras.layers.Dense(512, activation="relu", name="as_fc1")(as_flatten_1)
    as_fc2 = keras.layers.Dense(512, activation="relu", name="as_fc2")(as_fc1)
    as_output = keras.layers.Dense(2, activation='softmax', name="as_output")(as_fc2)

    # Uncomment and complete if needed
    #Local patch comparison branch
    lpc_flatten_1 = share_weight_net(lpc_input_1)
    lpc_fc_o1 = keras.layers.Dense(512, activation="relu", name="lpc_fc1")(lpc_flatten_1)

    lpc_flatten_2 = share_weight_net(lpc_input_2)
    lpc_fc_o2 = keras.layers.Dense(512, activation="relu", name="lpc_fc2")(lpc_flatten_2)

    lpc = keras.layers.Lambda(lambda x: tf.math.square(x[0] - x[1]), name='lpc')([lpc_fc_o1, lpc_fc_o2])

    # Define the model
    model = keras.Model(
        inputs=[as_input,lpc_input_1,lpc_input_2], 
        outputs=[as_output,lpc] 
    )

    # Compile the model with appropriate losses and metrics
    model.compile(
        optimizer='adam',
        loss={
            "as_output": keras.losses.SparseCategoricalCrossentropy(),
            "lpc": tpc_loss},
        loss_weights={
            "as_output": 1,
            "lpc": 2.5*math.exp(-5)
        },
        metrics={"as_output": 'accuracy'}
    )

    return model

def tpc_loss(y_true, y_pred):
    return tf.reduce_sum(y_pred, axis=1, keepdims=True)

if __name__ == '__main__':
    model = get_model()
    model.summary()
    # Uncomment to plot model architecture
    # tf.keras.utils.plot_model(
    #     model,
    #     to_file='model1.png',
    #     show_shapes=True,
    #     show_dtype=False,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=200,
    #     show_layer_activations=True,
    # )
