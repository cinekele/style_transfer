import os.path

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def compute_content_cost(activation_content, activation_generated):
    _, height, width, channels = activation_generated.get_shape().as_list()

    activation_content_unrolled = tf.reshape(activation_content, shape=[-1, channels])
    activation_generated_unrolled = tf.reshape(activation_generated, shape=[-1, channels])

    squared_difference = tf.square(tf.subtract(activation_content_unrolled, activation_generated_unrolled))
    cost = tf.reduce_sum(squared_difference)
    return cost


def gram_matrix(matrix):
    gram = tf.matmul(matrix, tf.transpose(matrix))
    return gram


def compute_layer_style_cost(style_output, generated_output):
    _, height, width, channels = generated_output.get_shape().as_list()

    activation_style = tf.transpose(tf.reshape(style_output, (height * width, channels)), perm=(1, 0))
    activation_generated = tf.transpose(tf.reshape(generated_output, (height * width, channels)), perm=(1, 0))

    gram_style = gram_matrix(activation_style)
    gram_generated = gram_matrix(activation_generated)
    cost = tf.reduce_sum(tf.square(tf.subtract(gram_style, gram_generated))) / (2 * channels * width * height) ** 2
    return cost


def compute_style_cost(style_image_output, generated_image_output, style_layers):
    cost = 0
    activation_style = style_image_output
    activation_generated = generated_image_output
    for i, weight in enumerate(style_layers):
        cost_layer = compute_layer_style_cost(activation_style[i], activation_generated[i])
        cost += cost_layer * weight[1]
    return cost


@tf.function()
def compute_total_cost(content_cost, style_cost, alpha=10, beta=40):
    total_cost = alpha * content_cost + beta * style_cost
    return total_cost


def generate_start_image(content_image):
    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
    return tf.Variable(generated_image)


def get_layers_outputs(trained_model, layer_names):
    outputs = [trained_model.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([trained_model.input], outputs)
    return model


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def show_image(generated_image, name, path=''):
    image = tensor_to_image(generated_image)
    image.show()
    image.save(os.path.join(path, name))


def train(model, style_image, content_image, epochs, optimizer, style_layers, content_layer, save_after_epoch, path):
    generated_image = generate_start_image(content_image)
    chosen_layers = style_layers + content_layer
    model_with_outputs = get_layers_outputs(model, chosen_layers)

    preprocessed_content = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    activation_content = model_with_outputs(preprocessed_content)[-1]

    preprocessed_style = tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    activation_style = model_with_outputs(preprocessed_style)[:-1]

    @tf.function()
    def _train_step(image):
        with tf.GradientTape() as tape:
            activation_generated = model_with_outputs(image)
            style_cost = compute_style_cost(activation_style, activation_generated[:-1], style_layers)
            content_cost = compute_content_cost(activation_content, activation_generated[-1])
            total_cost = compute_total_cost(content_cost, style_cost)

        grad = tape.gradient(total_cost, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, 0.0, 1.0))

    for i in tqdm(range(epochs)):
        _train_step(generated_image)
        if i % save_after_epoch == 0:
            show_image(generated_image, f"image_{i // save_after_epoch}.jpg", path)


def main():
    img_size = 500
    content_image = np.array(Image.open("../gallery/input_images/PKiN.jpg").resize((img_size, img_size)))
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
    style_image = np.array(Image.open("../gallery/input_images/StarryNight.jpg").resize((img_size, img_size)))
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

    vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                            input_shape=(img_size, img_size, 3),
                                            weights='imagenet')
    vgg.trainable = False

    optimizer = tf.keras.optimizers.Adam()
    style_layers = [
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)]
    content_layers = [('block5_conv4', 1)]
    epochs = 20000
    save_after_epoch = 500
    path_to_save = '../gallery/created_pictures'
    train(vgg, style_image, content_image, epochs, optimizer, style_layers,
          content_layers, save_after_epoch, path_to_save)


if __name__ == '__main__':
    main()
