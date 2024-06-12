# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""MaMMUT Demo using the saved models.
"""

from collections.abc import Sequence

from absl import app
from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf
# Enable SentencePiece op by https://github.com/tensorflow/hub/issues/463
import tensorflow_text  # pylint:disable=unused-import

np.set_printoptions(precision=3)

_SAVED_MODEL_PATH = flags.DEFINE_string(
    'saved_model_path', './checkpoints/mammut_retrieval_model',
    'Path to the saved model.')

_IMAGE_TEXT_PAIRS = [
    ('./images/green.jpg', 'Green broccolis and fruit in a bowl on a table.'),
    ('./images/seaplane.jpg', 'A seaplane in the lake against dark mountains.'),
    ('./images/top-tennis.jpg', 'A tennis player about to serve.'),
    ('./images/horse.jpg', 'A horse grazing on a grass field.'),
    ('./images/zebra.jpg', 'Three zebras in a dry land with some bush.'),
    ('./images/baguette.jpg', 'A baguette with some ham in it.'),
]


def get_input():
  """Get fake input for the model."""
  # Drop last PNG channel.
  for image_path, text in _IMAGE_TEXT_PAIRS:
    image = tf.constant(Image.open(image_path))
    encoded_frame = tf.image.encode_jpeg(image)
    batched_image_bytes = tf.expand_dims(encoded_frame, axis=0)
    examples = {
        'image_bytes': batched_image_bytes,
        'text': tf.strings.as_string([text]),
    }
    yield examples, image_path, text


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  generate_example = get_input()
  model_imported = tf.saved_model.load(_SAVED_MODEL_PATH.value)
  model_imported_fn = model_imported.signatures['serving_default']
  image_embeddings, text_embeddings = [], []
  for example, image_path, text in generate_example:
    output = model_imported_fn(**example)
    print(f'Image: {image_path}')
    print(f'Text: {text}')
    image_embeddings.append(output['normalized_image_embedding'].numpy())
    text_embeddings.append(output['normalized_text_embedding'].numpy())

  image_embeddings = np.concatenate(image_embeddings, axis=0)
  text_embeddings = np.concatenate(text_embeddings, axis=0)
  similarity = np.matmul(image_embeddings, text_embeddings.T)
  print('================================================================')
  print('Image-text similarity matrix:')
  print(similarity)
  argmax_indices = np.argmax(similarity, axis=-1)
  print('Similarity matrix argmax indices:')
  print(argmax_indices)
  expected_indices = np.arange(argmax_indices.size)
  # Assert each image embedding is most aligned with its own text embedding.
  np.testing.assert_array_equal(argmax_indices, expected_indices)


if __name__ == '__main__':
  app.run(main)
