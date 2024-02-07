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
from PIL import Image
import tensorflow as tf
# Enable SentencePiece op by https://github.com/tensorflow/hub/issues/463
import tensorflow_text  # pylint:disable=unused-import


_SAVED_MODEL_PATH = './checkpoints/mammut_vqa_model'
_IMAGE_QA_PAIRS = [
    ('./images/green.jpg', 'is there green in the image?'),
    ('./images/seaplane.jpg', 'is this a beach sitting?'),
    ('./images/top-tennis.jpg', 'where is the player?'),
    ('./images/horse.jpg', 'how many boats in the photo?'),
    ('./images/zebra.jpg', 'how many zebras are in the photo?'),
    ('./images/baguette.jpg', 'what kind of ham is in the sandwich?'),
]


def get_input():
  """Get fake input for the model."""
  # Drop last PNG channel.
  for image_path, question_text in _IMAGE_QA_PAIRS:
    image = tf.constant(Image.open(image_path))
    encoded_frame = tf.image.encode_jpeg(image)
    batched_image_bytes = tf.expand_dims(encoded_frame, axis=0)
    examples = {
        'image_bytes': batched_image_bytes,
        'text': tf.strings.as_string([question_text]),
    }
    yield examples, image_path, question_text


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  generate_example = get_input()
  model_imported = tf.saved_model.load(_SAVED_MODEL_PATH)
  model_imported_fn = model_imported.signatures['serving_default']
  for example, image_path, question_text in generate_example:
    output = model_imported_fn(**example)
    answer_text = output['output_0'].numpy().decode('utf-8')
    print(f'Image: {image_path}')
    print(f'Question: {question_text}')
    print(f'Answer: {answer_text}')


if __name__ == '__main__':
  app.run(main)
