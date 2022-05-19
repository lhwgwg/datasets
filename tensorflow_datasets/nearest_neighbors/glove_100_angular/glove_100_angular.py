# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""glove_100_angular dataset."""

import h5py
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """
Pre-trained Global Vectors for Word Representation (GloVe) embeddings for
approximate nearest neighbor search. This database consists of 1,183,514 data
points, each represented in 100 float32 (stored in feature 'train'). The test
query consists of 10,000 data points (stored in feature 'test'). The ground
truth neighbors are determined by the angular (cosine) distance. The 100 ground
truth neighbors are stored by the indices in the feature 'neighbors' and by the
cosine distances in 'distances'.

The entire dataset is stored in a single split 'dadtaset' with a single record
consists of four features: 'train', 'test', 'neighbors', and 'distances'. This
is due to tfds is not very efficient on processing dataset with million-scale
records (slow to process and can easily trigger hash collisions).
"""

_CITATION = """
@inproceedings{pennington2014glove,
  author = {Jeffrey Pennington and Richard Socher and Christopher D. Manning},
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  title = {GloVe: Global Vectors for Word Representation},
  year = {2014},
  pages = {1532--1543},
  url = {http://www.aclweb.org/anthology/D14-1162},
}
"""

_URL = 'http://ann-benchmarks.com/glove-100-angular.hdf5'


class Glove100Angular(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for glove_100_angular dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        # this is feature, not split
        features=tfds.features.FeaturesDict({
            'train':
                tfds.features.Tensor(shape=(None, 100), dtype=tf.float32),
            'test':
                tfds.features.Tensor(shape=(None, 100), dtype=tf.float32),
            'distances':
                tfds.features.Tensor(shape=(None, 100), dtype=tf.float32),
            'neighbors':
                tfds.features.Tensor(shape=(None, 100), dtype=tf.int32),
        }),
        homepage='https://nlp.stanford.edu/projects/glove/',
        citation=_CITATION,
        disable_shuffling=True,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract({'file': _URL})
    return {
        'database': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Pulls data from hdf5 in to a single record split."""
    # It is very easy for the O(million) items in train to get a hash collision,
    # even when an extra index is inserted into the features.  Therefore we
    # simply wrap the entire dataset into a single record.
    with tf.io.gfile.GFile(path['file'], 'rb') as f:
      with h5py.File(f, 'r') as dataset_file:
        yield 'database', {
            h5key: dataset_file[h5key][:]
            for h5key in ['train', 'test', 'neighbors', 'distances']
        }
