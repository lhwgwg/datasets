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

# TODO(vaxelrod): appropriately recognize that this code is madapted from
# https://huggingface.co/datasets/google/xtreme_s/blob/main/xtreme_s.py
"""Data for the XTREME-S Benchmark."""

import collections
import os

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@article{conneau2022xtreme,
  title={XTREME-S: Evaluating Cross-lingual Speech Representations},
  author={Conneau, Alexis and Bapna, Ankur and Zhang, Yu and Ma, Min and von Platen, Patrick and Lozhkov, Anton and Cherry, Colin and Jia, Ye and Rivera, Clara and Kale, Mihir and others},
  journal={arXiv preprint arXiv:2203.10752},
  year={2022}
}
"""

_DESCRIPTION = """\
XTREME-S covers four task families: speech recognition, classification, speech-to-text translation and retrieval. Covering 102
languages from 10+ language families, 3 different domains and 4
task families, XTREME-S aims to simplify multilingual speech
representation evaluation, as well as catalyze research in “universal” speech representation learning.

In this version, only the FLEURS dataset is provided, which covers speech
recognition and speech-to-text translation.
"""

# TODO(vaxelrod): remove t following, it's to align diffs only.
"""_MINDS_14_LANG = sorted([

    "cs-CZ",
    "de-DE",
    "en-AU",
    "en-GB",
    "en-US",
    "es-ES",
    "fr-FR",
    "it-IT",
    "ko-KR",
    "nl-NL",
    "pl-PL",
    "pt-PT",
    "ru-RU",
    "zh-CN",
])
"""

# TODO(vaxelrod): re-introduce python formatting, just keeping formatting
# same as original huggingface code for now for fewer diffs in initial review.
# pyformat: disable
_FLEURS_LANG_TO_ID = collections.OrderedDict([
                      ("Afrikaans", "af"), ("Amharic", "am"),
                      ("Arabic", "ar"), ("Armenian", "hy"),
                      ("Assamese", "as"), ("Asturian", "ast"),
                      ("Azerbaijani", "az"), ("Belarusian", "be"),
                      ("Bengali", "bn"), ("Bosnian", "bs"),
                      ("Bulgarian", "bg"), ("Burmese", "my"),
                      ("Catalan", "ca"), ("Cebuano", "ceb"),
                      ("Mandarin Chinese", "cmn_hans"),
                      ("Cantonese Chinese", "yue_hant"),
                      ("Croatian", "hr"), ("Czech", "cs"),
                      ("Danish", "da"), ("Dutch", "nl"),
                      ("English", "en"), ("Estonian", "et"),
                      ("Filipino", "fil"), ("Finnish", "fi"),
                      ("French", "fr"), ("Fula", "ff"),
                      ("Galician", "gl"), ("Ganda", "lg"),
                      ("Georgian", "ka"), ("German", "de"),
                      ("Greek", "el"), ("Gujarati", "gu"),
                      ("Hausa", "ha"), ("Hebrew", "he"),
                      ("Hindi", "hi"), ("Hungarian", "hu"),
                      ("Icelandic", "is"), ("Igbo", "ig"),
                      ("Indonesian", "id"), ("Irish", "ga"),
                      ("Italian", "it"), ("Japanese", "ja"),
                      ("Javanese", "jv"), ("Kabuverdianu", "kea"),
                      ("Kamba", "kam"), ("Kannada", "kn"),
                      ("Kazakh", "kk"), ("Khmer", "km"),
                      ("Korean", "ko"), ("Kyrgyz", "ky"),
                      ("Lao", "lo"), ("Latvian", "lv"),
                      ("Lingala", "ln"), ("Lithuanian", "lt"),
                      ("Luo", "luo"), ("Luxembourgish", "lb"),
                      ("Macedonian", "mk"), ("Malay", "ms"),
                      ("Malayalam", "ml"), ("Maltese", "mt"),
                      ("Maori", "mi"), ("Marathi", "mr"),
                      ("Mongolian", "mn"), ("Nepali", "ne"),
                      ("Northern-Sotho", "nso"),
                      ("Norwegian", "nb"), ("Nyanja", "ny"),
                      ("Occitan", "oc"), ("Oriya", "or"),
                      ("Oromo", "om"), ("Pashto", "ps"),
                      ("Persian", "fa"), ("Polish", "pl"),
                      ("Portuguese", "pt"), ("Punjabi", "pa"),
                      ("Romanian", "ro"), ("Russian", "ru"),
                      ("Serbian", "sr"), ("Shona", "sn"),
                      ("Sindhi", "sd"), ("Slovak", "sk"),
                      ("Slovenian", "sl"), ("Somali", "so"),
                      ("Sorani-Kurdish", "ckb"), ("Spanish", "es"),
                      ("Swahili", "sw"), ("Swedish", "sv"),
                      ("Tajik", "tg"), ("Tamil", "ta"),
                      ("Telugu", "te"), ("Thai", "th"),
                      ("Turkish", "tr"), ("Ukrainian", "uk"),
                      ("Umbundu", "umb"), ("Urdu", "ur"),
                      ("Uzbek", "uz"), ("Vietnamese", "vi"),
                      ("Welsh", "cy"), ("Wolof", "wo"),
                      ("Xhosa", "xh"), ("Yoruba", "yo"),
                      ("Zulu", "zu")
])
# pyformat: enable

_FLEURS_LANG_SHORT_TO_LONG = {v: k for k, v in _FLEURS_LANG_TO_ID.items()}

_FLEURS_LANG = sorted([
    "af_za", "am_et", "ar_eg", "as_in", "ast_es", "az_az", "be_by", "bn_in",
    "bs_ba", "ca_es", "ceb_ph", "cmn_hans_cn", "yue_hant_hk", "cs_cz", "cy_gb",
    "da_dk", "de_de", "el_gr", "en_us", "es_419", "et_ee", "fa_ir", "ff_sn",
    "fi_fi", "fil_ph", "fr_fr", "ga_ie", "gl_es", "gu_in", "ha_ng", "he_il",
    "hi_in", "hr_hr", "hu_hu", "hy_am", "id_id", "ig_ng", "is_is", "it_it",
    "ja_jp", "jv_id", "ka_ge", "kam_ke", "kea_cv", "kk_kz", "km_kh", "kn_in",
    "ko_kr", "ckb_iq", "ky_kg", "lb_lu", "lg_ug", "ln_cd", "lo_la", "lt_lt",
    "luo_ke", "lv_lv", "mi_nz", "mk_mk", "ml_in", "mn_mn", "mr_in", "ms_my",
    "mt_mt", "my_mm", "nb_no", "ne_np", "nl_nl", "nso_za", "ny_mw", "oc_fr",
    "om_et", "or_in", "pa_in", "pl_pl", "ps_af", "pt_br", "ro_ro", "ru_ru",
    "bg_bg", "sd_in", "sk_sk", "sl_si", "sn_zw", "so_so", "sr_rs", "sv_se",
    "sw_ke", "ta_in", "te_in", "tg_tj", "th_th", "tr_tr", "uk_ua", "umb_ao",
    "ur_pk", "uz_uz", "vi_vn", "wo_sn", "xh_za", "yo_ng", "zu_za"
])
_FLEURS_LONG_TO_LANG = {
    _FLEURS_LANG_SHORT_TO_LONG["_".join(k.split("_")[:-1]) or k]: k
    for k in _FLEURS_LANG
}
_FLEURS_LANG_TO_LONG = {v: k for k, v in _FLEURS_LONG_TO_LANG.items()}

_FLEURS_GROUP_TO_LONG = collections.OrderedDict({
    "western_european_we": [
        "Asturian", "Bosnian", "Catalan", "Croatian", "Danish", "Dutch",
        "English", "Finnish", "French", "Galician", "German", "Greek",
        "Hungarian", "Icelandic", "Irish", "Italian", "Kabuverdianu",
        "Luxembourgish", "Maltese", "Norwegian", "Occitan", "Portuguese",
        "Spanish", "Swedish", "Welsh"
    ],
    "eastern_european_ee": [
        "Armenian", "Belarusian", "Bulgarian", "Czech", "Estonian", "Georgian",
        "Latvian", "Lithuanian", "Macedonian", "Polish", "Romanian", "Russian",
        "Serbian", "Slovak", "Slovenian", "Ukrainian"
    ],
    "central_asia_middle_north_african_cmn": [
        "Arabic", "Azerbaijani", "Hebrew", "Kazakh", "Kyrgyz", "Mongolian",
        "Pashto", "Persian", "Sorani-Kurdish", "Tajik", "Turkish", "Uzbek"
    ],
    "sub_saharan_african_ssa": [
        "Afrikaans", "Amharic", "Fula", "Ganda", "Hausa", "Igbo", "Kamba",
        "Lingala", "Luo", "Northern-Sotho", "Nyanja", "Oromo", "Shona",
        "Somali", "Swahili", "Umbundu", "Wolof", "Xhosa", "Yoruba", "Zulu"
    ],
    "south_asian_sa": [
        "Assamese", "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam",
        "Marathi", "Nepali", "Oriya", "Punjabi", "Sindhi", "Tamil", "Telugu",
        "Urdu"
    ],
    "south_east_asian_sea": [
        "Burmese", "Cebuano", "Filipino", "Indonesian", "Javanese", "Khmer",
        "Lao", "Malay", "Maori", "Thai", "Vietnamese"
    ],
    "chinese_japanase_korean_cjk": [
        "Mandarin Chinese", "Cantonese Chinese", "Japanese", "Korean"
    ],
})
_FLEURS_LONG_TO_GROUP = {
    a: k for k, v in _FLEURS_GROUP_TO_LONG.items() for a in v
}
_FLEURS_LANG_TO_GROUP = {
    _FLEURS_LONG_TO_LANG[k]: v for k, v in _FLEURS_LONG_TO_GROUP.items()
}

_ALL_DATASET_CONFIGS = {
    "fleurs": _FLEURS_LANG,
}

_ALL_CONFIGS = []  # e.g. mls.en, covost.en.sv, ...
for sub_data, langs in _ALL_DATASET_CONFIGS.items():
  for lang in langs:
    _ALL_CONFIGS.append(f"{sub_data}.{lang}")

# add "all" for all datasets except 'BABEL'
_ALL_CONFIGS += [
    "fleurs.all",
]

_DESCRIPTIONS = {
    "fleurs": "FLEURS is the speech version of the FLORES machine translation"
              " benchmark, covering 2000 n-way parallel sentences in n=102"
              " languages.",
}

_CITATIONS = {
    "fleurs": "",
}

_HOMEPAGE_URLS = {
    "fleurs": "",
}

_DATA_URLS = {
    "fleurs": [
        "https://storage.googleapis.com/xtreme_translations/FLEURS102/{}.tar.gz"
    ],
}


class XtremeSConfig(tfds.core.BuilderConfig):
  """BuilderConfig for XTREME-S."""

  def __init__(self, name, dataset_name, lang_name, description, citation,
               homepage, data_urls):
    super(XtremeSConfig, self).__init__(
        name=name,
        version=tfds.core.Version("2.0.0"),
        description=self.description,
    )
    self.name = name
    self.dataset_name = dataset_name
    self.lang_name = lang_name
    self.description = description
    self.citation = citation
    self.homepage = homepage
    self.data_urls = data_urls


def _build_config(name):
  dataset_name = name.split(".")[0]
  lang_name = ".".join(name.split(".")[1:])

  return XtremeSConfig(
      name=name,
      dataset_name=dataset_name,
      lang_name=lang_name,
      description=_DESCRIPTIONS[dataset_name],
      citation=_CITATIONS[dataset_name],
      homepage=_HOMEPAGE_URLS[dataset_name],
      data_urls=_DATA_URLS[dataset_name],
  )


class XtremeS(tfds.core.GeneratorBasedBuilder):
  """XTREME-S Benchmark datasets."""

  DEFAULT_WRITER_BATCH_SIZE = 1000
  BUILDER_CONFIGS = [_build_config(name) for name in _ALL_CONFIGS]

  def _info(self):
    languages = _ALL_DATASET_CONFIGS[self.builder_config.dataset_name]

    # TODO(vaxelrod): remove these comments, using to align diffs
    # if self.config.dataset_name in ["mls", "voxpopuli"]:
    #   features = datasets.Features({

    # elif self.config.dataset_name in ["babel"]:
    #   features = datasets.Features({

    # elif self.config.dataset_name in ["covost2"]:
    #   features = datasets.Features({

    # elif self.config.dataset_name == "minds14":
    #   features = datasets.Features({

    # elif self.config.dataset_name == "fleurs":
    if self.builder_config.dataset_name == "fleurs":
      features = tfds.features.FeaturesDict({
          "id":
              tf.dtypes.int32,
          "num_samples":
              tf.dtypes.int32,
          "path":
              tf.dtypes.string,
          "audio":
              tfds.features.Audio(sample_rate=16_000),
          "transcription":
              tfds.features.Text(),
          "raw_transcription":
              tfds.features.Text(),
          "gender":
              tfds.features.ClassLabel(names=["male", "female", "other"]),
          "lang_id":
              tfds.features.ClassLabel(names=languages),
          "language":
              tf.dtypes.string,
          "lang_group_id":
              tfds.features.ClassLabel(
                  names=list(_FLEURS_GROUP_TO_LONG.keys())),
      })

      return tfds.core.DatasetInfo(
          builder=self,
          description=self.builder_config.description + "\n" + _DESCRIPTION,
          features=features,
          supervised_keys=("audio", "transcription"),
          homepage=self.builder_config.homepage,
          citation=self.builder_config.citation + "\n" + _CITATION,
      )

  def _split_generators(self, *args, **kwargs):
    if self.builder_config.dataset_name == "fleurs":
      return self._fleurs_split_generators(*args, **kwargs)

  def _generate_examples(self, *args, **kwargs):
    if self.builder_config.dataset_name == "fleurs":
      yield from self._fleurs_generate_examples(*args, **kwargs)

  # Path within the Fleurs archive files. Replaced during tests.
  # TODO(vaxelrod): remove?
  _FLEURS_DATA_SUBDIRECTORY = ""

  # TODO(vaxelrod): remove these comments. They are here to help align diffs.
  # MLS
  # def _mls_split_generators(self, dl_manager):

  # def _mls_generate_examples(self, data_dirs, sub_folder=""):
  #   """Generate examples from a Multilingual LibriSpeech data dir."""

  # Voxpopuli
  # def _voxpopuli_split_generators(self, dl_manager):

  # Covost2
  # def _covost_2_split_generators(self, dl_manager):

  # MINDS-14
  # def _minds14_split_generators(self, dl_manager):

  # BABEL
  # def _babel_split_generators(self, dl_manager):

  # Fleurs
  def _fleurs_split_generators(self, dl_manager):
    data_url_format = self.builder_config.data_urls[0]

    if self.builder_config.lang_name == "all":
      data_urls = {l: data_url_format.format(l) for l in _FLEURS_LANG}
    else:
      data_urls = {
          self.builder_config.lang_name:
              data_url_format.format(self.builder_config.lang_name)
      }

    archive_path = dl_manager.download_and_extract(data_urls)
    sub_dir = XtremeS._FLEURS_DATA_SUBDIRECTORY

    try:
      audio_path = {
          l: os.path.join(v, sub_dir, l, "audio")
          for l, v in archive_path.items()
      }
      text_path = {
          l: os.path.join(v, sub_dir, l) for l, v in archive_path.items()
      }
    except AttributeError:
      # In tests, archive_path is just a PosixGPath, it doesn't have items().
      l = self.builder_config.lang_name
      audio_path = {l: os.path.join(archive_path, sub_dir, l, "audio")}
      text_path = {l: os.path.join(archive_path, sub_dir, l)}

    return [
        tfds.core.SplitGenerator(
            name=tfds.core.Split.TRAIN,
            gen_kwargs={
                "audio_path": {
                    l: os.path.join(v, "train") for l, v in audio_path.items()
                },
                "text_path": {
                    l: os.path.join(v, "train.tsv")
                    for l, v in text_path.items()
                },
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.core.Split.VALIDATION,
            gen_kwargs={
                "audio_path": {
                    l: os.path.join(v, "dev") for l, v in audio_path.items()
                },
                "text_path": {
                    l: os.path.join(v, "dev.tsv") for l, v in text_path.items()
                },
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.core.Split.TEST,
            gen_kwargs={
                "audio_path": {
                    l: os.path.join(v, "test") for l, v in audio_path.items()
                },
                "text_path": {
                    l: os.path.join(v, "test.tsv")
                    for l, v in text_path.items()
                },
            },
        ),
    ]

  def _fleurs_generate_examples(self, audio_path, text_path):
    key = 0

    gender_to_id = {"MALE": 0, "FEMALE": 1, "OTHER": 2}

    for lang_id in text_path.keys():
      text_file = text_path[lang_id]
      audio_dir = audio_path[lang_id]

      with tf.io.gfile.GFile(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
          (
              # index is the same for all languages to keep n-way parallelism of
              # translations
              index,
              file_name,
              raw_transcription,
              transcription,  # normalized transcription
              _,  # character transcription
              num_samples,  # total number of frames
              gender,
          ) = line.strip().split("\t")

          lang_group = _FLEURS_LANG_TO_GROUP[lang_id]

          yield key, {
              "id":
                  int(index),
              "path":
                  os.path.join(audio_dir, file_name),
              "audio":
                  os.path.join(audio_dir, file_name),
              "raw_transcription":
                  raw_transcription,
              "transcription":
                  transcription,
              "num_samples":
                  int(num_samples),
              "gender":
                  gender_to_id[gender],
              "lang_id":
                  _FLEURS_LANG.index(lang_id),
              "language":
                  _FLEURS_LANG_TO_LONG[lang_id],
              "lang_group_id":
                  list(_FLEURS_GROUP_TO_LONG.keys()).index(lang_group),
          }
          key += 1
