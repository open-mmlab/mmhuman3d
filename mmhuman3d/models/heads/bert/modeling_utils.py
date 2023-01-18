# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

import copy
import json
import logging
import os
from io import open

import mmcv
from torch import nn

logger = logging.getLogger(__name__)


class PretrainedConfig(object):
    """Base class for all configuration classes.

    Handle a few common parameters and methods for loading/downloading/saving
    configurations.
    """
    pretrained_config_archive_map = {}

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.torchscript = kwargs.pop('torchscript', False)

    def save_pretrained(self, save_directory):
        """Save a configuration object to a directory, so that it can be re-
        loaded using the `from_pretrained(save_directory)` class method."""
        assert os.path.isdir(
            save_directory), 'Saving path should be a directory'

        output_config_file = os.path.join(save_directory, 'config.json')

        self.to_json_file(output_config_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiate a PretrainedConfig from a pre-trained model configuration.

        Params:
            pretrained_model_name_or_path: either:
                - a string with the `shortcut name` of a pre-trained model
                configuration to load from cache or download and cache if not
                already stored in cache (e.g. 'bert-base-uncased').
                - a path to a `directory` containing a configuration file saved
                    using the `save_pretrained(save_directory)` method.
                - a path or url to a saved configuration `file`.
            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should
                not be used.
            return_unused_kwargs: (`optional`) bool:
                If False, this function returns the final configuration object.
                If True, this functions returns a tuple (config, unused_kwargs)
                where `unused_kwargs` is a dictionary consisting of key/value
                pairs whose keys are not configuration attributes:
                ie the part of kwargs which has not been used to update config
                and is otherwise ignored.
            kwargs: (`optional`) dict:
                Dictionary of key/value pairs with which to update the
                configuration object after loading.
                - The values in kwargs of any keys which are configuration
                attributes will be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not*
                configuration attributes is controlled by the
                `return_unused_kwargs` keyword parameter.

        """
        # cache_dir = kwargs.pop('cache_dir', None)
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)

        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_file = cls.pretrained_config_archive_map[
                pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path,
                                       'config.json')
        else:
            config_file = pretrained_model_name_or_path
        # Load config
        config = cls.from_dict(mmcv.Config.fromfile(config_file))

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'

    def to_json_file(self, json_file_path):
        """Save this instance to a json file."""
        with open(json_file_path, 'w', encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class PreTrainedModel(nn.Module):
    """Base class for all models.

    Handle loading/storing model config and a simple interface for downloading
    and loading pretrained models.
    """
    config_class = PretrainedConfig
    pretrained_model_archive_map = {}
    base_model_prefix = ''
    input_embeddings = None

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                'Param cfg in `{}` should be an instance of PretrainedConfig'
                'To create a model from a pretrained model use '
                '`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(
                    self.__class__.__name__, self.__class__.__name__))
        # Save config in model
        self.config = config
