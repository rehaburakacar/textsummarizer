# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

import io
import json
import os

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from ..configuration_utils import PretrainedConfig
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..models.auto.configuration_auto import AutoConfig
from ..models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
from ..models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import http_get, is_tf_available, is_torch_available, logging

from .base import (
    Pipeline,
    get_default_model,
    infer_framework_load_model,
)

from .text2text_generation import SummarizationPipeline
from .text_classification import TextClassificationPipeline
import tensorflow as tf
from ..models.auto.modeling_tf_auto import (    
    TFAutoModelForSeq2SeqLM,
    TFAutoModelForSequenceClassification,    
)




logger = logging.get_logger(__name__)


# Register all the supported tasks here
TASK_ALIASES = {
    "sentiment-analysis": "text-classification",
    "ner": "token-classification",
}
SUPPORTED_TASKS = {
    
    "text-classification": {
        "impl": TextClassificationPipeline,
        "tf": (TFAutoModelForSequenceClassification,) if is_tf_available() else (),
        "pt": (AutoModelForSequenceClassification,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": "distilbert-base-uncased-finetuned-sst-2-english",
                "tf": "distilbert-base-uncased-finetuned-sst-2-english",
            },
        },
        "type": "text",
    },
    
    "summarization": {
        "impl": SummarizationPipeline,
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
        "default": {"model": {"pt": "sshleifer/distilbart-cnn-12-6", "tf": "t5-small"}},
        "type": "text",
    }
   
}

NO_FEATURE_EXTRACTOR_TASKS = set()
NO_TOKENIZER_TASKS = set()
for task, values in SUPPORTED_TASKS.items():
    if values["type"] == "text":
        NO_FEATURE_EXTRACTOR_TASKS.add(task)
    elif values["type"] in {"audio", "image"}:
        NO_TOKENIZER_TASKS.add(task)
    elif values["type"] != "multimodal":
        raise ValueError(f"SUPPORTED_TASK {task} contains invalid type {values['type']}")


def get_supported_tasks() -> List[str]:
    """
    Returns a list of supported task strings.
    """
    supported_tasks = list(SUPPORTED_TASKS.keys()) + list(TASK_ALIASES.keys())
    supported_tasks.sort()
    return supported_tasks


def get_task(model: str, use_auth_token: Optional[str] = None) -> str:
    tmp = io.BytesIO()
    headers = {}
    if use_auth_token:
        headers["Authorization"] = f"Bearer {use_auth_token}"

    try:
        http_get(f"https://huggingface.co/api/models/{model}", tmp, headers=headers)
        tmp.seek(0)
        body = tmp.read()
        data = json.loads(body)
    except Exception as e:
        raise RuntimeError(f"Instantiating a pipeline without a task set raised an error: {e}")
    if "pipeline_tag" not in data:
        raise RuntimeError(
            f"The model {model} does not seem to have a correct `pipeline_tag` set to infer the task automatically"
        )
    if data.get("library_name", "transformers") != "transformers":
        raise RuntimeError(f"This model is meant to be used with {data['library_name']} not with transformers")
    task = data["pipeline_tag"]
    return task


def check_task(task: str) -> Tuple[Dict, Any]:
    
    if task in TASK_ALIASES:
        task = TASK_ALIASES[task]
    if task in SUPPORTED_TASKS:
        targeted_task = SUPPORTED_TASKS[task]
        return targeted_task, None

    if task.startswith("translation"):
        tokens = task.split("_")
        if len(tokens) == 4 and tokens[0] == "translation" and tokens[2] == "to":
            targeted_task = SUPPORTED_TASKS["translation"]
            return targeted_task, (tokens[1], tokens[3])
        raise KeyError(f"Invalid translation task {task}, use 'translation_XX_to_YY' format")

    raise KeyError(f"Unknown task {task}, available tasks are {get_supported_tasks() + ['translation_XX_to_YY']}")


def pipeline(
    task: str = None,
    model: Optional = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    use_auth_token: Optional[Union[str, bool]] = None,
    model_kwargs: Dict[str, Any] = None,
    pipeline_class: Optional[Any] = None,
    **kwargs
) -> Pipeline:  

   
  
    if model_kwargs is None:
        model_kwargs = {}

    if task is None and model is None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline without either a task or a model"
            "being specified."
            "Please provide a task class or a model"
        )

    if model is None and tokenizer is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with tokenizer specified but not the model "
            "as the provided tokenizer may not be compatible with the default model. "
            "Please provide a PreTrainedModel class or a path/identifier to a pretrained model when providing tokenizer."
        )
    if model is None and feature_extractor is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with feature_extractor specified but not the model "
            "as the provided feature_extractor may not be compatible with the default model. "
            "Please provide a PreTrainedModel class or a path/identifier to a pretrained model when providing feature_extractor."
        )

    if task is None and model is not None:
        if not isinstance(model, str):
            raise RuntimeError(
                "Inferring the task automatically requires to check the hub with a model_id defined as a `str`."
                f"{model} is not a valid model_id."
            )
        task = get_task(model, use_auth_token)

    # Retrieve the task
    targeted_task, task_options = check_task(task)
    if pipeline_class is None:
        pipeline_class = targeted_task["impl"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        # At that point framework might still be undetermined
        model = get_default_model(targeted_task, framework, task_options)
        logger.warning(f"No model was supplied, defaulted to {model} (https://huggingface.co/{model})")

    # Retrieve use_auth_token and add it to model_kwargs to be used in .from_pretrained
    model_kwargs["use_auth_token"] = model_kwargs.get("use_auth_token", use_auth_token)

    # Config is the primordial information item.
    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config, revision=revision, _from_pipeline=task, **model_kwargs)
    elif config is None and isinstance(model, str):
        config = AutoConfig.from_pretrained(model, revision=revision, _from_pipeline=task, **model_kwargs)

    model_name = model if isinstance(model, str) else None

    # Infer the framework from the model
    # Forced if framework already defined, inferred if it's None
    # Will load the correct model if possible
    model_classes = {"tf": targeted_task["tf"], "pt": targeted_task["pt"]}
    framework, model = infer_framework_load_model(
        model,
        model_classes=model_classes,
        config=config,
        framework=framework,
        revision=revision,
        task=task,
        **model_kwargs,
    )

    model_config = model.config

    load_tokenizer = type(model_config) in TOKENIZER_MAPPING or model_config.tokenizer_class is not None
    load_feature_extractor = type(model_config) in FEATURE_EXTRACTOR_MAPPING or feature_extractor is not None

    if task in NO_TOKENIZER_TASKS:
        # These will never require a tokenizer.
        # the model on the other hand might have a tokenizer, but
        # the files could be missing from the hub, instead of failing
        # on such repos, we just force to not load it.
        load_tokenizer = False
    if task in NO_FEATURE_EXTRACTOR_TASKS:
        load_feature_extractor = False

    if load_tokenizer:
        # Try to infer tokenizer from model or config name (if provided as str)
        if tokenizer is None:
            if isinstance(model_name, str):
                tokenizer = model_name
            elif isinstance(config, str):
                tokenizer = config
            else:
                # Impossible to guess what is the right tokenizer here
                raise Exception(
                    "Impossible to guess which tokenizer to use. "
                    "Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                )

        # Instantiate tokenizer if needed
        if isinstance(tokenizer, (str, tuple)):
            if isinstance(tokenizer, tuple):
                # For tuple we have (tokenizer name, {kwargs})
                use_fast = tokenizer[1].pop("use_fast", use_fast)
                tokenizer_identifier = tokenizer[0]
                tokenizer_kwargs = tokenizer[1]
            else:
                tokenizer_identifier = tokenizer
                tokenizer_kwargs = model_kwargs

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_identifier, revision=revision, use_fast=use_fast, _from_pipeline=task, **tokenizer_kwargs
            )

    if load_feature_extractor:
        # Try to infer feature extractor from model or config name (if provided as str)
        if feature_extractor is None:
            if isinstance(model_name, str):
                feature_extractor = model_name
            elif isinstance(config, str):
                feature_extractor = config
            else:
                # Impossible to guess what is the right feature_extractor here
                raise Exception(
                    "Impossible to guess which feature extractor to use. "
                    "Please provide a PreTrainedFeatureExtractor class or a path/identifier "
                    "to a pretrained feature extractor."
                )

        # Instantiate feature_extractor if needed
        if isinstance(feature_extractor, (str, tuple)):
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                feature_extractor, revision=revision, _from_pipeline=task, **model_kwargs
            )

            if (
                feature_extractor._processor_class
                and feature_extractor._processor_class.endswith("WithLM")
                and isinstance(model_name, str)
            ):
                try:
                    import kenlm  # to trigger `ImportError` if not installed
                    from pyctcdecode import BeamSearchDecoderCTC

                    if os.path.isdir(model_name) or os.path.isfile(model_name):
                        decoder = BeamSearchDecoderCTC.load_from_dir(model_name)
                    else:
                        language_model_glob = os.path.join(
                            BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY, "*"
                        )
                        alphabet_filename = BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME
                        allow_regex = [language_model_glob, alphabet_filename]
                        decoder = BeamSearchDecoderCTC.load_from_hf_hub(model_name, allow_regex=allow_regex)

                    kwargs["decoder"] = decoder
                except ImportError as e:
                    logger.warning(
                        f"Could not load the `decoder` for {model_name}. Defaulting to raw CTC. Try to install `pyctcdecode` and `kenlm`: (`pip install pyctcdecode`, `pip install https://github.com/kpu/kenlm/archive/master.zip`): Error: {e}"
                    )

    if task == "translation" and model.config.task_specific_params:
        for key in model.config.task_specific_params:
            if key.startswith("translation"):
                task = key
                warnings.warn(
                    f'"translation" task was used, instead of "translation_XX_to_YY", defaulting to "{task}"',
                    UserWarning,
                )
                break

    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer

    if feature_extractor is not None:
        kwargs["feature_extractor"] = feature_extractor

    return pipeline_class(model=model, framework=framework, task=task, **kwargs)
