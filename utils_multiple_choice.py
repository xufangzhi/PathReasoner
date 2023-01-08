# coding=utf-8

import numpy as np
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import jsonlines
import gensim
import tqdm
from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
import glob

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    passage_mask: Optional[List[int]]
    option_mask: Optional[List[int]]
    question_mask: Optional[List[int]]
    space_bpe_ids: Optional[List[List[int]]]
    split_bpe_ids: Optional[List[List[int]]]
    variable_tags: Optional[List[List[int]]]
    predicate_tags: Optional[List[int]]
    negation_tags: Optional[List[List[int]]]
    inverse_tags: Optional[List[int]]
    label: Optional[int]



class Split(Enum):
    train = "train"
    dev = "eval"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class MyMultipleChoiceDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            arg_tokenizer,
            relations,
            punctuations,
            task: str,
            max_seq_length: Optional[int] = None,
            max_ngram: int = 5,
            overwrite_cache=False,
            mode: Split = Split.train,
            demo=False,
        ):
            processor = processors[task]()

            if not os.path.isdir(os.path.join(data_dir, "cached_data")):
                os.mkdir(os.path.join(data_dir, "cached_data"))

            cached_features_file = os.path.join(
                data_dir,
                "cached_data",
                "dagn_cached_{}_{}_{}_{}_{}".format(
                    mode.value,
                    tokenizer.__class__.__name__,
                    str(max_seq_length),
                    task,
                    "_demo" if demo else ""
                ),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    label_list = processor.get_labels()
                    if mode == Split.dev:
                        if demo:
                            examples = processor.get_dev_demos(data_dir)
                        else:
                            examples = processor.get_dev_examples(data_dir)
                    elif mode == Split.test:
                        examples = processor.get_test_examples(data_dir)
                    elif mode == Split.train:
                        if demo:
                            examples = processor.get_train_demos(data_dir)
                        else:
                            examples = processor.get_train_examples(data_dir)
                    else:
                        raise Exception()
                    logger.info("Training examples: %s", len(examples))


                    self.features = convert_examples_to_arg_features(
                        examples,
                        label_list,
                        arg_tokenizer,
                        relations,
                        punctuations,
                        max_seq_length,
                        tokenizer,
                        max_ngram
                    )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]



class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()



class ReclorProcessor(DataProcessor):
    """Processor for the ReClor data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train_v3.json")), "train")
        # return self._create_examples(self._read_json(os.path.join(data_dir, "train_qtype.json")), "train")

    def get_train_demos(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "100_train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "val_qtype.json")), "dev")

    def get_dev_demos(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "100_val.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test_qtype.json")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _read_jsonl(self, input_file):
        reader = jsonlines.Reader(open(input_file, "r"))
        lines = [each for each in reader]
        return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []
        for d in lines:
            context = d['context']
            question = d['question']
            answers = d['answers']
            label = 0 if type == "test" else d['label'] # for test set, there is no label. Just use 0 for convenience.
            id_string = d['id_string']
            # qtype = d['qtype']
            examples.append(
                InputExample(
                    example_id = id_string,
                    question = question,
                    contexts=[context, context, context, context],  # this is not efficient but convenient
                    endings=[answers[0], answers[1], answers[2], answers[3]],
                    label = label,
                    # qtype = [qtype, qtype, qtype, qtype]
                    )
                )
        return examples


class DreamProcessor(DataProcessor):
    """Processor for the ReClor data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _read_jsonl(self, input_file):
        reader = jsonlines.Reader(open(input_file, "r"))
        lines = [each for each in reader]
        return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []
        for d in lines:
            context = " ".join(d[0])
            question = d[1][0]['question']
            answers = d[1][0]['choice']
            label = answers.index(d[1][0]['answer'])
            # qtype = d['qtype']
            if len(answers)==3:
                examples.append(
                    InputExample(
                        example_id = "",
                        question = question,
                        contexts=[context, context, context],  # this is not efficient but convenient
                        endings=[answers[0], answers[1], answers[2]],
                        label = label,
                        # qtype = [qtype, qtype, qtype, qtype]
                        )
                    )
        return examples


class LogiQAProcessor(DataProcessor):
    """ Processor for the LogiQA data set. """

    def get_demo_examples(self, data_dir):
        logger.info("LOOKING AT {} demo".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "10_logiqa.txt")), "demo")

    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "filtered_data_0.9_v2.txt")), "train")
        # return self._create_examples(self._read_txt(os.path.join(data_dir, "Train_v1.txt")), "train")

    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Eval_v1.txt")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Test_v1.txt")), "test")

    def get_labels(self):
        return [0, 1, 2, 3]

    def _read_txt(self, input_file):
        with open(input_file, "r") as f:
            lines = f.readlines()
        return lines

    def _create_examples(self, lines, type):
        """ LogiQA: each 8 lines is one data point.
                The first line is blank line;
                The second is right choice;
                The third is context;
                The fourth is question;
                The remaining four lines are four options.
        """
        label_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        assert len(lines) % 8 ==0, 'len(lines)={}'.format(len(lines))
        n_examples = int(len(lines) / 8)
        examples = []
        # for i, line in enumerate(examples):
        for i in range(n_examples):
            label_str = lines[i*8+1].strip()
            context = lines[i*8+2].strip()
            question = lines[i*8+3].strip()
            answers = lines[i*8+4 : i*8+8]

            examples.append(
                InputExample(
                    example_id = " ",  # no example_id in LogiQA.
                    question = question,
                    contexts = [context, context, context, context],
                    endings = [item.strip()[2:].strip() for item in answers],
                    label = label_map[label_str]
                )
            )
        assert len(examples) == n_examples
        return examples

class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples

class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "ARC-Challenge-Train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "ARC-Challenge-Dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "ARC-Challenge-Test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        # There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "1234":
                return int(truth) - 1
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[
                            options[0]["para"].replace("_", ""),
                            options[1]["para"].replace("_", ""),
                            options[2]["para"].replace("_", ""),
                            options[3]["para"].replace("_", ""),
                        ],
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth,
                    )
                )

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples


def manual_prompt(raw_question):
    trigger = ['which one of the following', 'Which one of the following', 'Which of the following', 
               'which of the following', 'Each of the following', 'Of the following, which one',
              'Each one of the following statements','which of the of the following',
               'Each one of the following','each of the following',
               'Which if the following', 'The following', 'Any of the following statements',
               'Which only of the following', 'Which one the following',
               'Of the following claims, which one', 'Any of the following',
               'All of the following', 'Which of following',
               'Of the following statements, which one','which one of me following',
               'Of the following claims, which', 'Of the following propositions, which one',
               'Which one of me following', 'Which of he following'
              ]
    for t in trigger:
        if raw_question.find(t) != -1:
            return raw_question.replace(t, "_")
    return raw_question

def normalize_text(text):
    if text[0] == ".":
        text = text[1:]
    text = text.replace(". . .", ".")
    text = text.replace(". .", ".")
    text = text.replace(". \"", "\"")
    text = text.replace(".\"", "\"")
    text = text.replace("! \"", "\"")
    text = text.replace("!\"", "\"")
    text = text.replace(".).", ").")
    text = text.replace(". ).", ").")
    text = text.replace(".!!",".")
    text = text.replace(". !!", ".")
    text = text.replace("!\".","\".")
    text = text.replace("! \".","\".")
    text = text.replace("...",".")
    text = text.replace("..",".")
    return text

def convert_examples_to_arg_features(
    examples: List[InputExample],
    label_list: List[str],
    arg_tokenizer,
    relations: Dict,
    punctuations: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    max_ngram: int,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`

    context -> chunks of context
            -> domain_words to Dids
    option -> chunk of option
           -> domain_words in Dids
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    cal_node = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = normalize_text(context)
            # text_b = manual_prompt(example.question)   # question containing _
            text_b = normalize_text(example.question)
            text_c = normalize_text(ending)

            stopwords = list(gensim.parsing.preprocessing.STOPWORDS) + punctuations
            inputs, no_contain = arg_tokenizer(text_a, text_b, text_c, tokenizer, stopwords, relations, punctuations, max_ngram, max_length)
            choices_inputs.append(inputs)
            if no_contain==1:
                break

        label = label_map[example.label]
        input_ids = [x['input_ids'] for x in choices_inputs]
        attention_mask = ([x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None)

        a_mask = [x["a_mask"] for x in choices_inputs]
        b_mask = [x["b_mask"] for x in choices_inputs]  # list[list]
        c_mask = [x["c_mask"] for x in choices_inputs]  # list[list]
        space_bpe_ids = [x["space_bpe_ids"] for x in choices_inputs]
        split_bpe_ids = [x["split_bpe_ids"] for x in choices_inputs]
        variable_tags = [x["variable_tags"] for x in choices_inputs]
        predicate_tags = [x["predicate_tags"] for x in choices_inputs]
        negation_tags = [x["negation_tags"] for x in choices_inputs]
        inverse_tags = [x["inverse_tags"] for x in choices_inputs]
        # if isinstance(argument_bpe_ids[0], tuple):  # (argument_bpe_pattern_ids, argument_bpe_type_ids)
        #     arg_bpe_pattern_ids, arg_bpe_type_ids = [], []
        #     for choice_pattern, choice_type in argument_bpe_ids:
        #         assert (np.array(choice_pattern) > 0).tolist() == (np.array(choice_type) > 0).tolist(), 'pattern: {}\ntype: {}'.format(
        #             choice_pattern, choice_type)
        #         arg_bpe_pattern_ids.append(choice_pattern)
        #         arg_bpe_type_ids.append(choice_type)
        #     argument_bpe_ids = (arg_bpe_pattern_ids, arg_bpe_type_ids)
        if no_contain!=1:
            features.append(
                InputFeatures(
                    example_id=example.example_id,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=None,
                    passage_mask=a_mask,
                    option_mask=b_mask,
                    question_mask=c_mask,
                    space_bpe_ids=space_bpe_ids,
                    split_bpe_ids=split_bpe_ids,
                    variable_tags=variable_tags,
                    predicate_tags=predicate_tags,
                    negation_tags=negation_tags, 
                    inverse_tags=inverse_tags, 
                    label=label,
                )
            )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features




processors = {"reclor": ReclorProcessor, "logiqa": LogiQAProcessor, "race":RaceProcessor, "arc":ArcProcessor, "dream":DreamProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"reclor", 4, "logiqa", 4}

