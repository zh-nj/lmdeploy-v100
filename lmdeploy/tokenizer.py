# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

from lmdeploy.utils import get_logger

# this file will be copied to triton server, make sure all
# importing are starting from the package root lmdeploy


@dataclass
class DetokenizeState:
    """A state collection of incrementally detekenization.

    Args:
        ids_offset (int): offset to all input ids. In LMDeploy, the output
            ids length is not one by one. It could be random by random.
        prev_tokens (List[str] | None): for incrementally decoding.
            Default to None, which means the first round.
        prefix_offset (int): the start index of tokens to be converted to
            string (prev + new tokens). Default to 0 for the first round.
        read_offset (int): the end index of tokens to be converted to
            string (prev token). Default to 0 for the first round.
    """
    ids_offset: int = 0
    prev_tokens: Optional[List[str]] = None
    prefix_offset: int = 0
    read_offset: int = 0

    def as_tuple(self) -> Tuple:
        """Return a tuple of states."""
        return (self.ids_offset, self.prev_tokens, self.prefix_offset, self.read_offset)


class HuggingFaceTokenizer:
    """A wrapper of transformers' AutoTokenizer.

    Args:
        model_dir (str): the directory of the tokenizer model
    """

    def __init__(self, model_dir: str):
        self._check_transformers_version(model_dir)
        from transformers import AutoTokenizer
        self.logger = get_logger('lmdeploy')
        self.model = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self._prefix_space_tokens = None

        if self.model.eos_token_id is None:
            generation_config_file = osp.join(model_dir, 'generation_config.json')
            if osp.exists(generation_config_file):
                with open(generation_config_file, 'r') as f:
                    cfg = json.load(f)
                    self.model.eos_token_id = cfg['eos_token_id']
            elif hasattr(self.model, 'eod_id'):  # Qwen remote
                self.model.eos_token_id = self.model.eod_id

        # for stop words
        self._vocab_size_with_added: int = None
        self._maybe_decode_bytes: bool = None
        # TODO maybe lack a constant.py
        self._indexes_tokens_deque = deque(maxlen=10)
        self.max_indexes_num = 5
        self.token2id = {}

    def _check_transformers_version(self, model_dir: str):
        import transformers
        from packaging import version

        from lmdeploy.archs import get_model_arch

        logger = get_logger('lmdeploy')

        current_transformers_version = version.parse(transformers.__version__)
        cfg = get_model_arch(model_dir)[1]
        cfg_ver = getattr(cfg, 'transformers_version', None)
        if cfg_ver is None:
            llm_config = getattr(cfg, 'llm_config', None)
            if llm_config:
                cfg_ver = getattr(llm_config, 'transformers_version', None)
        if cfg_ver is None:
            return
        required_transformers_version = version.parse(cfg_ver)
        if current_transformers_version < required_transformers_version:
            logger.warning(
                f'The current version of `transformers` is transformers=={current_transformers_version}, '  # noqa: E501
                f'which is lower than the required version transformers=={required_transformers_version}. '  # noqa: E501
                'Please upgrade to the required version.')

    def get_vocab(self):
        """Get vocab."""
        return self.model.get_vocab()

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self.model.vocab_size

    @property
    def vocab_size_with_added(self):
        """Vocabulary size with added vocab."""
        if self._vocab_size_with_added is not None:
            return self._vocab_size_with_added
        self._vocab_size_with_added = len(self.model.get_vocab())
        return self._vocab_size_with_added

    @property
    def bos_token_id(self):
        """Begin of the sentence token id."""
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        """End of the sentence token id."""
        return self.model.eos_token_id

    @property
    def prefix_space_tokens(self):
        """Tokens without prefix space."""
        if self._prefix_space_tokens is None:
            vocab = self.model.convert_ids_to_tokens(list(range(self.vocab_size)))
            self._prefix_space_tokens = {
                i
                for i, tok in enumerate(vocab) if tok.startswith('▁' if isinstance(tok, str) else b' ')
            }
        return self._prefix_space_tokens

    def _maybe_add_prefix_space(self, tokens: List[int], decoded: str):
        """Maybe add prefix space for incremental decoding."""
        if len(tokens) and not decoded.startswith(' ') and\
                tokens[0] in self.prefix_space_tokens:
            return ' ' + decoded
        else:
            return decoded

    @property
    def maybe_decode_bytes(self):
        """Check if self.model.convert_ids_to_tokens return not a str value."""
        if self._maybe_decode_bytes is None:
            self._maybe_decode_bytes = False
            vocab = self.model.convert_ids_to_tokens(list(range(self.vocab_size)))
            for tok in vocab:
                if not isinstance(tok, str):
                    self._maybe_decode_bytes = True
                    break
        return self._maybe_decode_bytes

    def indexes_containing_token(self, token: str):
        """Return all the possible indexes, whose decoding output may contain
        the input token."""
        # traversing vocab is time consuming, can not be accelerated with
        # multi threads (computation) or multi process (can't pickle tokenizer)
        # so, we maintain latest 10 stop words and return directly if matched
        for _token, _indexes in self._indexes_tokens_deque:
            if token == _token:
                return _indexes

        if self.token2id == {}:
            # decode is slower than convert_ids_to_tokens
            if self.maybe_decode_bytes:
                for i in range(self.vocab_size):
                    try:
                        self.token2id[self.model.decode(i)] = i
                    except:  # noqa: E722
                        # some tokens just can't be decoded by `decode`
                        pass
            else:
                self.token2id = {self.model.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        if token == ' ':  # ' ' is special
            token = '▁'
        indexes = [i for _token, i in self.token2id.items() if token in _token]
        if len(indexes) > self.max_indexes_num:
            # multiple id decode to same token
            indexes = [i for i in indexes if self.decode([i]) == token]
            indexes = indexes[:self.max_indexes_num]
            self.logger.warning(f'There are too many(>{self.max_indexes_num}) possible '
                                f'indexes may decoding {token}, we will use {indexes} only')
        # there might be token id that exceeds self.vocab_size
        if len(indexes) == 0:
            indexes = self.encode(token, False)
            if len(indexes) != 1:
                self.logger.warning(f'The token {token}, its length of indexes {indexes} is '
                                    'not 1. Currently, it can not be used as stop words')
                indexes = []
        self._indexes_tokens_deque.append((token, indexes))
        return indexes

    def encode(self, s: str, add_bos: bool = True, add_special_tokens: bool = True, **kwargs):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
            add_bos (bool): Whether to add `bos` token id when encoding
                the prompt
            add_special_tokens (bool): Whether or not to add special tokens
                when encoding the prompt
        Returns:
            list[int]: token ids
        """
        encoded = self.model.encode(s, add_special_tokens=add_special_tokens, **kwargs)
        if not add_bos:
            # in the middle of a session
            if len(encoded) and encoded[0] == self.bos_token_id:
                encoded = encoded[1:]
        return encoded

    def decode(self, t: Sequence[int], offset: Optional[int] = None, skip_special_tokens: bool = True):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
            offset (int): for incrementally decoding. Default to None, which
                means not applied.
            skip_special_tokens (bool): Whether or not to remove special
                tokens in the decoding.
        Returns:
            str: text of decoding tokens
        """
        t = t[offset:]
        out_string = self.model.decode(t, skip_special_tokens=skip_special_tokens)
        if offset:
            logger = get_logger('lmdeploy')
            logger.warning('For incrementally detokenization, please try '
                           'detokenize_incrementally function instead.')
            out_string = self._maybe_add_prefix_space(t, out_string)
        return out_string

    @staticmethod
    def _convert_tokens_to_string_with_added_encoders(
        tokenizer,
        output_tokens: List[str],
        skip_special_tokens: bool,
        spaces_between_special_tokens: bool,
    ) -> str:
        if tokenizer.is_fast or not tokenizer.get_added_vocab():
            return tokenizer.convert_tokens_to_string(output_tokens)
        # Adapted from
        # https://github.com/vllm-project/vllm/blob/v0.2.7/vllm/transformers_utils/tokenizer.py#L68-L99
        sub_texts = []
        current_sub_text = []
        all_special_tokens = set(tokenizer.all_special_tokens)
        for token in output_tokens:
            if skip_special_tokens and token in all_special_tokens:
                continue
            if token in tokenizer.get_added_vocab():
                if current_sub_text:
                    sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                    sub_texts.append(sub_text)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
            sub_texts.append(sub_text)
        if spaces_between_special_tokens:
            return ' '.join(sub_texts)
        else:
            return ''.join(sub_texts)

    # Based on
    # https://github.com/vllm-project/vllm/blob/v0.2.7/vllm/transformers_utils/tokenizer.py#L105-L165
    def detokenize_incrementally(self,
                                 all_input_ids: Sequence[int],
                                 state: DetokenizeState,
                                 skip_special_tokens: bool = True,
                                 spaces_between_special_tokens: bool = True):
        """Incrementally detokenize the input indexes.

        Args:
            all_input_ids (List[int]): a list of token ids. Expected to be
                different sections of a long sequence.
            state (DetokenizeState): an instance of DetokenizeState. Consists
                of incrementally decoding states.
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be True.
            spaces_between_special_tokens (bool): Whether or not to add spaces
                between special tokens. Default to be True.
        Returns:
            str: decoding output string of the current round.
            state (DetokenizeState): an instance of DetokenizeState. Consists
                of incrementally decoding states.
        """
        tokenizer = self.model
        ids_offset, prev_tokens, prefix_offset, read_offset = state.as_tuple()
        # This is the first iteration for this sequence
        new_tokens = tokenizer.convert_ids_to_tokens(all_input_ids[ids_offset:],
                                                     skip_special_tokens=skip_special_tokens)
        # `convert_ids_to_tokens` returns None for out-of-range token_id
        new_tokens = new_tokens or []
        new_tokens = [x for x in new_tokens if x is not None] if None in new_tokens else new_tokens
        if prev_tokens is None:
            # Please notice that in VLLM, indexes are detokenized one by one
            # while in LMDeploy, every turn, the detokenized indexes length
            # can be different.
            prev_tokens = tokenizer.convert_ids_to_tokens(all_input_ids[:ids_offset],
                                                          skip_special_tokens=skip_special_tokens)
            # `convert_ids_to_tokens` returns None for out-of-range token_id
            prev_tokens = prev_tokens or []
            prev_tokens = [x for x in prev_tokens if x is not None] if None in prev_tokens else prev_tokens
            read_offset = len(prev_tokens)
            if skip_special_tokens and new_tokens and new_tokens[0] in tokenizer.all_special_ids:
                read_offset = read_offset + 1  # skip special token

        output_tokens = prev_tokens + new_tokens
        prev_tokens += new_tokens
        prefix_text = self._convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:read_offset],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        new_text = self._convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

        # update state and get final decoded output
        if len(new_text) > len(prefix_text) and not new_text.endswith('�'):
            # utf-8 char at the end means it's a potential unfinished byte
            # sequence from byte fallback tokenization.
            # If it's in the middle, it's probably a real invalid id generated
            # by the model
            prefix_offset = read_offset
            read_offset = len(output_tokens)
            new_text = new_text[len(prefix_text):]
        else:
            new_text = ''

        return new_text, DetokenizeState(len(all_input_ids), prev_tokens, prefix_offset, read_offset)

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        add_special_tokens = False
        return self.model(s, add_special_tokens=add_special_tokens)


class ChatGLM4Tokenizer(HuggingFaceTokenizer):
    """Tokenizer of GLM4."""

    def __init__(self, model_path):
        super(ChatGLM4Tokenizer, self).__init__(model_path)
        original_pad = self.model._pad

        def __pad(*args, **kwargs):
            if 'padding_side' in kwargs:
                kwargs.pop('padding_side')
            return original_pad(*args, **kwargs)

        # fix for transformers>4.45.0
        self.model._pad = __pad

    def encode(self, s: str, add_bos: bool = True, add_special_tokens: bool = True, **kwargs):
        """Tokenize a prompt."""
        # ChtGLM4Tokenizer hardcode `add_speical_tokens=False` when tokenizing
        # a prompt. Refer to https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/tokenization_chatglm.py#L227 # noqa E501
        return super(ChatGLM4Tokenizer, self).encode(s, add_bos, add_special_tokens=False, **kwargs)


class ChatGLMTokenizer(HuggingFaceTokenizer):
    """Tokenizer of GLM2."""

    def __init__(self, model_path):
        super(ChatGLMTokenizer, self).__init__(model_path)
        original_pad = self.model._pad

        def __pad(*args, **kwargs):
            if 'padding_side' in kwargs:
                kwargs.pop('padding_side')
            return original_pad(*args, **kwargs)

        # fix for transformers>4.45.0
        self.model._pad = __pad


class GptOssTokenizer(HuggingFaceTokenizer):
    """Tokenizer of GPT-OSS."""

    def __init__(self, model_dir: str):
        super(GptOssTokenizer, self).__init__(model_dir)
        from openai_harmony import HarmonyEncodingName, Role, StreamableParser, load_harmony_encoding
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.role = Role.ASSISTANT
        self.parser = partial(StreamableParser, encoding, role=Role.ASSISTANT)

    def detokenize_incrementally(self,
                                 all_input_ids: Sequence[int],
                                 state: DetokenizeState,
                                 skip_special_tokens: bool = True,
                                 spaces_between_special_tokens: bool = True):
        if not hasattr(state, 'stream'):
            state.stream = self.parser()

        response = ''
        stream = state.stream
        for token_id in all_input_ids[state.ids_offset:]:
            stream.process(token_id)
            if stream.current_channel in ['final', 'analysis'] and stream.current_role == self.role:
                response += stream.last_content_delta or ''

        state.ids_offset = len(all_input_ids)
        return response, state


class GGUFTokenizer:
    """Tokenizer built from GGUF file metadata (tokens + BPE merges).

    Reads ``tokenizer.ggml.*`` fields from a GGUF file and constructs a
    ``tokenizers.Tokenizer`` with a BPE model.  Provides the same public
    interface as :class:`HuggingFaceTokenizer`.

    Args:
        gguf_path (str): path to a ``.gguf`` file (or first shard).
    """

    def __init__(self, gguf_path: str):
        import os
        import time as _time
        import gguf as _gguf
        from tokenizers import Tokenizer as _Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        self.logger = get_logger('lmdeploy')

        # --- locate the GGUF file ---
        if os.path.isdir(gguf_path):
            candidates = sorted(f for f in os.listdir(gguf_path)
                                if f.endswith('.gguf'))
            if not candidates:
                raise FileNotFoundError(
                    f'No .gguf files in {gguf_path}')
            gguf_path = os.path.join(gguf_path, candidates[0])

        self.logger.info(f'GGUFTokenizer: opening {gguf_path}')
        _t0 = _time.time()
        reader = _gguf.GGUFReader(gguf_path)
        self.logger.info(f'GGUFTokenizer: GGUFReader done in '
                         f'{_time.time() - _t0:.1f}s')
        fields = reader.fields
        _extract = self._extract_field_value

        # --- read tokenizer metadata ---
        tok_model = _extract(fields.get('tokenizer.ggml.model'))
        self.logger.info(f'GGUFTokenizer: tokenizer model = {tok_model}')
        tokens_field = fields.get('tokenizer.ggml.tokens')
        merges_field = fields.get('tokenizer.ggml.merges')
        types_field = fields.get('tokenizer.ggml.token_type')

        if tokens_field is None:
            raise ValueError(
                'GGUF file has no tokenizer.ggml.tokens metadata')

        # Extract token list
        _t1 = _time.time()
        tokens = [bytes(tokens_field.parts[idx]).decode('utf-8')
                  for idx in tokens_field.data]
        self.logger.info(f'GGUFTokenizer: extracted {len(tokens)} tokens '
                         f'in {_time.time() - _t1:.1f}s')

        # Extract merge rules (may be absent for unigram/sentencepiece)
        merges = []
        if merges_field is not None and len(merges_field.data) > 0:
            _t2 = _time.time()
            merges = [bytes(merges_field.parts[idx]).decode('utf-8')
                      for idx in merges_field.data]
            self.logger.info(f'GGUFTokenizer: extracted {len(merges)} merges '
                             f'in {_time.time() - _t2:.1f}s')

        # Extract token types (1=normal, 3=control, 6=byte, etc.)
        token_types = None
        if types_field is not None:
            token_types = [types_field.parts[idx][0].item()
                           for idx in types_field.data]

        # --- special token ids ---
        self._bos_token_id = _extract(
            fields.get('tokenizer.ggml.bos_token_id'))
        self._eos_token_id = _extract(
            fields.get('tokenizer.ggml.eos_token_id'))
        self._pad_token_id = _extract(
            fields.get('tokenizer.ggml.padding_token_id'))
        self._unk_token_id = _extract(
            fields.get('tokenizer.ggml.unknown_token_id'))
        self._add_bos = bool(_extract(
            fields.get('tokenizer.ggml.add_bos_token')) or False)

        # --- chat template ---
        self.chat_template = _extract(
            fields.get('tokenizer.chat_template'))

        # --- build vocab dict ---
        vocab = {tok: i for i, tok in enumerate(tokens)}
        self._tokens = tokens
        self._vocab = vocab
        self._vocab_size = len(tokens)

        # Identify special / control token ids
        self._special_token_ids = set()
        if token_types is not None:
            for i, tt in enumerate(token_types):
                if tt == 3:  # control token
                    self._special_token_ids.add(i)
        # Always include bos/eos/pad/unk
        for tid in (self._bos_token_id, self._eos_token_id,
                    self._pad_token_id, self._unk_token_id):
            if tid is not None:
                self._special_token_ids.add(tid)

        # --- build tokenizers.Tokenizer ---
        _t3 = _time.time()
        merge_tuples = []
        for m in merges:
            parts = m.split(' ', 1)
            if len(parts) == 2:
                merge_tuples.append(tuple(parts))

        bpe = BPE(vocab=vocab, merges=merge_tuples)
        self.logger.info(f'GGUFTokenizer: BPE model built in '
                         f'{_time.time() - _t3:.1f}s')
        self._tokenizer = _Tokenizer(bpe)

        # GPT-2 style BPE uses byte-level pre-tokenizer
        if tok_model in ('gpt2', None):
            self._tokenizer.pre_tokenizer = ByteLevel(
                add_prefix_space=False)
            self._tokenizer.decoder = ByteLevelDecoder()

        # --- caches ---
        self._prefix_space_tokens = None
        self._vocab_size_with_added: int = None
        self._maybe_decode_bytes: bool = None
        self._indexes_tokens_deque = deque(maxlen=10)
        self.max_indexes_num = 5
        self.token2id = {}

        # Provide a .model attribute for code that accesses
        # tokenizer.model.model (e.g. api_server logprobs).
        self.model = self
        self.logger.info(f'GGUFTokenizer: init complete in '
                         f'{_time.time() - _t0:.1f}s, '
                         f'vocab_size={self._vocab_size}')

    # ------------------------------------------------------------------
    # Static helper (reuse GGUF field extraction logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_field_value(field):
        """Convert a GGUF ReaderField to a plain Python value."""
        if field is None:
            return None
        from gguf import GGUFValueType
        if not field.types:
            return None
        vtype = field.types[0]
        if vtype == GGUFValueType.STRING:
            return bytes(field.parts[-1]).decode('utf-8')
        if vtype == GGUFValueType.ARRAY:
            if len(field.types) < 2:
                return []
            elem_type = field.types[1]
            if elem_type == GGUFValueType.STRING:
                return [bytes(field.parts[idx]).decode('utf-8')
                        for idx in field.data]
            return [field.parts[idx][0].item() for idx in field.data]
        if field.data:
            return field.parts[field.data[0]][0].item()
        return field.parts[-1][0].item()

    # ------------------------------------------------------------------
    # Public API (mirrors HuggingFaceTokenizer)
    # ------------------------------------------------------------------

    def get_vocab(self):
        """Get vocab."""
        return dict(self._vocab)

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self._vocab_size

    @property
    def vocab_size_with_added(self):
        """Vocabulary size with added vocab."""
        if self._vocab_size_with_added is not None:
            return self._vocab_size_with_added
        self._vocab_size_with_added = len(self._vocab)
        return self._vocab_size_with_added

    @property
    def bos_token_id(self):
        """Begin of the sentence token id."""
        return self._bos_token_id

    @property
    def eos_token_id(self):
        """End of the sentence token id."""
        return self._eos_token_id

    @property
    def prefix_space_tokens(self):
        """Token ids whose decoded form starts with a space."""
        if self._prefix_space_tokens is None:
            self._prefix_space_tokens = set()
            for i, tok in enumerate(self._tokens):
                if i < self._vocab_size and tok.startswith('Ġ'):
                    self._prefix_space_tokens.add(i)
        return self._prefix_space_tokens

    @property
    def maybe_decode_bytes(self):
        """Check if tokens contain non-str values (always False here)."""
        return False

    # ------------------------------------------------------------------
    # Compatibility: convert_ids_to_tokens / convert_tokens_to_string
    # These are used by detokenize_incrementally and api_server logprobs.
    # ------------------------------------------------------------------

    @property
    def all_special_tokens(self):
        """Set of special token strings."""
        return {self._tokens[i] for i in self._special_token_ids
                if i < len(self._tokens)}

    @property
    def all_special_ids(self):
        """Set of special token ids."""
        return set(self._special_token_ids)

    @property
    def is_fast(self):
        """Mimic HF tokenizer attribute."""
        return True

    def get_added_vocab(self):
        """Return empty dict (no added vocab beyond base)."""
        return {}

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Convert token id(s) to token string(s)."""
        if isinstance(ids, int):
            if ids < 0 or ids >= len(self._tokens):
                return None
            if skip_special_tokens and ids in self._special_token_ids:
                return None
            return self._tokens[ids]
        result = []
        for i in ids:
            if i < 0 or i >= len(self._tokens):
                result.append(None)
            elif skip_special_tokens and i in self._special_token_ids:
                continue
            else:
                result.append(self._tokens[i])
        return result

    def convert_tokens_to_string(self, tokens):
        """Convert a list of token strings back to a readable string."""
        if not tokens:
            return ''
        # Join tokens and let the tokenizer decoder handle byte-level
        ids = []
        for tok in tokens:
            tid = self._vocab.get(tok)
            if tid is not None:
                ids.append(tid)
        if not ids:
            return ''
        return self._tokenizer.decode(ids)

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    def encode(self, s: str, add_bos: bool = True,
               add_special_tokens: bool = True, **kwargs):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
            add_bos (bool): Whether to add bos token id
            add_special_tokens (bool): Whether to add special tokens
        Returns:
            list[int]: token ids
        """
        encoded = self._tokenizer.encode(s).ids
        if add_bos and self._add_bos and self._bos_token_id is not None:
            if not encoded or encoded[0] != self._bos_token_id:
                encoded = [self._bos_token_id] + list(encoded)
        if not add_bos:
            if encoded and encoded[0] == self._bos_token_id:
                encoded = encoded[1:]
        return encoded

    def decode(self, t: Sequence[int], offset: Optional[int] = None,
               skip_special_tokens: bool = True):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
            offset (int): for incrementally decoding.
            skip_special_tokens (bool): Whether to remove special tokens.
        Returns:
            str: decoded text
        """
        t = list(t)
        if offset is not None:
            t = t[offset:]
        if skip_special_tokens:
            t = [i for i in t if i not in self._special_token_ids]
        if not t:
            return ''
        return self._tokenizer.decode(t)

    def _maybe_add_prefix_space(self, tokens: List[int], decoded: str):
        """Maybe add prefix space for incremental decoding."""
        if len(tokens) and not decoded.startswith(' ') and \
                tokens[0] in self.prefix_space_tokens:
            return ' ' + decoded
        return decoded

    def detokenize_incrementally(self,
                                 all_input_ids: Sequence[int],
                                 state: DetokenizeState,
                                 skip_special_tokens: bool = True,
                                 spaces_between_special_tokens: bool = True):
        """Incrementally detokenize the input indexes.

        Uses the same algorithm as HuggingFaceTokenizer but delegates
        to our own convert_ids_to_tokens / convert_tokens_to_string.
        """
        tokenizer = self  # self acts as the inner tokenizer
        ids_offset, prev_tokens, prefix_offset, read_offset = \
            state.as_tuple()

        new_tokens = tokenizer.convert_ids_to_tokens(
            all_input_ids[ids_offset:],
            skip_special_tokens=skip_special_tokens)
        new_tokens = new_tokens or []
        new_tokens = [x for x in new_tokens if x is not None] \
            if None in new_tokens else new_tokens

        if prev_tokens is None:
            prev_tokens = tokenizer.convert_ids_to_tokens(
                all_input_ids[:ids_offset],
                skip_special_tokens=skip_special_tokens)
            prev_tokens = prev_tokens or []
            prev_tokens = [x for x in prev_tokens if x is not None] \
                if None in prev_tokens else prev_tokens
            read_offset = len(prev_tokens)
            if (skip_special_tokens and new_tokens
                    and new_tokens[0] in tokenizer.all_special_ids):
                read_offset = read_offset + 1

        output_tokens = prev_tokens + new_tokens
        prev_tokens += new_tokens

        prefix_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:read_offset])
        new_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:])

        if len(new_text) > len(prefix_text) and \
                not new_text.endswith('\ufffd'):
            prefix_offset = read_offset
            read_offset = len(output_tokens)
            new_text = new_text[len(prefix_text):]
        else:
            new_text = ''

        return new_text, DetokenizeState(
            len(all_input_ids), prev_tokens, prefix_offset, read_offset)

    def indexes_containing_token(self, token: str):
        """Return all possible indexes whose decoding may contain token."""
        for _token, _indexes in self._indexes_tokens_deque:
            if token == _token:
                return _indexes

        if self.token2id == {}:
            self.token2id = {tok: i for i, tok in enumerate(self._tokens)
                             if i < self._vocab_size}

        if token == ' ':
            token = 'Ġ'
        indexes = [i for _token, i in self.token2id.items()
                   if token in _token]
        if len(indexes) > self.max_indexes_num:
            indexes = [i for i in indexes if self.decode([i]) == token]
            indexes = indexes[:self.max_indexes_num]
            self.logger.warning(
                f'Too many(>{self.max_indexes_num}) possible indexes '
                f'for {token}, using {indexes} only')
        if len(indexes) == 0:
            indexes = self.encode(token, False)
            if len(indexes) != 1:
                self.logger.warning(
                    f'Token {token}, indexes {indexes} length is not 1. '
                    'Cannot be used as stop words')
                indexes = []
        self._indexes_tokens_deque.append((token, indexes))
        return indexes

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        if isinstance(s, str):
            return self._tokenizer.encode(s)
        return self._tokenizer.encode_batch(s)


class Tokenizer:
    """Tokenize prompts or de-tokenize tokens into texts.

    Args:
        model_path (str): the path of the tokenizer model
    """

    def __init__(self, model_path: str):
        # GGUF file detection — use GGUFTokenizer directly.
        from lmdeploy.archs import is_gguf_model
        if is_gguf_model(model_path):
            self.model = GGUFTokenizer(model_path)
            self.logger = get_logger('lmdeploy')
            return

        from transformers import AutoConfig, PretrainedConfig
        try:
            model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:  # noqa
            model_cfg = PretrainedConfig.from_pretrained(model_path, trust_remote_code=True)
        is_gpt_oss = getattr(model_cfg, 'model_type', '') == 'gpt_oss'
        from transformers.models.auto.tokenization_auto import get_tokenizer_config
        tokenizer_config = get_tokenizer_config(model_path, trust_remote_code=True)
        config_tokenizer_class = tokenizer_config.get('tokenizer_class')
        if config_tokenizer_class == 'ChatGLM4Tokenizer':
            self.model = ChatGLM4Tokenizer(model_path)
        elif config_tokenizer_class == 'ChatGLMTokenizer':
            self.model = ChatGLMTokenizer(model_path)
        elif is_gpt_oss:
            self.model = GptOssTokenizer(model_path)
        else:
            self.model = HuggingFaceTokenizer(model_path)
        self.logger = get_logger('lmdeploy')

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self.model.vocab_size

    @property
    def bos_token_id(self):
        """Begin of the sentence token id."""
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        """End of the sentence token id."""
        return self.model.eos_token_id

    def get_vocab(self):
        """Get vocab."""
        return self.model.get_vocab()

    def encode(self, s: str, add_bos: bool = True, add_special_tokens: bool = True, **kwargs):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
            add_bos (bool): Whether to add `bos` token id when encoding
                the prompt
            add_special_tokens (bool): Whether or not to add special tokens
                when encoding the prompt
        Returns:
            list[int]: token ids
        """
        encoded = self.model.encode(s, add_bos, add_special_tokens, **kwargs)
        if encoded[:2] == [self.bos_token_id] * 2:
            self.logger.warning(f'Detected duplicate bos token {self.bos_token_id} in prompt, '
                                'this will likely reduce response quality, one of them will be'
                                'removed')
            encoded = encoded[1:]
        return encoded

    def decode(
        self,
        t: Sequence[int],
        offset: Optional[int] = None,
        skip_special_tokens: bool = True,
    ):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
            offset (int): for incrementally decoding. Default to None, which
                means not applied.
            skip_special_tokens (bool): Whether or not to remove special
                tokens in the decoding.
        Returns:
            str: text of decoding tokens
        """
        return self.model.decode(t, offset, skip_special_tokens)

    def detokenize_incrementally(self,
                                 all_input_ids: Sequence[int],
                                 state: DetokenizeState,
                                 skip_special_tokens: bool = True,
                                 spaces_between_special_tokens: bool = True):
        """Incrementally detokenize the input indexes.

        Args:
            all_input_ids (List[int]): a list of token ids. Expected to be
                different sections of a long sequence.
            state (DetokenizeState): an instance of DetokenizeState. Consists
                of incrementally decoding states.
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be True.
            spaces_between_special_tokens (bool): Whether or not to add spaces
                between special tokens. Default to be True.
        Returns:
            str: decoding output string of the current round.
            state (DetokenizeState): an instance of DetokenizeState. Consists
                of incrementally decoding states.
        """
        return self.model.detokenize_incrementally(all_input_ids,
                                                   state=state,
                                                   skip_special_tokens=skip_special_tokens,
                                                   spaces_between_special_tokens=spaces_between_special_tokens)

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        return self.model(s)

    def indexes_containing_token(self, token):
        """Return all the possible indexes, whose decoding output may contain
        the input token."""
        encoded = self.encode(token, add_bos=False)
        if len(encoded) > 1:
            self.logger.warning(f'The token {token}, its length of indexes {encoded} is over '
                                'than 1. Currently, it can not be used as stop words')
            return []
        return self.model.indexes_containing_token(token)
