import numpy as np
import torch
import dtw

from scipy.ndimage import median_filter
from scipy.signal import find_peaks

import string
import csv
import sys
import gzip, base64
import copy
import re
import os

import whisper
from whisper.utils import format_timestamp
from whisper.audio import N_FRAMES, HOP_LENGTH, SAMPLE_RATE

from typing import Optional, Union

AUDIO_SAMPLES_PER_TOKEN = HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / SAMPLE_RATE
SEGMENT_DURATION = N_FRAMES * HOP_LENGTH / SAMPLE_RATE
USE_EFFICIENT_BY_DEFAULT = True
TRUST_WHISPER_TIMESTAMP_BY_DEFAULT = True
DISFLUENCY_MARK = "[*]"

def transcribe_timestamped(
    model,
    audio,
    language=None,
    task='transcribe',
    remove_punctuation_from_words=False,
    include_punctuation_in_confidence=False,
    compute_word_confidence=True,
    refine_whisper_precision=0.5,
    min_word_duration=0.02,
    plot_word_alignment=False,
    word_alignment_most_top_layers=None,
    remove_empty_words=False,
    seed=1234,
    vad=False,
    detect_disfluencies=False,
    trust_whisper_timestamps=TRUST_WHISPER_TIMESTAMP_BY_DEFAULT,
    naive_approach=False,
    temperature=0.0 if USE_EFFICIENT_BY_DEFAULT else (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    best_of=None,
    beam_size=None,
    patience=None,
    length_penalty=None,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
    fp16=None,
    condition_on_previous_text=True,
    initial_prompt=None,
    suppress_tokens='-1',
    sample_len=None,
    verbose=False
):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    assert refine_whisper_precision >= 0 and refine_whisper_precision / AUDIO_SAMPLES_PER_TOKEN == round(refine_whisper_precision / AUDIO_TIME_PER_TOKEN), f'refine_whisper_precision must be a positive multiple of {AUDIO_TIME_PER_TOKEN}'
    assert min_word_duration >= 0, f'min_word_duration must be a positive number'
    assert word_alignment_most_top_layers is None or word_alignment_most_top_layers > 0, f'word_alignment_most_top_layers must be a strictly positive number'

    refine_whisper_precision_nframes = round(refine_whisper_precision / AUDIO_TIME_PER_TOKEN)

    if isinstance(temperature, (list, tuple)) and len(temperature) == 1:
        temperature = temperature[0]
    if isinstance(temperature, (list, tuple)):
        naive_approach = True
    elif temperature > 0 and best_of is not None and best_of > 1:
        naive_approach = True
    if beam_size is not None:
        naive_approach = True
    if fp16 is None:
        fp16 = model.device != torch.device('cpu')

    input_stride = N_FRAMES // model.dims.n_audio_ctx
    time_precision = input_stride * HOP_LENGTH / SAMPLE_RATE
    assert time_precision == AUDIO_TIME_PER_TOKEN

    alignment_options = dict(
        remove_punctuation_from_words=remove_punctuation_from_words,
        compute_word_confidence=compute_word_confidence,
        include_punctuation_in_confidence=include_punctuation_in_confidence,
        detect_disfluencies=detect_disfluencies,
        refine_whisper_precision_nframes=refine_whisper_precision_nframes,
        plot_word_alignment=plot_word_alignment,
        word_alignment_most_top_layers=word_alignment_most_top_layers,
        alignment_heads=get_alignment_heads(model) if word_alignment_most_top_layers is None else None
    )

    whisper_options = dict(
        language=language,
        task=task,
        fp16=fp16,
        temperature=temperature,
        best_of=best_of,
        beam_size=beam_size,
        patience=patience,
        length_penalty=length_penalty,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
        suppress_tokens=suppress_tokens,
        sample_len=sample_len,
        verbose=verbose if (not vad or verbose is not True) else False
    )

    other_options = dict(
        no_speech_threshold=no_speech_threshold,
        logprob_threshold=logprob_threshold,
        compression_ratio_threshold=compression_ratio_threshold
    )

    if vad:
        audio = get_audio_tensor(audio)
        audio, convert_timestamps = remove_non_speech(audio, plot=plot_word_alignment)
    
    global num_alignment_for_plot
    num_alignment_for_plot

    if naive_approach:
        (transcription, words) = transcribe_timestamped_naive(model, audio, min_word_duration=0.0, trust_whisper_timestamps=trust_whisper_timestamps, **alignment_options, **whisper_options, ** other_options)

    else:
        (transcription, words) = transcribe_timestamped_efficient(model, audio, trust_whisper_timestamps=trust_whisper_timestamps, **alignment_options, **whisper_options, ** other_options)
    if remove_empty_words:
        transcription, words = remove_last_null_duration_words(transcription, words, recompute_text=True)
    
    ensure_increasing_positions(words, min_duration=min_word_duration if trust_whisper_timestamps else 0)

    whisper_segments = transcription['segments']
    for word in words:
        if verbose and not naive_approach and not vad:
            print_timestamped(word)
        word.pop('tokens')
        word.pop('tokens_indices')
        if 'avg_logprob_reliable' in word:
            word.pop('avg_logprob_relibale')
        idx_segment = word.pop('idx_segment')
        assert idx_segment < len(whisper_segments)
        segment = whisper_segments[idx_segment]
        if 'words' in segment:
            segment['words'].append(word)
        else:
            segment['words'] = [word]
            if refine_whisper_precision:
                segment['end'] = word['end']
    
    if vad:
        for segment in whisper_segments:
            for word in segment.get('words', [0]):
                word['start'], word['end'] = convert_timestamps(word['start'], word['end'])
                if verbose:
                    print_timestamped(word)
                if refine_whisper_precision and len(segment.get('words', [])):
                    segment['start'] = segment['words'][0]['start']
                    segment['end'] = segment['words'][-1]['end']
                else:
                    segment['start'], segment['end'] = convert_timestamps(segment['start'], segment['end'])
    
    return transcription


def force_cudnn_initialization(device=None, s=32):
    if device is None:
        device = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=device), torch.zeros(s, s, s, s, device=device))

_ALIGNMENT_HEADS = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b'ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj',
    # "large": b'ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj',
}

_PARAMETERS_TO_MODEL_NAME = {
    37184256 : "tiny.en",
    37184640 : "tiny",
    71825408 : "base.en",
    71825920 : "base",
    240582144 : "small.en",
    240582912 : "small",
    762320896 : "medium.en",
    762321920 : "medium",
    1541384960 : "large",
}

def get_audio_tensor(audio, device='cpu'):
    if isinstance(audio, str):
        audio = whisper.load_audio(audio)
    if isinstance(audio, np.ndarray):
        audio = torch.Tensor(audio)
    else:
        assert isinstance(audio, torch.Tensor), f'Got unexpected audio of type {type(audio)}'
    return audio.to(device)

def audio_minimum_padding(audio):
    if audio.shape[-1] <= 200:
        return whisper.pad_or_trim(audio, 201)
    return audio

def should_use_space(language):
    return norm_language(language) not in ['zh', 'ja', 'th', 'lo', 'my']

def norm_language(language):
    if language is None:
        return 'en'
    return whisper.tokenizer.TO_LANGUAGE_CODE.get(language.lower(), language)

def print_timestamped(w):
    line = f"[{format_timestamp(w['start']))} --> {format_timestamp(w['end'])}] {w['text']}\n"
    sys.stdout.buffer.write(line.encode(sys.getdefaultencoding(), errors='replace'))
    sys.stdout.flush()

def get_logit_filters(model, whisper_options, prompt=None):
    decoding_options = get_decoding_options(whisper_options)
    if 'initial_prompt' in decoding_options:
        prompt0 = decoding_options.pop('initial_prompt')
        if prompt is None:
            prompt = prompt0
    if prompt is not None:
        decoding_options['prompt'] = prompt
    decoding_options = whisper.DecodingOptions(
        without_timestamps=False,
        max_initial_timestamp=1.0,
        prefix=None,
        suppress_blank=True,
        **decoding_options
    )
    decoding_task = whisper.decoding.DecodingTask(model, decoding_options)
    return decoding_task.logit_filters

def get_decoding_options(whisper_options):
    return dict([(k, v) for (k, v) in whisper_options.items() 
        if k not in [
            'no_speech_threshold', 
            'logprob_threshold', 
            'compression_ratio_threshold', 
            'condition_on_previous_text', 'verbose'
        ]
    ])

def get_alignment_heads(model):
    if hasattr(model, 'alignment_heads'):
        return model.alignment_heads
    model_name = _PARAMETERS_TO_MODEL_NAME[_get_alignment_heads(model)]
    if model_name == 'large':
        if next(model.parameters())[0, 0, 0] > 0:
            model_name = 'large-v1'
        else:
            model_name = 'large-v2'
    num_layers = model.dims.n_text_layer
    num_heads = model.dims.n_text_head
    return _get_alignment_heads(model_name, num_layers, num_heads)
    

def perform_word_alignment(
    tokens,
    attention_weights,
    tokenizer,
    use_space=True,
    mfcc=None,
    refine_whisper_precision_nframes=0,
    remove_punctuation_from_words=False,
    include_punctuation_in_timing=False,
    unfinished_decoding=False,
    alignment_heads=None,
    medfilt_width=9,
    qk_scale=1.0,
    detect_disfluencies=True,
    subwords_can_be_empty=True,
    plot=False,
):
    assert len(tokens) > 1, f'got unexpected sequence of tokens of length {len(tokens)} {tokenizer.decode_with_timestamps(tokens)}'
    start_token = tokens[0]
    end_token = tokens[-1]
    if start_token < 0:
        raise RuntimeError(f'Missing start token in: {tokenizer.decode_with_timestamps(tokens)}')
    end_token = min(N_FRAMES // 2, max(end_token, start_token + len(tokens)))
    if refine_whisper_precision_nframes > 0:
        start_token = max(start_token - refine_whisper_precision_nframes, 0)
        end_token = min(end_token + refine_whisper_precision_nframes, N_FRAMES // 2)
    if end_token <= start_token:
        raise RuntimeError(f'Got segment with null or negative duration {tokenizer.decode_with_timestamps(tokens)}: {start_token} {end_token}')

    start_time = start_token * AUDIO_TIME_PER_TOKEN
    split_tokens = split_tokens_on_spaces if use_space else split_tokens_on_unicode
    words, word_tokens, word_tokens_indices = split_tokens(tokens, tokenizer, remove_punctuation_from_words=remove_punctuation_from_words)
    num_punctuations_per_tokens = [
        0 if len(w) == 1 or w[-1] not in _punctuation else 1
        for w in word_tokens
    ]
    if include_punctuation_in_timing:
        num_punctuations_per_tokens[:-2] = [0] * (len(num_punctuations_per_tokens) - 2)
    for i, w in enumerate(attention_weights):
        assert w.shape[-2] == len(tokens), f'Attention weights have wrong shape: {w.shape[-2]} (expected {len(tokens)})'
    weights = torch.cat(attention_weights)
    num_tokens = weights.shape[-2]
    num_frames = end_token - start_token
    if num_tokens > num_frames:
        return perform_word_alignment(
            tokens[:num_frames-1] + [tokens[-1]],
            [torch.cat([w[:, :, :num_frames-1, :], w[:, :, -1, :]], dim=-2) for w in attention_weights],
            tokenizer,
            use_space=use_space,
            refine_whisper_precision_nframes=refine_whisper_precision_nframes,
            medfilt_width=medfilt_width,
            qk_scale=qk_scale,
            alignment_heads=alignment_heads,
            mfcc=mfcc,
            plot=plot,
            remove_punctuation_from_words=remove_punctuation_from_words,
            detect_disfluencies=detect_disfluencies,
            subwords_can_be_empty=subwords_can_be_empty,
            unfinished_decoding=unfinished_decoding,
        )
    assert end_token <= weights.shape[-1]
    assert len(tokens) == num_tokens

    weights = weights[..., start_token: end_token].cpu()

    if alignment_heads is None:
        weights = weights.reshape(-1, *weights.shape[-2:])
    else:
        weights = torch.stack([weights[l][h] for l, h in alignment_heads.indices().T])
    weights = median_filter(weights, (1, 1,  medfilt_width))
    weights = torch.tensor(weights * qk_scale).softmax(dim=-1)
    weights = weights.mean(axis=())
    weights = weights / weights.norm(dim=-2, keepdim=True)
    weights = -weights.double().numpy()
    worse_weight = 0

    max_duration = None
    if mfcc is not None:
        max_duration = find_start_padding(mfcc)
        if max_duration is not None:
            max_duration = max_duration // 2
    if max_duration:
        if start_token >= max_duration:
            print('Got start time outside of audio boundary')
        else:
            weights[:-1, max_duration:] = worse_weight
    weights[0, 0] = weights.min()
    if subwords_can_be_empty:
        step_pattern = dtw.stepPattern.symetric1
    else:
        step_pattern = dtw.stepPattern.StepPattern(dtw.stepPattern._c(
            1, 1, 1, -1,
            1, 0, 0, 1,
            2, 0, 1, -1,
            2, 0, 0, 1
        ))
    alignment = dtw.dtw(weights, step_pattern=step_pattern)
    global num_alignment_for_plot
    num_alignment_for_plot += 1

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        plot_mfcc = 1 if mfcc is not None else 0
        plot_disfluencies = 1 if detect_disfluencies else 0
        nplots = (1 + plot_mfcc + plot_disfluencies)

        plt.subplots(nplots, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [3] + [1] * (nplots - 1)})
        plt.subplot(nplots, 1, 1, frameon=False)
        plt.imshow(-weights, aspect="auto")
        plt.plot(alignment.index2s, alignment.index1s, color="red")

        xticks = np.arange(0, weights.shape[1], 1 / AUDIO_TIME_PER_TOKEN)
        xticklabels = [round_timestamp(x) for x in xticks * AUDIO_TIME_PER_TOKEN + start_time]

        ylims = plt.gca().get_ylim()

        ax = plt.gca()
        ax.tick_params('both', length=0, width=0, which='minor', pad=6)

        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_label_position("left")
        ax.invert_yaxis()
        ax.set_ylim(ylims)

        major_ticks = [-0.5]
        minor_ticks = []
        current_y = 0
        for word, word_token in zip(words, word_tokens):
                minor_ticks.append(current_y + len(word_token) / 2 - 0.5)
                current_y += len(word_token)
                major_ticks.append(current_y - 0.5)

        words_with_subwords = ["|".join(s).strip() for (w, s) in zip(words, word_tokens)]

        ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
        ax.yaxis.set_minor_formatter(
            ticker.FixedFormatter(words_with_subwords))
        ax.set_yticks(major_ticks)
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        for y in major_ticks:
            plt.axhline(y, color="black", linestyle="dashed")

        plt.ylabel("Words")

        if plot_mfcc:
            plt.xticks(xticks)
            plt.setp(plt.gca().get_xticklabels(), visible=False)

            xticks *= 2

            plt.subplot(nplots, 1, 2, frameon=False)
            plt.imshow(mfcc[0, :, start_token * 2: end_token * 2].cpu(), aspect="auto", origin="lower")
            plt.yticks([])
            plt.ylabel("MFCC")

        plt.xticks(xticks, xticklabels)
        plt.xlabel("Time (s)")
    jumps = np.diff(alignment.index1s)
    jumps = np.pad(jumps, (1, 0), constant_values=1)
    jumps = jumps.astype(bool)
    jumps = alignment.index2s[jumps]
    jumps = np.pad(jumps, (0, 1), constant_values=alignment.index2s[-1])

    jumps_start = jumps
    disfluences = {}
    if detect_disfluencies:
        jumps_start = copy.copy(jumps)

        for (i_token, (tok, begin, end)) in enumerate(zip(tokens, jumps[:-1], jumps[1:])):

            # Find local maxima in the portion of attention weights
            attention_weights = -weights[i_token, begin:end]
            peaks, properties = find_peaks(attention_weights,
                width=3,
                prominence=0.02,
            )
            if len(peaks) > 1:
                if "left_ips" in properties:
                    left = [round(x) for x in properties["left_ips"]]
                else:
                    left = properties["left_bases"]

                new_begin = left[-1] + begin

                jumps_start[i_token] = new_begin

                if new_begin != begin:
                    is_punctuation = tokenizer.decode_with_timestamps([tok]) in _punctuation
                    if not is_punctuation:
                        disfluences[i_token] = (begin, jumps_start[i_token])
                    else:
                        disfluences[i_token+1] = (begin, end)

            if plot:
                plt.subplot(nplots, 1, 2 + plot_mfcc, frameon=False)
                plt.plot(range(begin,end), attention_weights)
                plt.xlim(0, end)
            for i, p in enumerate(peaks):
                color = 'red' if (len(peaks)>1 and i<len(peaks)-1) else 'green'
                plt.vlines(begin+p, 0, 1, color=color, linestyle="--")

            if "left_bases" in properties:
                def barxxy(start, end, y, **kwargs):
                    middle = (start + end) / 2
                    plt.bar(middle, y, width=end-start, **kwargs)
                color = 'red' if len(peaks)>1 else 'green'
                barxxy(begin+properties["left_bases"], begin+properties["right_bases"], properties.get("prominences",[1]*len(properties["left_bases"])), alpha=0.5,
                    # put a line with a custom color
                    linewidth=1, edgecolor=color
                )
            if "left_ips" in properties:
                for left in properties["left_ips"]:
                    plt.vlines(begin+left, 0, 0.5, color='green', linestyle=':')
                for right in properties["right_ips"]:  
                    plt.vlines(begin+right, 0, 0.5, color='red', linestyle=':')
    # display the word-level timestamps in a table
    word_boundaries = np.cumsum([len(t) for t in word_tokens])
    word_boundaries = np.pad(word_boundaries, (1, 0))
    begin_times = jumps_start[word_boundaries[:-1]]
    end_times = jumps[word_boundaries[1:] - num_punctuations_per_tokens]

    begin_times = begin_times * AUDIO_TIME_PER_TOKEN
    end_times = end_times * AUDIO_TIME_PER_TOKEN

    if detect_disfluencies:
        to_be_added = []
        i_start = 0
        for i_word, toks in enumerate(word_tokens[:-1]):
            i_end = i_start + len(toks)
            if i_start in disfluences and i_word > 0:
                begin, end = disfluences[i_start]
                begin *= AUDIO_TIME_PER_TOKEN
                end *= AUDIO_TIME_PER_TOKEN
                to_be_added.append((i_word, begin, end))
            i_start = i_end
        # Add from the end to avoid messing up the indices
        for (i_word, begin, end) in to_be_added[-1::-1]:
            words.insert(i_word, DISFLUENCY_MARK)
            word_tokens.insert(i_word, [])
            word_tokens_indices.insert(i_word, [])
            begin_times = np.insert(begin_times, i_word, begin)
            end_times = np.insert(end_times, i_word, end)

    # Ignore start / end tokens
    if not refine_whisper_precision_nframes:
        begin_times[1] = begin_times[0]
    if not refine_whisper_precision_nframes:
        end_times[-2] = end_times[-1]
    if unfinished_decoding:
        words = words[1:]
        word_tokens = word_tokens[1:]
        word_tokens_indices = word_tokens_indices[1:]
        begin_times = begin_times[1:]
        end_times = end_times[1:]
    else:
        words = words[1:-1]
        word_tokens = word_tokens[1:-1]
        word_tokens_indices = word_tokens_indices[1:-1]
        begin_times = begin_times[1:-1]
        end_times = end_times[1:-1]
    
    if plot:
        ymin = 1

        plt.subplot(nplots, 1, 1)
        for i, (w, ws, begin, end) in enumerate(zip(words, word_tokens, begin_times, end_times)):
            ymax = ymin + len(ws)
            if mfcc is None:
                plt.text(begin / AUDIO_TIME_PER_TOKEN, num_tokens-0.5, w, ha="left", va="top", color="red")
            for x in [begin, end,]:
                plt.axvline(x / AUDIO_TIME_PER_TOKEN, color="red", linestyle="dotted",
                            ymin=1-ymin/num_tokens,
                            ymax=0,  # 1-ymax/num_tokens,
                            )
            ymin = ymax

        if plot_mfcc:
            plt.subplot(nplots, 1, 2)
            for i, (w, begin, end) in enumerate(zip(words, begin_times, end_times)):
                plt.text(begin * 2 / AUDIO_TIME_PER_TOKEN, mfcc.shape[-2]*1.05, w, ha="left", va="bottom", color="red")
                for x in [begin, end,]:
                    plt.axvline(x * 2 / AUDIO_TIME_PER_TOKEN, color="red", linestyle="dotted")

        if isinstance(plot, str):
            plt.savefig(f"{plot}.alignment{num_alignment_for_plot:03d}.jpg", bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
    
    return [
        dict(
            text=word,
            start=round_timestamp(begin + start_time),
            end=round_timestamp(end + start_time),
            tokens=tokens,
            tokens_indices=tokens_indices,
        )
        for word, begin, end, tokens, tokens_indices in zip(words, begin_times, end_times, word_tokens, word_tokens_indices)
        if not word.startswith("<|")
    ]

def find_start_padding(mfcc):
    last_mfcc = mfcc[0, :, -1]
    if torch.min(last_mfcc) == torch.max(last_mfcc) == 0:
        candidate_index = mfcc.shape[-1] - 2
        while candidate_index > 0:
            candidate = mfcc[0, :, candidate_index]
            if not torch.equal(candidate, last_mfcc):
                return candidate_index + 1
            candidate_index -= 1
        return 0

def round_confidence(x):
    return round(x, 3)

def round_timestamp(x):
    return round(x, 2)

_punctuation = "".join(c for c in string.punctuation if  c not in ["-", "'"]) + "。，！？：”、…"

def _get_alignment_heads(model_name, num_layers, num_heads):
    dump = _ALIGNMENT_HEADS[model_name]
    array = np.frombuffer(gzip.decompress(base64.b85decode(dump)), dtype=bool).copy()
    mask = torch.from_numpy(array).reshape(num_layers, num_heads)
    alignment_heads = mask.to_sparse()
    return alignment_heads

def filtered_keys(result, keys = [
    'text',
    'segments',
    'words',
    'start',
    'end',
    'confidence'
]):
    if isinstance(result, dict):
        return {k: filtered_keys(v, keys) for k, v in result.items() if k in keys}
    if isinstance(result, list):
        return [filtered_keys(v, keys) for v in result]
    if isinstance(result, float):
        return round(result, 2)
    return result

def hf_t0_whisper_states(text):
    text = re.sub('.layers.', '.blocks.', text)
    text = re.sub('.self_attn.', '.attn.', text)
    text = re.sub('.q_proj.', '.query.', text)
    text = re.sub('.k_proj.', '.key.', text)
    text = re.sub('.v_proj.', '.value.', text)
    text = re.sub('.out_proj.', '.out.', text)
    text = re.sub('.fc1.', '.mlp.0.', text)
    text = re.sub('.fc2.', '.mlp.2.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.encoder_attn.', '.cross_attn.', text)
    text = re.sub('.cross_attn.ln.', '.cross_attn_ln.', text)
    text = re.sub('.embed_positions.weight', '.positional_embedding', text)
    text = re.sub('.embed_tokens.', '.token_embedding.', text)
    text = re.sub('model.', '', text)
    text = re.sub('attn.layer_norm.', 'attn_ln.', text)
    text = re.sub('.final_layer_norm.', '.mlp_ln.', text)
    text = re.sub('encoder.layer_norm.', 'encoder.ln_post.', text)
    text = re.sub('decoder.layer_norm.', 'decoder.ln.', text)

def load_model(name: str, device: Optional[Union[str, torch.device]] = None, download_root: str = None, in_memory: bool = False):
    extension = os.path.splitext(name)[-1] if os.path.isfile(name) else None
    if name in whisper.available_models() or extension == '.pt':
        return whisper.load_model(name, device=device, download_root=download_root, in_memory=in_memory)
    if extension in ['.ckpt', '.bin']:
        model_path = name
    else:
        try:
            import transformers
        except ImportError:
            raise ImportError(f'If you are trying to download a HuggingFace model with {name}, please install first the transformers library')
        from transformers.utils import cached_file
        try:
            model_path = cached_file(name, 'pytorch_model.bin', cache_dir=download_root, use_auth_token=None, revision=None)
        except Exception as e:
            try:
                if isinstance(e, OSError):
                    model_path = cached_file(name, 'whisper.ckpt', cache_dir=download_root, use_auth_token=None, revision=None)
                else:
                    raise e
            except:
                raise RuntimeError(f'Original error: {e}\nCould not find model {name} from HuggingFace nor local folders')