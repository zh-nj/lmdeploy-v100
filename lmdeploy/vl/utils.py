# Copyright (c) OpenMMLab. All rights reserved.
import os
from io import BytesIO
from typing import Union

import pybase64
import requests
from PIL import Image, ImageFile

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def encode_image_base64(image: Union[str, Image.Image]) -> str:
    """Encode raw date to base64 format."""
    buffered = BytesIO()
    FETCH_TIMEOUT = int(os.environ.get('LMDEPLOY_FETCH_TIMEOUT', 10))
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        if isinstance(image, str):
            url_or_path = image
            if url_or_path.startswith('http'):
                response = requests.get(url_or_path, headers=headers, timeout=FETCH_TIMEOUT)
                response.raise_for_status()
                buffered.write(response.content)
            elif os.path.exists(url_or_path):
                with open(url_or_path, 'rb') as image_file:
                    buffered.write(image_file.read())
        elif isinstance(image, Image.Image):
            image.save(buffered, format='PNG')
    except Exception as error:
        if isinstance(image, str) and len(image) > 100:
            image = image[:100] + ' ...'
        logger.error(f'{error}, image={image}')
        # use dummy image
        image = Image.new('RGB', (32, 32))
        image.save(buffered, format='PNG')
    res = pybase64.b64encode(buffered.getvalue()).decode('utf-8')
    return res


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """Load image from base64 format."""
    return Image.open(BytesIO(pybase64.b64decode(image)))


def load_image(image_url: Union[str, Image.Image]) -> Image.Image:
    """Load image from url, local path or openai GPT4V."""
    FETCH_TIMEOUT = int(os.environ.get('LMDEPLOY_FETCH_TIMEOUT', 10))
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if isinstance(image_url, Image.Image):
            img = image_url
        elif image_url.startswith('http'):
            response = requests.get(image_url, headers=headers, timeout=FETCH_TIMEOUT)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        elif image_url.startswith('data:image'):
            img = load_image_from_base64(image_url.split(',')[1])
        else:
            # Load image from local path
            img = Image.open(image_url)

        # check image valid
        img = img.convert('RGB')
    except Exception as error:
        if isinstance(image_url, str) and len(image_url) > 100:
            image_url = image_url[:100] + ' ...'
        logger.error(f'{error}, image_url={image_url}')
        # use dummy image
        img = Image.new('RGB', (32, 32))

    return img


def load_video(video_url: str,
               max_num_frames: int = None,
               fps: float = 2.0) -> list:
    """Load video from url, local path or base64, return frame list.

    Args:
        video_url: HTTP/HTTPS URL, local file path, or base64 data:video.
        max_num_frames: Max frames to extract. Defaults to env
            LMDEPLOY_MAX_NUM_FRAMES or 32.
        fps: Target frame rate for extraction. Default 2.0.

    Returns:
        List of PIL.Image.Image frames in RGB, or empty list on error.
    """
    import tempfile

    import numpy as np

    if max_num_frames is None:
        max_num_frames = int(os.environ.get('LMDEPLOY_MAX_NUM_FRAMES', 32))

    FETCH_TIMEOUT = int(os.environ.get('LMDEPLOY_FETCH_TIMEOUT', 10))
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        # Resolve video to a local file path
        tmp_file = None
        if video_url.startswith(('http://', 'https://')):
            response = requests.get(
                video_url, headers=headers, timeout=FETCH_TIMEOUT)
            response.raise_for_status()
            tmp_file = tempfile.NamedTemporaryFile(
                suffix='.mp4', delete=False)
            tmp_file.write(response.content)
            tmp_file.flush()
            video_path = tmp_file.name
        elif video_url.startswith('data:video'):
            data = video_url.split(',', 1)[1]
            raw = pybase64.b64decode(data)
            tmp_file = tempfile.NamedTemporaryFile(
                suffix='.mp4', delete=False)
            tmp_file.write(raw)
            tmp_file.flush()
            video_path = tmp_file.name
        else:
            video_path = video_url

        # Decode video frames
        frames = _decode_video_frames(video_path, max_num_frames, fps, np)

        # Cleanup temp file
        if tmp_file is not None:
            tmp_file.close()
            os.unlink(tmp_file.name)

        return frames

    except Exception as error:
        if isinstance(video_url, str) and len(video_url) > 100:
            video_url = video_url[:100] + ' ...'
        logger.error(f'{error}, video_url={video_url}')
        return []


def _decode_video_frames(video_path, max_num_frames, fps, np):
    """Decode video and return uniformly sampled PIL frames."""
    # Try decord first
    try:
        import decord
        decord.bridge.set_bridge('native')
        vr = decord.VideoReader(video_path)
        total = len(vr)
        n_frames = min(total, max_num_frames)
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        frames_np = vr.get_batch(indices.tolist()).asnumpy()
        return [Image.fromarray(f).convert('RGB') for f in frames_np]
    except ImportError:
        pass

    # Try cv2
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError(f'Cannot read video: {video_path}')
        n_frames = min(total, max_num_frames)
        indices = set(np.linspace(0, total - 1, n_frames, dtype=int).tolist())
        frames = []
        for idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames
    except ImportError:
        pass

    # Try av (PyAV)
    try:
        import av as _av
        container = _av.open(video_path)
        stream = container.streams.video[0]
        total = stream.frames or 0
        # If frame count unknown, decode all first
        if total <= 0:
            all_frames = [
                f.to_image().convert('RGB')
                for f in container.decode(video=0)
            ]
            total = len(all_frames)
            n_frames = min(total, max_num_frames)
            indices = np.linspace(
                0, total - 1, n_frames, dtype=int).tolist()
            return [all_frames[i] for i in indices]
        n_frames = min(total, max_num_frames)
        indices = set(
            np.linspace(0, total - 1, n_frames, dtype=int).tolist())
        frames = []
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in indices:
                frames.append(frame.to_image().convert('RGB'))
        container.close()
        return frames
    except ImportError:
        pass

    raise RuntimeError(
        'No video decoder available. Install one of: decord, opencv-python, av'
    )

