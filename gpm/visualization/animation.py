# -----------------------------------------------------------------------------.
# MIT License

# Copyright (c) 2024 GPM-API developers
#
# This file is part of GPM-API.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------.
"""This module contains tools for creating animations."""
import contextlib
import os
import shutil
import subprocess
import tempfile

import numpy as np

# -----------------------------------------------------------------------------.


def check_input_files(image_filepaths):
    """Check valid input file paths."""
    # Check at least 2 files
    if not isinstance(image_filepaths, list):
        raise ValueError("Expecting a list.")
    if len(image_filepaths) < 2:
        raise ValueError("Expecting a list 2 file paths")

    # Check file exist
    if np.any([not os.path.exists(fpath) for fpath in image_filepaths]):
        raise ValueError("Not all file paths exists on disk.")

    # Check file format
    # TODO

    return image_filepaths


def check_frame_settings(frame_duration, frame_rate, return_duration=False):
    if frame_duration is not None and frame_rate is not None:
        raise ValueError("Either specify frame_duration or frame_rate.")

    if frame_duration is None and frame_rate is None:
        frame_rate = 4

    if frame_duration is not None:
        frame_rate = int(1000 / frame_duration)
    else:
        frame_duration = int(1000 / frame_rate)
    if return_duration:
        return frame_duration
    return frame_rate


def _move_and_rename_image_to_tmp_dir(filepaths, tmp_dir=None, delete_inputs=False):
    # Create a temporary directory
    tmp_dir = tempfile.mkdtemp(prefix="tmp_images_", dir=tmp_dir)
    pattern = "image_{:04d}.png"
    # Copy files to the temporary directory and rename them
    # - Do not move because a single image can be referenced multiple times in filepaths
    for i, filepath in enumerate(filepaths):
        new_filename = pattern.format(i + 1)  # Start index from 1
        new_filepath = os.path.join(tmp_dir, new_filename)
        shutil.copy(filepath, new_filepath)

    # Delete inputs
    if delete_inputs:
        for filepath in filepaths:
            with contextlib.suppress(FileNotFoundError):
                os.remove(filepath)

    pattern = "image_%04d.png"
    return tmp_dir, pattern


def get_image_size(image_path):
    from PIL import Image

    img = Image.open(image_path)
    width, height = img.size
    return width, height


def create_gifski_gif(
    image_filepaths,
    gif_fpath,
    sort=False,
    frame_duration=None,
    frame_rate=None,
    loop=0,
    quality=100,
    motion_quality=100,
    lossy_quality=100,
    optimize=True,
    delete_inputs=False,
    verbose=True,
):
    """
    Create a GIF from a list of image filepaths using gifski.

    Gifski generates per-frame optimized palettes, combines palettes across frames
    and achieve thousands of colors per frame for maximum quality.

    Either specify ``frame_rate`` or ``frame_duration``.

    Parameters
    ----------
    image_filepaths : list of str
        List of filepaths of input images. The images will be used to create the GIF.
    gif_fpath : str
        Filepath of the output GIF.
    sort : bool, optional
        If ``True``, sort the input images in ascending order. The default is ``False``.
    frame_duration : int, optional
        Duration in milliseconds (ms) of each GIF frame.
        The default is 250 ms.
    frame_rate: int, optional
        The number of individual frames displayed in one second.
        The default is 4.
    quality: str, optional
        Controls the overall output quality. The default is 100.
        Lower values reduces quality but may reduce file size.
    motion_quality: str, optional
        Controls motion quality. The default is 100.
        Lower values reduce motion quality.
    lossy_quality: str, optional
        Controls PNG quantization quality. The default is 100.
        Lower values introduce noise and streaks.
    optimize: bool, optional
        If ``True``, it improves the GIF quality.
        If ``False``, it speeds up the GIF creation at quality expense.
    loop : int, optional
        Number of times the GIF should loop. Set to 0 for infinite loop. The default is 0.
    delete_inputs : bool, optional
        If ``True``, delete the original input images after creating the GIF. The default is ``False``.

    Notes
    -----
    This function uses gifski to create the GIF. Ensure that gifski is installed and
    accessible in the system's ``PATH``.

    On Linux systems, it can be installed using ``sudo snap install gifski``.

    More GIFski information at: https://github.com/ImageOptim/gifski

    More PNG quantization information at: https://pngquant.org/

    Examples
    --------
    >>> filepaths = ["image1.png", "image2.png", "image3.png"]
    >>> gif_fpath = "output.gif"
    >>> create_gifski_gif(filepaths, gif_fpath, sort=True, frame_duration=200)
    """
    # Define frame rate
    frame_rate = check_frame_settings(frame_duration, frame_rate)

    # Sort image filepaths if required
    if sort:
        image_filepaths.sort()

    # Retrieve image width and height
    width, height = get_image_size(image_filepaths[0])

    # Move images to a temporary directory
    # - gifski is not able to process image that are not in /home/*
    # --> I can not use /tmp or /ltenas
    base_dir = os.path.join(os.path.expanduser("~"), "tmp_gifski")  # /home/<user>/tmp_gifski
    os.makedirs(base_dir, exist_ok=True)
    tmp_dir, _ = _move_and_rename_image_to_tmp_dir(image_filepaths, tmp_dir=base_dir, delete_inputs=delete_inputs)
    input_pattern = os.path.join(tmp_dir, "image_*.png")
    tmp_fpath = os.path.join(base_dir, os.path.basename(gif_fpath))

    # Define basic command
    gifski_cmd = [
        "gifski",
        # Do not sort files
        "--no-sort",
        # Framerate
        "--fps",
        str(frame_rate),  # input framerate. must be greater or equal to fps
        # Set output quality
        "--quality",
        str(quality),
        "--motion-quality",
        str(motion_quality),
        "--lossy-quality",
        str(lossy_quality),
        # Loops
        "--repeat",
        str(loop),
        # GIF size
        "--width",
        str(width),
        "--height",
        str(height),
    ]

    # Add optimization option if specified
    if optimize:
        gifski_cmd.extend(["--extra"])

    # Add output filepath
    # - Overwrite existing !
    gifski_cmd.extend(["--output", tmp_fpath])

    # Add input file paths pattern
    gifski_cmd.extend([input_pattern])

    # Define the command
    gifski_cmd = " ".join(gifski_cmd)

    print(gifski_cmd)
    # Run gifski using subprocess
    try:
        subprocess.run(gifski_cmd, check=True, shell=True, capture_output=not verbose)
        shutil.move(tmp_fpath, gif_fpath)
    except subprocess.CalledProcessError as e:
        print(f"Error creating GIF: {e}")

    # Remove temporary directory and its contents
    shutil.rmtree(tmp_dir)


def create_pillow_gif(image_filepaths, gif_fpath, sort=False, frame_duration=None, frame_rate=None, loop=0):
    """
    Create a GIF from a list of image filepaths.

    Either specify ``frame_rate`` or ``frame_duration``.

    Parameters
    ----------
    image_filepaths : list
        List of image file paths.
    gif_fpath: str
        File path where to save the gif.
    sort : bool, optional
        If ``True``, sort the input images in ascending order. The default is ``False``.
    frame_duration : int, optional
        Duration in milliseconds (ms) of each GIF frame.
        The default is 250 ms.
    frame_rate: int, optional
        The number of individual frames displayed in one second.
        The default is 4.
    loop : int, optional
        Number of times the APNG should loop. Set to 0 for infinite loop. The default is 0.

    """
    from PIL import GifImagePlugin, Image

    GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY

    # Checks
    image_filepaths = check_input_files(image_filepaths)
    frame_duration = check_frame_settings(frame_duration, frame_rate, return_duration=True)

    # Sort file names to ensure proper sequence
    if sort:
        image_filepaths.sort()

    # Create a list of image objects
    images = [Image.open(fpath) for fpath in image_filepaths]

    # Save GIF
    _ = images[0].save(
        gif_fpath,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        loop=loop,
        duration=frame_duration,
        optimize=False,
        lossless=True,
    )


def _get_ffmpeg_scale_value(width, height):
    # Keep image size
    if width is None and height is None:
        scale_value = "iw:ih"
        return scale_value
    # Set the unspecified height/width to -1 to keep the aspect ratio
    if height is None:
        height = -1
    if width is None:
        width = -1
    # Define the scale value
    if height is not None and width is not None:
        scale_value = f"w={width}:h={height}"
    return scale_value


def _get_ffmpeg_palettegen_value(stats_mode=None):
    valid_stats_modes = ["single", "full", "diff"]
    value = "palettegen"
    if stats_mode in valid_stats_modes:
        value = f"palettegen=stats_mode={stats_mode}:reserve_transparent=1"
    return value


def check_valid_dither(dither):
    valid_dither = ["none", "bayer", "floyd_steinberg", "heckbert", "sierra2", "sierra2_4a", "sierra_lite"]
    if dither is None:
        dither = "none"
    if dither not in valid_dither:
        raise ValueError(f"Valid dither values are {valid_dither}")
    return dither


def _get_ffmpeg_paletteuse_value(stats_mode, dither=None, bayer_scale=1):
    dither = check_valid_dither(dither)  # stats_mode="single":  # per frame palette
    value = f"paletteuse=new=1:dither={dither}" if stats_mode == "single" else f"paletteuse=dither={dither}"
    if dither == "bayer":
        value = f"{value}:bayer_scale={bayer_scale}"
    return value


def _get_ffmpeg_filter_complex_value(
    frame_rate,
    width=None,
    height=None,
    interpolation="lanczos",
    stats_mode="single",
    dither="none",
    bayer_scale=1,
):

    scale_value = _get_ffmpeg_scale_value(width=width, height=height)
    palettegen_value = _get_ffmpeg_palettegen_value(stats_mode=stats_mode)
    paletteuse_value = _get_ffmpeg_paletteuse_value(stats_mode=stats_mode, dither=dither, bayer_scale=bayer_scale)
    filter_value = "".join(
        [
            "[0:v] " f"fps={frame_rate},",
            f"scale={scale_value}:flags={interpolation}," "split [a][b];",
            f"[a] {palettegen_value} [p];" f"[b][p] {paletteuse_value}",
        ],
    )
    return filter_value


def create_ffmpeg_gif(
    image_filepaths,
    gif_fpath,
    sort=False,
    frame_duration=None,
    frame_rate=None,
    loop=0,
    width=None,
    height=None,
    interpolation="lanczos",
    stats_mode="single",
    dither=None,
    bayer_scale=1,
    delete_inputs=False,
    verbose=True,
):
    """
    Create a GIF from a list of image filepaths using ffmpeg.

    Either specify ``frame_rate`` or ``frame_duration``.

    Parameters
    ----------
    image_filepaths : list of str
        List of filepaths of input images. The images will be used to create the GIF.
    gif_fpath : str
        Filepath of the output GIF.
    sort : bool, optional
        If ``True``, sort the input images in ascending order. Default is False.
    frame_duration : int, optional
        Duration in milliseconds (ms) of each GIF frame.
        The default is 250 ms.
    frame_rate: int, optional
        The number of individual frames displayed in one second.
        The default is 4.
    width: int, optional
        The width of the output GIF.
        If ``None`` and height is provided, is set to ``-1`` to preserve aspect ratio.
        If ``None`` and height is also ``None``, the output has same size as the input images.
    height: int, optional
        The height of the output GIF.
        If ``None`` and width is provided, is set to ``-1`` to preserve aspect ratio.
        If ``None`` and width is also ``None``, the output has same size as the input images.
    interpolation: str, optional
        Interpolation method to use if need to rescale the input image.
        Valid values are ``'bilinear'``, ``'lanczos'``, ``'bicubic'``. The default is ``'lanczos'``.
    loop : int, optional
        Number of times the GIF should loop. Set to 0 for infinite loop. The default is 0.
    per_frame_palette : bool, optional
        If ``True``, enable definition of a palette for each GIF frame. The default is ``True``.
    stats_mode: str, optional
        The method to establish the color palette.
        Valid values are ``'single'``,``'full'``, ``'diff'``.
        If ``'full'``, compute a single palette over all frames.
        If ``'single'``, compute a palette for each frame.
        If ``'diff'``, compute histograms only for pixels that differs from previous frame.
        The ``'diff'`` mode might be relevant to give more importance to the moving part
        of your input if the background is static.
        The default is ``'single'``.
    dither: str or None, optional
        Dithering method to apply during color quantization. Available options are:
        ``None`` or ``'none'`` (no dithering), ``'bayer'``, ``'floyd_steinberg'``,
        ``'heckbert'``, ``'sierra2'``, ``'sierra2_4a'``, ``'sierra_lite'``.
        The default is ``None``.
    delete_inputs : bool, optional
        If ``True``, delete the original input images after creating the GIF. The default is ``False``.

    Notes
    -----
    This function uses ffmpeg to create the GIF. Ensure that ffmpeg is installed and
    accessible in the system's ``PATH``.

    Examples
    --------
    >>> filepaths = ["image1.png", "image2.png", "image3.png"]
    >>> gif_fpath = "output.gif"
    >>> create_ffmpeg_gif(filepaths, gif_fpath, sort=True, frame_duration=200)
    """
    # TODO: dither is wrong !
    frame_rate = check_frame_settings(frame_duration, frame_rate)

    # Sort image filepaths if required
    if sort:
        image_filepaths.sort()

    # Move images to temporary directory
    tmp_dir, pattern = _move_and_rename_image_to_tmp_dir(image_filepaths, delete_inputs=delete_inputs)
    input_pattern = os.path.join(tmp_dir, pattern)

    filter_value = _get_ffmpeg_filter_complex_value(
        frame_rate=frame_rate,
        width=width,
        height=height,
        interpolation=interpolation,
        stats_mode=stats_mode,
        dither=dither,
        bayer_scale=bayer_scale,
    )
    # Construct the ffmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        # Framerate
        "-framerate",
        str(frame_rate),  # input framerate. must be greater or equal to fps
        # Inputs
        "-i",
        input_pattern,
        # Filters
        "-filter_complex",
        f"'{filter_value}'",
        # Codec
        "-c:v",
        "gif",
        # Loops
        "-loop",
        str(loop),
    ]

    # Add output file path to ffmpeg command
    # - Overwrite existing !
    ffmpeg_cmd.extend(["-y", gif_fpath])

    # Define the command
    ffmpeg_cmd = " ".join(ffmpeg_cmd)

    print(ffmpeg_cmd)
    # Run ffmpeg using subprocess
    try:
        subprocess.run(ffmpeg_cmd, check=True, shell=True, capture_output=not verbose)
    except subprocess.CalledProcessError as e:
        print(f"Error creating GIF: {e}")

    # Remove temporary directory and its contents
    shutil.rmtree(tmp_dir)


def check_mp4_codec(codec):
    valid_codecs = ["libx264", "libx265", "mpeg4", "vp9", "av1"]
    if codec not in valid_codecs:
        raise ValueError(f"Invalid codec {codec}. Valid codecs are {valid_codecs}.")
    return codec


def check_mp4_optimization(optimization):
    valid_optimization = [
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    ]
    if optimization not in valid_optimization:
        raise ValueError(
            f"""Invalid optimization method '{optimization}'.
                         Valid optimization methods are {valid_optimization}.""",
        )
    return optimization


def create_ffmpeg_mp4(
    image_filepaths,
    mp4_fpath,
    sort=False,
    frame_duration=None,
    frame_rate=None,
    codec="libx264",
    optimization="placebo",
    delete_inputs=False,
    verbose=True,
):
    """
    Create a MP4 from a list of image filepaths using ffmpeg.

    Either specify ``frame_rate`` or ``frame_duration``.
    YouTube and Vimeo don't really appreciate video with < 0.5 FPS
    Only MP4 videos with H.264 codec are supported across all Powerpoint versions.

    Parameters
    ----------
        image_filepaths : list of str
            List of filepaths of input images. The images will be used to create the MP4.
        mp4_fpath : str
            Filepath of the output MP4.
        sort : bool, optional
            If ``True``, sort the input images in ascending order. The default is ``False``.
        frame_duration : int, optional
            Duration in milliseconds (ms) of each MP4 frame.
            The default is 250 ms.
        frame_rate: int, optional
            The number of individual frames displayed in one second.
            The default is 4.
        codec: str, optional
            MP4 codec.
            Valid codecs are ``"libx264"``, ``"libx265"``, ``"mpeg4"``, ``"vp9"``, ``"av1"``.
            The default is ``"libx264"``.
            The ``'libx264'`` (H.264 AVC codec) is widely used for compatibility and good compression.
            The ``'libx265'`` (H.265 HEVC codec) offers better compression than H.264 but
            requires more processing power.
            The ``'vp9'`` codec, is developed by Google, and is known for its high compression efficiency.
            The ``'av1'`` codec is a newer open-source codec that aims to provide even better compression
            efficiency than H.265 and VP9. It's gaining popularity but may have varying
            levels of support on different devices and software. Video generation is slow
            and rendering on VLC and many others video players is not well supported.
            The ``'mpeg4'`` (MPEG-4 Part 2) codec is older and less efficient than H.264
            and H.265 but is still compatible with many devices and software
        optimization: str, optional
            Whether to optimize video encoding by using ``'-preset <optimization>'``.
            The default optimizazion used by FFMPEG is ``'medium'``.
            If ``'placebo'`` (the default of this function), the optimization slows down
            the file creation and increases the file size, but provides the best possible video quality.
        delete_inputs : bool, optional
            If ``True``, it delete the original input images after creating the MP4. The default is ``False``.

    Notes
    -----
        This function uses ffmpeg to create the MP4. Ensure that FFMPEG is installed and
        accessible in the system's ``PATH``.

    Examples
    --------
        >>> filepaths = ["image1.png", "image2.png", "image3.png"]
        >>> mp4_fpath = "output.mp4"
        >>> create_ffmpeg_gif(filepaths, mp4_fpath, sort=True, frame_duration=200)
    """
    frame_rate = check_frame_settings(frame_duration, frame_rate)
    codec = check_mp4_codec(codec)
    optimization = check_mp4_optimization(optimization)
    if codec not in ["libx264", "libx265"]:
        optimization = False

    # Sort image filepaths if required
    if sort:
        image_filepaths.sort()

    # Move images to temporary directory
    tmp_dir, pattern = _move_and_rename_image_to_tmp_dir(image_filepaths, delete_inputs=delete_inputs)
    input_pattern = os.path.join(tmp_dir, pattern)

    # Construct the ffmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        # Framerate
        "-r:v",
        str(frame_rate),
        # Inputs
        "-i",
        input_pattern,
        # Codec
        "-c:v",
        codec,
        # Disable audio processing
        "-an",
    ]

    # Add optimization option if specified
    if optimization:
        ffmpeg_cmd.extend(["-preset", optimization])

    # Add output file path to ffmpeg command
    # - Overwrite existing (with -y) !
    ffmpeg_cmd.extend(["-y", mp4_fpath])

    # Define the command
    ffmpeg_cmd = " ".join(ffmpeg_cmd)

    # Run ffmpeg using subprocess
    try:
        subprocess.run(ffmpeg_cmd, check=True, shell=True, capture_output=not verbose)
    except subprocess.CalledProcessError as e:
        print(f"Error creating MP4: {e}")

    # Remove temporary directory and its contents
    shutil.rmtree(tmp_dir)


def create_apng(image_filepaths, apng_fpath, sort=False, frame_duration=None, frame_rate=None, loop=0):
    """Create an Animated PNG.

    Either specify ``frame_rate`` or ``frame_duration``.

    Parameters
    ----------
    image_filepaths : list of str
        List of filepaths of input images. The images will be used to create the GIF.
    apng_fpath : str
        Filepath of the output APNG.
    frame_duration : int, optional
        Duration in milliseconds (ms) of each GIF frame.
        The default is 250 ms.
    frame_rate: int, optional
        The number of individual frames displayed in one second.
        The default is 4.
    loop : int, optional
        Number of times the APNG should loop. Set to 0 for infinite loop. The default is 0.

    Notes
    -----
    PowerPoint currently does not support APNGs.

    Examples
    --------
    >>> filepaths = ["image1.png", "image2.png", "image3.png"]
    >>> apng_fpath = "output.png"
    >>> create_apng(filepaths, apng_fpath, sort=False, frame_duration=200, loop=0)

    """
    # Checks
    try:
        from apng import APNG
    except ModuleNotFoundError:
        print("The apng package is not installed. Please install it using 'pip install apng'")

    frame_duration = check_frame_settings(frame_duration, frame_rate, return_duration=True)

    # Sort image filepaths if required
    if sort:
        image_filepaths.sort()

    # Create APNG
    apng_class = APNG(num_plays=loop)
    apng_class.from_files(image_filepaths, delay=frame_duration).save(apng_fpath)
