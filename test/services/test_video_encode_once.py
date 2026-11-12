"""`encode_once`: a ready clip is not re-encoded, and nothing burned means the
final file is muxed by copy. Off by default, where every path is unchanged.

Real ffmpeg and MoviePy on small generated media: what is under test is that
the picture's bits survive, which only a real encode and decode can show.
"""

import os
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from app.config import config
from app.models.schema import (
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import video as vd
from app.utils import utils

CANVAS = VideoAspect.portrait.to_resolution()


def _ffmpeg(*args):
    subprocess.run(
        [utils.get_ffmpeg_binary(), "-y", "-loglevel", "error", *args],
        check=True,
        capture_output=True,
    )


def _clip(path, *, seconds=2, size=CANVAS, rate=30, pix_fmt="yuv420p"):
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc2=size={size[0]}x{size[1]}:rate={rate}",
        "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-pix_fmt", pix_fmt, "-an", path,
    )
    return path


def _tone(path, *, seconds, frequency):
    _ffmpeg("-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=24000",
            "-t", str(seconds), "-c:a", "libmp3lame", path)
    return path


def _frames(path):
    """Per-frame MD5 of the decoded video: equal lists mean equal pictures."""
    out = subprocess.run(
        [utils.get_ffmpeg_binary(), "-loglevel", "error", "-i", path, "-map", "0:v",
         "-f", "framemd5", "-"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [line.rsplit(",", 1)[1].strip() for line in out.splitlines()
            if line and not line.startswith("#")]


def _pcm(path):
    return subprocess.run(
        [utils.get_ffmpeg_binary(), "-loglevel", "error", "-i", path, "-map", "0:a",
         "-f", "s16le", "-ac", "1", "-"],
        check=True, capture_output=True,
    ).stdout


def _duration(path, stream):
    """A stream's duration in seconds, from ffprobe beside the ffmpeg binary."""
    probe = shutil.which("ffprobe") or os.path.join(
        os.path.dirname(utils.get_ffmpeg_binary()), "ffprobe"
    )
    result = subprocess.run(
        [probe, "-v", "error", "-select_streams", stream, "-show_entries",
         "stream=duration", "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


class TestEncodeOnce(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        self.temp_dir = tempfile.mkdtemp()
        self.ready = _clip(os.path.join(self.temp_dir, "ready.mp4"))
        self.voice = _tone(os.path.join(self.temp_dir, "voice.mp3"), seconds=1.9, frequency=440)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _combine(self, path, **overrides):
        output = os.path.join(self.temp_dir, "combined.mp4")
        arguments = dict(
            combined_video_path=output,
            video_paths=[path],
            audio_file=self.voice,
            video_aspect=VideoAspect.portrait,
            video_concat_mode=VideoConcatMode.sequential,
            video_transition_mode=VideoTransitionMode.none,
            max_clip_duration=3,
            threads=2,
            video_fit_mode=VideoFitMode.cover,
        )
        arguments.update(overrides)
        vd.combine_videos(**arguments)
        return output

    # --- combine_videos -----------------------------------------------------

    def test_a_ready_clip_is_passed_through_untouched(self):
        config.app["encode_once"] = True
        with patch.object(
            vd, "_write_videofile_with_codec_fallback",
            side_effect=AssertionError("a ready clip must not be re-encoded"),
        ):
            combined = self._combine(self.ready)
        got, source = _frames(combined), _frames(self.ready)
        # Cut to the narration (plus nothing): every frame is the source's.
        self.assertGreater(len(got), 0)
        self.assertEqual(got, source[: len(got)])
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, "temp-clip-1.mp4")))

    def test_without_the_flag_the_clip_is_re_encoded_as_before(self):
        with patch.object(
            vd, "_write_videofile_with_codec_fallback",
            wraps=vd._write_videofile_with_codec_fallback,
        ) as writer:
            self._combine(self.ready)
        writer.assert_called_once()

    def test_a_clip_that_is_not_ready_is_re_encoded(self):
        config.app["encode_once"] = True
        cases = {
            "25 fps": _clip(os.path.join(self.temp_dir, "r25.mp4"), rate=25),
            "smaller": _clip(os.path.join(self.temp_dir, "small.mp4"), size=(720, 1280)),
            "yuv444p": _clip(os.path.join(self.temp_dir, "444.mp4"), pix_fmt="yuv444p"),
        }
        for name, path in cases.items():
            with self.subTest(name):
                with patch.object(
                    vd, "_write_videofile_with_codec_fallback",
                    wraps=vd._write_videofile_with_codec_fallback,
                ) as writer:
                    self._combine(path)
                writer.assert_called_once()

    def test_longer_than_the_clip_limit_or_with_a_transition_is_re_encoded(self):
        config.app["encode_once"] = True
        for overrides in ({"max_clip_duration": 1},
                          {"video_transition_mode": VideoTransitionMode.fade_in}):
            with self.subTest(overrides):
                with patch.object(
                    vd, "_write_videofile_with_codec_fallback",
                    wraps=vd._write_videofile_with_codec_fallback,
                ) as writer:
                    self._combine(self.ready, **overrides)
                self.assertTrue(writer.called)

    def test_the_concat_copies_only_when_asked(self):
        commands = []

        def run(command, **_kwargs):
            commands.append(command)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        listed = os.path.join(self.temp_dir, "clip.mp4")
        with patch.object(vd.subprocess, "run", side_effect=run):
            for copy in (False, True):
                vd.concat_video_clips_with_ffmpeg(
                    clip_files=[listed], output_file=os.path.join(self.temp_dir, "o.mp4"),
                    threads=2, output_dir=self.temp_dir, max_duration=1.5, stream_copy=copy,
                )
        before, after = commands
        self.assertIn("-c:v", before)
        self.assertNotIn("-c:v", after)
        self.assertEqual(after[after.index("-c") + 1], "copy")
        self.assertIn("-t", after)

    def test_many_written_clips_are_joined_by_copy_with_the_flag(self):
        config.app["encode_once"] = True
        second = _clip(os.path.join(self.temp_dir, "second.mp4"), seconds=1, size=(720, 1280))
        with patch.object(
            vd, "concat_video_clips_with_ffmpeg", wraps=vd.concat_video_clips_with_ffmpeg
        ) as concat:
            self._combine(self.ready, video_paths=[self.ready, second])
        self.assertTrue(concat.call_args.kwargs["stream_copy"])

    # --- generate_video ------------------------------------------------------

    def _generate(self, combined, *, flag, bgm=None, override=None, subtitle_path=""):
        config.app["encode_once"] = flag
        output = os.path.join(self.temp_dir, f"final-{int(flag)}.mp4")
        params = VideoParams(
            video_subject="test", subtitle_enabled=False,
            bgm_type="custom" if (bgm or override) else "", bgm_volume=0.2,
        )
        with patch.object(vd, "get_bgm_file", return_value=bgm or ""):
            result = vd.generate_video(
                video_path=combined, audio_path=self.voice, subtitle_path=subtitle_path,
                output_file=output, params=params, bgm_file_override=override,
            )
        return output, result

    def _no_reencode(self):
        return patch.object(
            vd, "_write_videofile_with_codec_fallback",
            side_effect=AssertionError("nothing burned: the picture must not be re-encoded"),
        )

    def _leftovers(self):
        return [name for name in os.listdir(self.temp_dir) if "encode-once-audio" in name]

    def test_nothing_burned_means_the_picture_is_copied_and_the_audio_is_moviepys(self):
        """A looped track (a custom BGM, as SocialReel's), faded out: the
        picture is the input's frame for frame, and the audio is exactly what
        MoviePy itself writes, sample for sample, at the same length."""
        combined = _clip(os.path.join(self.temp_dir, "combined-in.mp4"), seconds=1.9)
        bgm = _tone(os.path.join(self.temp_dir, "bgm.mp3"), seconds=0.7, frequency=220)
        old, old_result = self._generate(combined, flag=False, bgm=bgm)
        with self._no_reencode():
            new, new_result = self._generate(combined, flag=True, bgm=bgm)
        self.assertTrue(old_result and new_result)
        self.assertEqual(_frames(new), _frames(combined))
        self.assertEqual(_pcm(new), _pcm(old))
        for stream in ("v", "a"):
            self.assertAlmostEqual(_duration(new, stream), _duration(old, stream), delta=1 / 30)
        self.assertEqual(self._leftovers(), [])

    def test_audio_longer_than_the_picture_is_cut_at_the_last_frame(self):
        """A provider's track passed as an override is not looped and may
        outlast the video; MoviePy then leaves audio playing past the last
        frame. The mux ends with the picture: same samples up to there."""
        combined = _clip(os.path.join(self.temp_dir, "combined-in.mp4"), seconds=1.9)
        long_track = _tone(os.path.join(self.temp_dir, "provider.mp3"), seconds=4, frequency=220)
        old, _ = self._generate(combined, flag=False, override=long_track)
        with self._no_reencode():
            new, _ = self._generate(combined, flag=True, override=long_track)
        new_pcm = _pcm(new)
        self.assertEqual(new_pcm, _pcm(old)[: len(new_pcm)])
        self.assertAlmostEqual(_duration(new, "a"), _duration(new, "v"), delta=1 / 30)
        self.assertEqual(_frames(new), _frames(combined))

    def test_a_failed_mux_falls_back_to_moviepy(self):
        combined = _clip(os.path.join(self.temp_dir, "combined-in.mp4"), seconds=1.9)
        real_run = subprocess.run

        def failing_mux(command, *args, **kwargs):
            if "-map" in command and "1:a:0" in command:
                return types.SimpleNamespace(returncode=1, stdout="", stderr="mux refused")
            return real_run(command, *args, **kwargs)

        with patch.object(vd.subprocess, "run", side_effect=failing_mux):
            with patch.object(
                vd, "_write_videofile_with_codec_fallback",
                wraps=vd._write_videofile_with_codec_fallback,
            ) as writer:
                output, _ = self._generate(combined, flag=True)
        writer.assert_called_once()
        self.assertTrue(os.path.getsize(output) > 0)
        leftovers = [name for name in os.listdir(self.temp_dir) if "encode-once-audio" in name]
        self.assertEqual(leftovers, [])

    def test_a_subtitle_file_keeps_the_moviepy_path(self):
        combined = _clip(os.path.join(self.temp_dir, "combined-in.mp4"), seconds=1.9)
        subtitle = os.path.join(self.temp_dir, "subtitle.srt")
        with open(subtitle, "w", encoding="utf-8") as handle:
            handle.write("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
        with patch.object(vd, "_mux_audio_by_copy",
                          side_effect=AssertionError("subtitles are burned: no copy")):
            with patch.object(vd, "_write_videofile_with_codec_fallback") as writer:
                with patch.object(vd, "SubtitlesClip"), patch.object(vd, "CompositeVideoClip"):
                    self._generate(combined, flag=True, subtitle_path=subtitle)
        writer.assert_called_once()

    def test_without_the_flag_generate_video_writes_with_moviepy(self):
        combined = _clip(os.path.join(self.temp_dir, "combined-in.mp4"), seconds=1.9)
        with patch.object(vd, "_mux_audio_by_copy",
                          side_effect=AssertionError("the flag is off")):
            with patch.object(
                vd, "_write_videofile_with_codec_fallback",
                wraps=vd._write_videofile_with_codec_fallback,
            ) as writer:
                self._generate(combined, flag=False)
        writer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
