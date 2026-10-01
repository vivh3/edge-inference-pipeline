"""Webcam capture. What the driver agrees to is not what was requested."""

import sys
import types

import pytest

from inference.capture import WebcamSource, _frame_size

WIDTH, HEIGHT, FPS, FOURCC, BUFFERSIZE = 3, 4, 5, 6, 38


def code(chars):
    return sum(ord(c) << (8 * i) for i, c in enumerate(chars))


class FakeCapture:
    """Records what was set, answers with what the driver settled on."""

    def __init__(self, settled, opened=True):
        self.settled = settled
        self.opened = opened
        self.calls = []

    def set(self, prop, value):
        self.calls.append((prop, value))
        return True

    def get(self, prop):
        return self.settled.get(prop, 0)

    def isOpened(self):
        return self.opened

    @property
    def props_set(self):
        return [prop for prop, _ in self.calls]


@pytest.fixture
def cv2_returning(monkeypatch):
    def install(capture):
        mod = types.ModuleType("cv2")
        mod.CAP_PROP_FRAME_WIDTH = WIDTH
        mod.CAP_PROP_FRAME_HEIGHT = HEIGHT
        mod.CAP_PROP_FPS = FPS
        mod.CAP_PROP_FOURCC = FOURCC
        mod.CAP_PROP_BUFFERSIZE = BUFFERSIZE
        mod.VideoWriter_fourcc = lambda *chars: code(chars)
        mod.VideoCapture = lambda device: capture
        monkeypatch.setitem(sys.modules, "cv2", mod)
        return capture

    return install


def mjpg_640x480():
    return FakeCapture({WIDTH: 640, HEIGHT: 480, FPS: 30.0, FOURCC: code("MJPG")})


def test_mjpg_is_requested_rather_than_left_to_the_driver(cv2_returning):
    # OpenCV's V4L2 backend defaults to uncompressed YUYV, which USB 2.0
    # bandwidth caps at a few fps -- and some cameras offer no YUYV mode at
    # the requested size at all.
    cap = cv2_returning(mjpg_640x480())
    WebcamSource().open()
    assert (FOURCC, code("MJPG")) in cap.calls


def test_the_pixel_format_is_set_before_the_resolution(cv2_returning):
    # The backend resolves size and rate within the chosen format, so a
    # format set afterwards can renegotiate the size just requested.
    cap = cv2_returning(mjpg_640x480())
    WebcamSource().open()
    assert cap.props_set.index(FOURCC) < cap.props_set.index(WIDTH)
    assert cap.props_set.index(FOURCC) < cap.props_set.index(HEIGHT)


def test_an_empty_fourcc_leaves_the_format_alone(cv2_returning):
    cap = cv2_returning(mjpg_640x480())
    WebcamSource(fourcc="").open()
    assert FOURCC not in cap.props_set


def test_the_drivers_answer_is_read_back_not_assumed(cv2_returning):
    # Asked for 640x480 MJPG; this driver settles on 1024x576 YUYV at 20 fps.
    cv2_returning(FakeCapture({WIDTH: 1024, HEIGHT: 576, FPS: 20.0, FOURCC: code("YUYV")}))
    source = WebcamSource(width=640, height=480)
    source.open()
    assert source.negotiated == {
        "width": 1024,
        "height": 576,
        "fps": 20.0,
        "fourcc": "YUYV",
    }


def test_a_device_that_will_not_open_raises_rather_than_yielding_nothing(cv2_returning):
    cv2_returning(FakeCapture({}, opened=False))
    with pytest.raises(RuntimeError):
        WebcamSource(device=3).open()


def test_frame_size_comes_from_the_array_not_from_the_request():
    class Array:
        shape = (576, 1024, 3)  # numpy order: rows, columns, channels

    assert _frame_size(Array(), (640, 480)) == (1024, 576)


def test_frame_size_falls_back_when_the_payload_has_no_shape():
    assert _frame_size(object(), (640, 480)) == (640, 480)
