from mmhuman3d.core.visualization.visualize_cameras import (
    visualize_chessboard_kinects_rgb,
    visualize_dumped_camera_parameter,
)


def test_visualize_chessboard_kinects_rgb():
    chessboard_test_file_path = \
        'tests/data/camera/calibration_chessboard_05_28_18_05_19.json'
    # interactive and show is set to False for pytest,
    # use GUI with interactive=True and show=True
    visualize_chessboard_kinects_rgb(
        chessboard_path=chessboard_test_file_path,
        interactive=False,
        show=False)


def test_visualize_dumped_camera_parameter():
    dumped_dir = \
        'tests/data/camera/dumped'
    visualize_dumped_camera_parameter(
        dumped_dir, interactive=False, show=False)
