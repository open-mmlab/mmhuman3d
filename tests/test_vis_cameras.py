import os

from mmhuman3d.core.visualization.visualize_cameras import (
    visualize_chessboard_kinects_rgb,
    visualize_smc_kinects_rgb,
)


# no screen in docker, run no test, only import
def run_locally():
    run_visualize_chessboard_kinects_rgb([])


def run_visualize_chessboard_kinects_rgb(argv):
    chessboard_test_file_path = \
        os.path.join(
            "tests", "data", "mocap",
            'calibration_chessboard_05_28_18_05_19.json'
        )
    # interactive is set to False for pytest, use GUI with interactive=True
    visualize_chessboard_kinects_rgb(
        chessboard_path=chessboard_test_file_path, interactive=True)


# smc file too large, not uploaded and tested in gitlab
def run_visualize_smc_rgb(argv):
    smc_test_file_path = \
        "/Users/gaoyang/Downloads/FTP/zoehuman/p666_a666.smc"
    visualize_smc_kinects_rgb(smc_path=smc_test_file_path, interactive=True)
