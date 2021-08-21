from google_drive_downloader import GoogleDriveDownloader
import os
import pytest
import shutil

TESTS_TMP_PATH = "./tests/tmp"
GOOGLE_DRIVE_MODELS_FILE_ID = "1w8xiZINqCzVlwfzXqMfFC3-NDNDr2kvu"
GOOGLE_DRIVE_MODELS_ZIP_OUTPUT_FILE = "Models-for-MIA.zip"
GOOGLE_DRIVE_MODELS_UNZIP_OUTPUT_DIR = "Models-for-MIA"


@pytest.fixture
def tests_tmp_path(autouse=True):
    if not os.path.exists(TESTS_TMP_PATH):
        os.makedirs(TESTS_TMP_PATH)

    yield TESTS_TMP_PATH

    shutil.rmtree(TESTS_TMP_PATH)


@pytest.fixture
def models_path(tests_tmp_path, autouse=True):
    GoogleDriveDownloader.download_file_from_google_drive(
        file_id=GOOGLE_DRIVE_MODELS_FILE_ID,
        dest_path=os.path.join(tests_tmp_path, GOOGLE_DRIVE_MODELS_ZIP_OUTPUT_FILE),
        unzip=True,
    )

    yield os.path.join(tests_tmp_path, GOOGLE_DRIVE_MODELS_UNZIP_OUTPUT_DIR)

    os.remove(os.path.join(tests_tmp_path, GOOGLE_DRIVE_MODELS_ZIP_OUTPUT_FILE))
    shutil.rmtree(os.path.join(tests_tmp_path, GOOGLE_DRIVE_MODELS_UNZIP_OUTPUT_DIR))
