# MIT License
#
# Copyright (c) 2017 Andrea Palazzi
#
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
import requests
import zipfile
import os


class GoogleDriveDownloader:
    """Minimal class to download shared files from Google Drive."""

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = "https://docs.google.com/uc?export=download"

    @staticmethod
    def download_file_from_google_drive(
        file_id: str, output_path: str, unzip: bool = False
    ):
        """Downloads a shared file from google drive into a given folder. Optionally unzips it.

        :param file_id: The file identifier. It can be obtained from the sharable link.
        :param output_path: The output path where to save the downloaded file.
        :param unzip: If True unzips a file. If the file is not a zip file, ignores it.
        """
        output_directory = os.path.dirname(output_path)

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if not os.path.exists(output_path):
            session = requests.Session()
            response = session.get(
                GoogleDriveDownloader.DOWNLOAD_URL, params={"id": file_id}, stream=True
            )
            token = GoogleDriveDownloader._get_confirm_token(response)

            if token:
                params = {"id": file_id, "confirm": token}
                response = session.get(
                    GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True
                )

            GoogleDriveDownloader._save_response_content(response, output_path)

            if unzip:
                with zipfile.ZipFile(output_path, "r") as z:
                    z.extractall(output_directory)

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    @staticmethod
    def _save_response_content(response, output_path):
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
