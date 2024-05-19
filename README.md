# Tetra3D: A K-Pop Idol Facial Recognition Script

## Project Overview

This project is a K-Pop idol face recognition script that identifies and encodes faces from a given set of images (located in the `load` folder). The primary purpose is to recognize and differentiate between faces of K-Pop idols using face recognition technology.

## Features

- Load and process images to find and encode faces.
- Parallel processing for faster face encoding.
- Save and load face encodings to and from a JSON file using TinyDB for caching.
- Skip images with no detectable faces.

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.9+
- `face_recognition` library
- `numpy`
- `concurrent.futures`
- `os`
- `json`

You can install the required packages using pip:

```sh
pip install -r requirements.txt
```

## Project Structure

```
Tetra3D/
├── load/ (add pictures of idols you want to train on)
├── identify/ (add pictures of idols to be identified)
├── .env (edit `load_folder` and `identify_folder` to the path on your machine)
├── .gitignore
├── face_encodings.json (this will be auto-generated on your first run)
└── main.py
```

- `main.py`: The main script containing the face encoding and processing functions.
- `face_encodings.json`: A JSON file to store the face encodings.
- `load/`: Directory where the images to be processed are stored.
- `.env`: Environment file for environment-specific variables.
- `.gitignore`: Git ignore file to exclude unnecessary files from version control.

## Usage

1. **Prepare the images:**

   Place the images of K-Pop idols you want to process in the `load/` directory. Ensure the images are in `.jpg`, or `.jpeg` format.

2. **Run the script:**

   Execute the `main.py` script to process the images and generate face encodings.

   ```sh
   python main.py
   ```

3. **Function Details:**

   - `encode_face(image_path)`: Encodes a single face from the given image path. Returns the image path and face encoding if a face is found, otherwise returns the image path and `None`.
   - `load_face_encodings(load_folder)`: Loads or generates face encodings for images in the specified folder. Utilizes parallel processing to speed up the encoding process.

## Example

Here's a brief example of how to use the script:

1. Add images of K-Pop idols to the `load/` folder.
2. Run the `main.py` script.
3. The script will process the images, encode the faces, and save the encodings to `face_encodings.json`.
4. If an image has no detectable faces, it will print a message and skip that image.