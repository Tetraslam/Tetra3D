import face_recognition
import os
import json
import numpy as np
from time import time
from concurrent.futures import ThreadPoolExecutor

encodings_file = 'face_encodings.json'

# Function to encode a single face
def encode_face(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if face_encodings:
        return image_path, face_encodings[0].tolist()
    else:
        print(f"No faces found in {image_path}. Skipping this image.")
        return image_path, None

# Function to load or generate face encodings with parallel processing
def load_face_encodings(load_folder):
    face_encodings = {}

    # Load existing encodings from file if it exists
    if os.path.exists(encodings_file):
        with open(encodings_file, 'r') as file:
            face_encodings = json.load(file)

    # Prepare a list of images to process
    images_to_process = [os.path.join(load_folder, filename) for filename in os.listdir(load_folder)
                         if filename.endswith(('.png', '.jpg', '.jpeg')) and filename not in face_encodings]

    # Use ThreadPoolExecutor to process images in parallel
    with ThreadPoolExecutor() as executor:
        results = executor.map(encode_face, images_to_process)

        for image_path, encoding in results:
            if encoding:
                filename = os.path.basename(image_path)
                face_encodings[filename] = encoding
                print(f"Encoded new face for {filename}")

    # Save updated encodings to file
    with open(encodings_file, 'w') as file:
        json.dump(face_encodings, file)

    return {k: np.array(v) for k, v in face_encodings.items()}

# Function to identify faces in the 'identify' folder
def identify_faces(identify_folder, face_encodings):
    for filename in os.listdir(identify_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            image_path = os.path.join(identify_folder, filename)
            unknown_image = face_recognition.load_image_file(image_path)

            # Find the first face in this image
            face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=0, model="hog")
            if face_locations:
                face_location = face_locations[0]
                unknown_face_encoding = face_recognition.face_encodings(unknown_image, [face_location])[0]

                distances = face_recognition.face_distance(list(face_encodings.values()), unknown_face_encoding)

                if len(distances) == 0:
                    print(f"No known faces to compare with {filename}")
                    continue

                best_match_index = distances.argmin()
                best_match_distance = distances[best_match_index]

                if best_match_distance <= 0.6:
                    best_match_filename = list(face_encodings.keys())[best_match_index]
                    best_match_filename = os.path.splitext(best_match_filename)[0]
                    print(f"Best match for face in {filename} is {best_match_filename} with a distance of {best_match_distance:.2f}")
                else:
                    print(f"No reliable match found for face in {filename}")
            else:
                print(f"No faces detected in {filename}")

if __name__ == "__main__":
    start_time = time()

    # Paths to folders
    load_folder = './load'
    identify_folder = './identify'

    # Load encodings
    face_encodings = load_face_encodings(load_folder)

    # Identify faces
    identify_faces(identify_folder, face_encodings)
    print(f"Time taken: {round(time() - start_time, 3)} seconds")
