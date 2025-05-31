# Face Recognition and Grouping


This project provides a complete face grouping and recognition system using Python. It includes a Jupyter Notebook interface with a simple UI built on `ipywidgets` to load images, group faces by similarity, and perform live webcam-based recognition.

---

## 📌 Overview

The application aims to:

1. **Load and preprocess images** from a user-specified folder.
2. **Detect faces** in each image using `face_recognition` (HOG model).
3. **Extract 128-dimensional face encodings** and group similar faces based on Euclidean distance.
4. **Save grouped faces** into separate folders on disk and create a ZIP archive of all grouped images.
5. **Provide a live webcam recognition mode** that matches a person’s face against the previously computed groups and allows downloading the matched group as a ZIP.
6. **Render a clean, responsive UI** (3-column grid) to display grouped images with hover overlays and full-screen preview.

This tool is especially useful for organizing large datasets of face images, managing photo collections, or testing simple face clustering and recognition pipelines.

---

## ✅ Features

* **Image Preprocessing & Compression**

  * Automatically resizes large images (max dimension = 800 px).
  * Compresses images over 1 MB to reduce memory usage.
  * Converts RGBA/Grayscale to RGB if needed.

* **Face Detection & Encoding**

  * Detects faces using `face_recognition.face_locations(..., model="hog")`.
  * Extracts face encodings with `face_recognition.face_encodings(..., num_jitters=3)` for better stability on small datasets.

* **Face Grouping (Clustering)**

  * Compares each face encoding to existing group centroids using Euclidean distance.
  * Default distance threshold = 0.45 (modify as needed in code).
  * Automatically creates group labels (“Face 1”, “Face 2”, …) and organizes multiple images of the same person into the same group.

* **Output & ZIP Generation**

  * Saves each group in its own subfolder under `individual_faces_output/`.
  * Generates a single ZIP archive (`all_faces.zip`) containing all grouped folders.
  * In webcam mode, creates a separate ZIP for the matched face group.

* **Interactive UI**

  * Folder path input widget to select the source image folder.
  * “Process Images” button to run grouping.
  * “Recognize via Webcam” button to run live face matching.
  * Progress bar to show image loading/grouping progress.
  * Output area renders a summary, download links, and a grid of grouped faces with hover overlays and full-screen preview.

---

## 🧰 Requirements

Ensure you have **Python 3.6+** installed. The main dependencies are:

```bash
pip install cmake
pip install dlib --verbose
pip install numpy==1.26.4
pip install opencv-python==4.11.0.86
pip install pillow==10.3.0
pip install ipywidgets==7.8.1
pip install face_recognition==1.3.0
pip install git+https://github.com/ageitgey/face_recognition_models
pip install scikit-learn
pip install tensorflow
```

* **cmake & dlib**: Required for the `face_recognition` library (calculates face landmarks and encodings).
* **numpy, opencv-python, Pillow**: Image loading, resizing, and format conversions.
* **ipywidgets**: Build the interactive UI inside Jupyter Notebook.
* **face\_recognition 1.3.0**: Core face detection and encoding functions.
* **face\_recognition\_models**: Pretrained models for face landmark detection and recognition.
* **scikit-learn**: (If you want to extend grouping/clustering logic beyond Euclidean threshold).
* **tensorflow**: (Optional; required if you plan to integrate any deep-learning enhancements.)

---

## 📂 Folder Structure

```
project/
│
├── images/                       # (User-provided) folder containing input images
│   ├── img1.jpg
│   ├── img2.png
│   └── …
│
├── individual_faces_output/      # Auto-created output folder
│   ├── Face 1/                   # Subfolder for Group 1
│   │   ├── imgA_UUID.jpg
│   │   └── imgB_UUID.jpg
│   ├── Face 2/                   # Subfolder for Group 2
│   │   ├── imgC_UUID.jpg
│   │   └── …
│   └── all_faces.zip             # ZIP containing all Face X folders
│
├── Project.ipynb                 # Jupyter Notebook with full code & UI
├── README.md                     # (This file)
└── requirements.txt (optional)   # If you prefer to list dependencies here
```

**Notes:**

* The `images/` folder should be created and populated by the user before running the notebook.
* After processing, the `individual_faces_output/` folder and `all_faces.zip` are automatically generated.
* In webcam mode, if a match is found for “Face K”, a ZIP named `Face K.zip` is written to `individual_faces_output/`.

---

## 🚀 How to Run

1. **Clone or Download** this repository to your local machine.

2. **Install Dependencies** using the commands above (or `pip install -r requirements.txt` if you create one).

3. **Launch Jupyter Notebook** in the project directory:

   ```bash
   jupyter notebook Project.ipynb
   ```

4. **In the Notebook**:

   * Locate the text input at the top labelled **“Folder Path”**.
   * Enter the full path to your `images/` folder (e.g., `/home/user/project/images/` or `C:\Users\username\project\images\`).
   * Click the **“Process Images”** button.

     * The UI will display a progress bar while loading each image (skipping files > 5 MB).
     * Face detection and encoding runs in the background; you will see console logs of processed filenames.
     * Once completed, you will get a summary (timestamp, total images processed, number of unique faces) and a download link for `all_faces.zip`.
     * A grid of grouped faces will appear, organized by “Face 1”, “Face 2”, etc.

       * Hover over any image to see the group label and original filename overlay.
       * Click on an image to view it full-screen in the modal.

5. **Webcam Recognition**:

   * After you have grouped faces at least once, click **“Recognize via Webcam”**.
   * Grant webcam permission if prompted.
   * The notebook will scan frames in real-time.
   * If a face matches any existing group (distance < 0.45), the loop stops, and you see a message like:

     ```
     Match found: Face 3.
     [Download Face 3.zip]
     ```
   * The matched group’s images are zipped into `individual_faces_output/Face 3.zip` with a download link.

6. **Exit Webcam Mode**:

   * Press the `q` key in the OpenCV window or close the notebook cell to terminate.

---

## ⚙️ Configuration & Customization

* **Distance Threshold** (`threshold = 0.45`):

  * Defines face similarity. Decrease to make grouping stricter (fewer false positives). Increase to allow more variance within a group.
  * Modify in the `group_similar_faces(...)` function near the top of the notebook.

* **Image Size & Compression**:

  * `max_size = 800` in `preprocess_image(...)`:

    * Ensures the longer side of an image does not exceed 800 px.
    * Reducing this number will decrease memory usage at the cost of encoding accuracy.
  * JPEG quality settings (`quality=85` for initial compression, `quality=75` if over 1 MB):

    * Adjust depending on your hardware constraints versus accuracy requirements.

* **Output Folder** (`output_folder = "individual_faces_output"`):

  * Change the name/path if you prefer a different location for saving grouped folders and ZIP files.

* **Model Selection**:

  * Currently uses the HOG model (`model="hog"`).
  * You can switch to the CNN model by changing `face_recognition.face_locations(image_array, model="cnn")` if you have a GPU-enabled environment. Note that the CNN model is more accurate but slower.

* **UI Appearance**:

  * Modify the CSS inside `after_ui_html` to change styling, grid layout, colors, or font sizes.

---

## 📸 Output Details

* **Grouped Images**:

  * After running “Process Images”, you will see a tiled gallery in the notebook.
  * Each group is preceded by a “Face X” label.
  * Filenames are displayed at the bottom overlay of each thumbnail.
  * Click any thumbnail to enlarge it in a full-screen modal overlay.

* **Folder Layout**:

  ```
  individual_faces_output/
  ├── Face 1/
  │   ├── sample_image1_a1b2c3d4.jpg
  │   ├── sample_image2_e5f6g7h8.jpg
  │   └── …
  ├── Face 2/
  │   ├── sample_image3_i9j0k1l2.jpg
  │   └── …
  ├── Face 3/
  │   ├── sample_image4_m3n4o5p6.jpg
  │   └── …
  └── all_faces.zip
  ```

  * Each file’s name is appended with an 8-character UUID hex to avoid collisions if two input images share the same filename.

* **Webcam Mode ZIP**:

  * If the webcam matches a person from a group (e.g., “Face 2”), the notebook creates `Face 2.zip` under `individual_faces_output/`.
  * You can download that ZIP directly from the HTML link that appears after a match is found.

---

## 🙋 Troubleshooting

1. **No Faces Detected**

   * Ensure your images are clear, well-lit, and show a frontal face.
   * Try increasing `num_jitters` in `face_encodings(...)` or lowering the grouping threshold.

2. **Errors Installing dlib**

   * On some systems you may need to install system packages for CMake, Boost, or build tools (e.g., `sudo apt-get install build-essential cmake`).
   * Try `pip install cmake` first, then re-run `pip install dlib --verbose`.

3. **Webcam Not Detected**

   * Verify your webcam is connected and not being used by another application.
   * Check that OpenCV can open device index 0 (`cap = cv2.VideoCapture(0)`).
   * If you have multiple cameras, try `cv2.VideoCapture(1)` or another index.

4. **Performance & Memory Constraints**

   * Downscale images further by lowering `max_size` in `preprocess_image`.
   * Remove `num_jitters=3` or set it to `0` for faster encoding at the expense of slight accuracy loss.
   * Use GPU/CNN model for faster, more accurate face detection if available.

---

## 📃 License

This project is licensed under the MIT License. Feel free to fork, modify, and distribute with proper attribution.

---

## 🙏 Acknowledgments

* [face\_recognition (by Adam Geitgey)](https://github.com/ageitgey/face_recognition)
* [dlib C++ Library](http://dlib.net/)
* [OpenCV](https://opencv.org/)
* [ipywidgets Documentation](https://ipywidgets.readthedocs.io/)

Thank you for using and exploring this face grouping and recognition application!
