# IE-643-Document-Layout-analysis
IE 643 project final
## VGG-19 training code
### The directory explaination for VGG_19
```text
These code do EDA for preparing the data 
The final data created through these is given in link to the drive
https://drive.google.com/file/d/1UD4E-rWnReUaTMp1gsdsyhwyyM8ZJuNA/view?usp=sharing 
```
```bash
./Data_Extraction/eda_1.py
./Data_Extraction/eda_2.py
./Data_Extraction/eda_3.py
```
So this has a folder marmot_usuals which contains the images of documents , marmot_columns contain the column mask and marmot_table contain the table mask.

 ```bash
./Some_Results/Table_Test_1.ipynb
./Training/First_Train_VGG.ipynb
./Training/Second_Train_VGG.ipynb
```
The above code specify the training for these would need models which you can download from the drive link and appropiately looking at model loaded in code load that in directory 
https://drive.google.com/file/d/1Bqr2MuTvhabSeRoxzysBr_uo_Pe2pVGi/view?usp=sharing

---

### Dependencies

To run this project, the following dependencies are required:

#### Python Version
- Python 3.x

#### Libraries
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **xml.etree.ElementTree**: For XML parsing.
- **xml.dom.minidom**: For working with XML documents.
- **Pillow (PIL)**: For image processing tasks.
- **OpenCV (cv2)**: For advanced image processing.
- **Matplotlib**: For visualization and plotting.
- **TensorFlow/Keras**: For creating and training deep learning models.

#### Installation
You can install the required dependencies using `pip`:

```bash
pip install numpy pandas pillow opencv-python matplotlib tensorflow keras
```

## Faster R-CNN 

### This is for the directory FasterRCNN

#### The Subdirecotries and code explaination
These codes below do the EDA ,the final data made from them can be found in the drive link  https://drive.google.com/file/d/1J_wwzbMr_Z1-u6uG_l8e0lOOouUiAMgG/view?usp=sharing 
```bash
./Data_Extraction/eda_1.ipynb
./Data_Extraction/eda_2.ipynb
./Data_Extraction/eda_3.ipynb
```
The codes below require you to load the model from the drive link https://drive.google.com/file/d/1ShKhrF8sVbYHoOCCvt6w0d89wtc5EEOj/view?usp=sharing
```bash 
./Testing_Code/Test_1.ipynb
./Training_Code/Execution_stopped.png
./Training_Code/First_Train_RCNN.ipynb
./Training_Code/Second_Train_RCNN.ipynb
./Training_Code/Third_Train_RCNN.ipynb
```

### Dependencies

Below are the dependencies required for this project:

### Python Version
- Python 3.x

### Libraries
- **PyTorch**: For deep learning model creation, training, and evaluation.
- **Torchvision**: For pre-trained Faster R-CNN models and image transformations.
- **TQDM**: For progress bars during training and evaluation.
- **Matplotlib**: For visualization of images, bounding boxes, and results.
- **Pillow (PIL)**: For image loading and manipulation.
- **NumPy**: For numerical computations.
- **Requests**: For downloading files from external sources.
- **JSON**: For handling annotation files in JSON format.

### Installation
To install the required dependencies, use the following `pip` command:

```bash
pip install torch torchvision tqdm matplotlib pillow numpy requests
```

## Combined_Experimentation

### This is for the zip file Combined_Testiing

---

1. **Table and Column Detection**:
   - Utilizes a fine-tuned **Faster R-CNN** model to detect tables and their column structures in images.
   - A **VGG-19-based segmentation** model is integrated to refine the table masks and extract precise column boundaries.

2. **Line Boundary Detection**:
   - Heuristic methods are applied to the column masks to identify individual line boundaries within the table.

3. **Caption Extraction**:
   - Based on the detected table, the corresponding caption is located heuristically to ensure association with the correct table.

4. **Optical Character Recognition (OCR)**:
   - After segmenting the table into rows and columns, **Pytesseract** is employed to extract textual data from each cell, generating a structured representation of the table.

5. **Output**:
   - The final output includes:
     - The **detected table with column masks and line boundaries visualized**.
     - A **textual representation of the table**.
     - The **table caption**.

---

Please correctly download the models from above links and load them .

### Dependencies

The project requires the following Python libraries and tools:

### Python Version
- Python 3.x

### Libraries and Frameworks
- **NumPy**: For numerical computations and array manipulations.
- **Pandas**: For data analysis and manipulation.
- **TensorFlow/Keras**: For building and training deep learning models.
- **PyTorch/Torchvision**: For object detection using pre-trained Faster R-CNN models.
- **Pillow (PIL)**: For image processing and manipulation.
- **Matplotlib**: For visualization of images, bounding boxes, and results.
- **OpenCV (cv2)**: For advanced image processing tasks.
- **Pytesseract**: For optical character recognition (OCR) to extract text from images.
- **ReportLab**: For generating PDFs programmatically.
- **Ast**: For safely evaluating Python expressions (e.g., for parsing annotations).
- **Re**: For working with regular expressions.

---

### Installation
To install the required dependencies, run the following command:

```bash
pip install numpy pandas tensorflow keras torch torchvision pillow matplotlib opencv-python pytesseract reportlab
```

# Document Layout Analysis with Object Detection and OCR 
##  This is for the interface code

This project provides a comprehensive pipeline for analyzing document layouts, detecting tables and figures, extracting captions, and performing OCR to structure textual data from documents. It uses:

- **Faster R-CNN**: For object detection to identify tables, figures, and text.
- **VGG-19**: For refining masks and segmenting columns in tables.
- **Pytesseract**: For Optical Character Recognition (OCR) to extract text.
- **Streamlit**: For creating an interactive web interface.

---

## Features

### **PDF to Image Conversion**
- Converts uploaded PDFs into high-resolution images for processing.

### **Object Detection**
- Detects tables, figures, and associated captions using Faster R-CNN.

### **Table Mask Refinement**
- Refines table masks and detects column boundaries using VGG-19.

### **OCR Integration**
- Extracts structured textual data from tables and captions using Pytesseract.

### **Interactive Interface**
- Provides an intuitive web-based interface using Streamlit for uploading PDFs and viewing results.

---

## Requirements

### **Python Version**
- Python 3.x

### **Libraries**
The following Python libraries are required:

- `torch`
- `torchvision`
- `tensorflow`
- `keras`
- `pandas`
- `numpy`
- `Pillow`
- `matplotlib`
- `cv2` (OpenCV)
- `pytesseract`
- `pdf2image`
- `streamlit`

---

## Installation

Install the required dependencies using `pip`:

```bash
pip install torch torchvision tensorflow keras pandas numpy pillow matplotlib opencv-python pytesseract pdf2image streamlit
```
## The Directory Structure of Output

So for Faster RCNN model and Vgg Model must correctly downloaded from above links and placed in the directory like this.

```bash 
├── Faster_RCNN/                # Directory containing the Faster R-CNN model
│   └── best_model (1).pth
├── VGG_19/                     # Directory containing the VGG-19 model
│   └── mymodel_277.keras
├── test_images/                # Directory where PDF pages will be saved as images
├── output/                     # Directory to store processed results
│   ├── tables/                 # Directory to store table images and CSVs
│   ├── figures/                # Directory to store figure images and captions
├── bounding_boxes.csv          # File containing detected bounding boxes
├── main.py                     # Main application script
└── README.md                   # Project documentation

```

### Usage
#### Step 1: Clone the Repository
Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/document-layout-analysis.git
cd document-layout-analysis
```
#### Step 2: Install Dependencies
Install the required dependencies:

```bash
pip install -r requirements.txt
```

#### Step 3: Run the Application
Run the Streamlit application using the following command:

```bash
streamlit run main.py
```

#### Step 4: Access the Web Interface
After running the application, Streamlit will provide a link (e.g., http://localhost:8501).
Open this link in your browser to access the application.

# Novelty Code :: CellNet: Semantic Segmentation for Cell Mask Prediction

This project implements a semantic segmentation model using a pre-trained **VGG-19** encoder and a custom decoder to predict cell masks. The pipeline includes data preprocessing, a TensorFlow-based data pipeline, and a fully functional encoder-decoder architecture.

---

## Features

### **Data Preprocessing**
- Processes images and corresponding masks for training and validation.
- Resizes images and masks to a uniform size of `1024x1024` and normalizes pixel values to `[0, 1]`.

### **Data Pipeline**
- Efficiently maps and caches datasets for training and testing.
- Supports dynamic shuffling, batching, and prefetching for improved performance.

### **Encoder-Decoder Architecture**
- **Encoder**: Utilizes the pre-trained **VGG-19** model up to the bottleneck layer.
- **Decoder**: Implements a custom decoder (`cell_decoder`) for up-sampling and mask prediction.

### **Visualization**
- The model architecture is visualized with all layers and shapes using TensorFlow's `plot_model`.

---

## Requirements

### **Python Version**
- Python 3.x

### **Libraries**
Install the following Python libraries using `pip`:
```bash
pip install tensorflow pandas numpy matplotlib pillow opencv-python
```

### Directory Strucutre

```bash
.
├── input/                              # Input images and masks directory
│   ├── val_1/                          # Input validation images
│   ├── output_mask_1/                  # Corresponding output masks
├── output/                             # Directory to save model outputs
├── main.py                             # Main script to define and train the model
├── README.md                           # Project documentation

```
For the val_1 and output_mask_1 data set please download from the link https://drive.google.com/file/d/1RY0KA0URu83mEoa9tKDKBTiSXZ3l7pBz/view?usp=sharing

Also while testing we need to input the model please download from the link
https://drive.google.com/file/d/1lQk4pXr77d2dnnlSC7GqG5BmueJMXuId/view?usp=sharing
