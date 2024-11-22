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


