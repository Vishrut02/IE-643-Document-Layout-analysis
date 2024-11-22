# IE-643-Document-Layout-analysis
IE 643 project final
## VGG-19 training code
### The directory explaination for VGG_19
```text
These code do EDA for preparing the data 
The final data created through these is given in link to the drive
https://drive.google.com/file/d/1UD4E-rWnReUaTMp1gsdsyhwyyM8ZJuNA/view?usp=sharing . So this has a folder marmot_usuals which contains the images of documents , marmot_columns contain the column mask and marmot_table contain the table mask.
```
```bash
./Data_Extraction/eda_1.py
./Data_Extraction/eda_2.py
./Data_Extraction/eda_3.py
```

So this has a folder marmot_usuals which contains the images of documents , marmot_columns contain the column mask and marmot_table contain the table mask.

./Some_Results/Table_Test_1.ipynb
./Training/First_Train_VGG.ipynb
./Training/Second_Train_VGG.ipynb
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
