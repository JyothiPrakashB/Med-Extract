# Med-Extract
![image](https://github.com/user-attachments/assets/837560a3-41de-4956-8602-1bd03d1d7052)

**NOTE**:
The live link is not possible to be provided as the Vison Transformers and Zephyr 7B models are resourse intensive and was trained on a rented GPU and the interface link was also hosted on the same. The video of the product working with 2 test cases and 1 explanation can be found with this link: https://drive.google.com/drive/folders/1XCMSfR37r0sTu2Pjtzr9NVjmFFQixf1z?usp=drive_link

## Introduction
MedExtract is an AI-powered tool designed to simplify complex medical communication by providing structured summaries of medical queries. It bridges the gap between patients and healthcare providers by supporting multilingual text (Hindi-English code-mixed) and medical images, enabling a more accessible and inclusive healthcare experience.

By combining advanced Natural Language Processing (NLP) with Vision Transformers, MedExtract efficiently processes both textual descriptions and clinical imagery to generate meaningful summaries that aid in diagnosis and decision-making. Whether it’s a patient describing symptoms in mixed language or a doctor needing quick insight into a case, MedExtract ensures communication is fast, accurate, and effortless.

## Directory structure
```
Med-Extract/
│
├── Model testing/                      
│   ├── multimodal_zephyr.py               
│   ├── test_multimodal_zephyr.py          
│   └── checkpoint_zephyr.pt  # THIS WILL BE SAVED AFTER RUNNING THE multimodal_zephyr.py SCRIPT              
│
├── data/                           
│   ├── train.csv                        
│   ├── val.csv                          
│   ├── test.csv                  
│   ├── Multimodal_images/  # UNZIP THE FILE PROVIDED VIA DRIVE LINK
│       ├── cyanosis
│       ├── dry scalp
│       └── ...              
│
├── new_chat.py                 
├── app.py                             
└── require.txt                        
```

## Runnig the files
Environment setup:
```
conda create -n medextract python=3.9
```
Install requirements:
```
pip install -r require.txt
```
Train model and save checkpoints:
```
python multimodal_zephyr.py
```
Load checkpoints and test (batch testing):
```
python test_multimodal_zephyr.py
```
CLI command for custom input (single input testing):
```
python new_chat.py --checkpoint /path/to/checkpoint.pt --image /path/to/image.jpg --text "Your Hinglish text here" --device cuda
```
Interface running: 
```
python app.py
```


