# Med-Extract

## Introduction
(add what was given)

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

The live link is not possible to be provided as the Vison Transformers and Zephyr 7B models are resourse intensive and was trained on a rented GPU and the interface link was also hosted on the same. The video of the product working with 2 test cases and 1 explanation can be found with this link: https://drive.google.com/drive/folders/1XCMSfR37r0sTu2Pjtzr9NVjmFFQixf1z?usp=drive_link
