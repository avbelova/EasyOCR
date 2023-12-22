## EasyOCR with OpenVINO backend installation guide
This is an instruction how to install EasyOCR library with integrated [OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino)   
Step 1. Install OpenVINO:
``` bash
python -m venv ov_env
source ov_env/bin/activate
pip install openvino-dev
```  
Step 2. Clone a fork with the integration, build a wheel file, install it into the same virtual environment
``` bash
git clone https://github.com/avbelova/EasyOCR.git
cd EasyOCR
git checkout feature/openvino/integration
pip install wheel
python setup.py bdist_wheel
pip install dist/<easyocr_wheel_name>.whl
```  
Step 3. Set up the path to the models. All needed models are [here](https://github.com/avbelova/EasyOCR/tree/feature/openvino/integration/openvino_models)  So, to make everything work, export OV_DET_MODEL_PATH and OV_REC_MODEL_PATH environmental variables with paths to .xml or .onnx files of the models, for example:  
``` bash
export OV_DET_MODEL_PATH=/<absolute path to>/easyocr_detector_en.xml
export OV_REC_MODEL_PATH=/<absolute path to>/1_recognition_model.xml
```  
Step 4. Choose the inference device. EasyOCR can be run by OpenVINO on Intel(R) CPU, Intel(R) Processor Graphics and Intel(R) Discrete Graphics. 
Export OV_DEVICE environmental variable to choose the inference device between “CPU”, “GPU”/"GPU.0" or “GPU.1” correspondingly.
The following example sets up a value to use Intel(R) Descrete Graphics:
``` bash
export OV_DEVICE=GPU.1 
```
Step 5. Use EasyOCR as usual. EasyOCR will print out which inference device is utilised, for example:  
       Text detection model is running with OpenVINO on GPU.1  
       Text recognition model is running with OpenVINO on GPU.1   
