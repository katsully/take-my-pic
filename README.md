using python 3.6 64 bit to run<br>
To install go [here](https://www.python.org/downloads/release/python-361/). Click on Windows x86-64 executable installer.<br>
	* make sure to add python.exe to the path<br>
	* do not remove any of the optional features during setup

Make sure cmake is installed (if not use msi installer)

Make sure Visual Studio has C++ for CMake checked off

For best practice, all python libraries should be installed in a virtual environment

pip install -r requirements.txt 

Must run script from the src directory



# Tensor Flow

if you get the error: Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found



1. install cuda toolkit: https://developer.nvidia.com/cuda-downloads
   - windows, x86_64, 10, exe
2. if you have GE Force installed already, do a custom install and unclick
   - GeForce Experience Software
   - GeForce Drivers
   - PhysX
3. restart terminal windows



### if you have issues with cudnn

1. update NVIDIA graphic drivers

2. download cudnn for windows: https://developer.nvidia.com/rdp/cudnn-download

3. Navigate to your <installpath> directory containing cuDNN.

4. Unzip the cuDNN package.

   ```
   cudnn-x.x-windows-x64-v8.x.x.x.zip
   ```

   or

   ```
   cudnn-x.x-windows10-x64-v8.x.x.x.zip
   ```

5. Copy the following files into the CUDA Toolkit directory.

   

   1. Copy <installpath>\cuda\bin\cudnn*.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\bin.
   2. Copy <installpath>\cuda\include\cudnn*.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\include.
   3. Copy <installpath>\cuda\lib\x64\cudnn*.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\lib\x64.

6. Set the following environment variables to point to where cuDNN is located. To access the value of the $(CUDA_PATH) environment variable, perform the following steps:

   

   1. Open a command prompt from the **Start** menu.

   2. Type Run and hit **Enter**.

   3. Issue the control sysdm.cpl command.

   4. Select the **Advanced** tab at the top of the window.

   5. Click **Environment Variables** at the bottom of the window.

   6. Ensure the following values are set:

      ```bash
      Variable Name: CUDA_PATH 
      Variable Value: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x
      ```

