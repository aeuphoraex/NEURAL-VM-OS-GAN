@echo off
echo ========================================
echo GAN-Powered Neural VM Converter
echo ========================================
echo.

REM Check if input file exists
if not exist "SKELETON_NEURAL_F16.gguf" (
    echo [!] Error: SKELETON_NEURAL_F16.gguf not found!
    echo Please place your GGUF file in this directory.
    pause
    exit /b 1
)

echo [*] Converting SKELETON_NEURAL_F16.gguf to GAN-powered Neural VM...
echo.

REM Run the converter
python gguf_gan_vm.py SKELETON_NEURAL_F16.gguf

if %ERRORLEVEL% NEQ 0 (
    echo [!] Conversion failed!
    pause
    exit /b 1
)

echo.
echo [*] Conversion complete!
echo [*] Output: SKELETON_NEURAL_F16_GAN_VM_F16.gguf
echo.
echo [*] You can now run the VM with:
echo     python gan_vm_runtime.py SKELETON_NEURAL_F16_GAN_VM_F16.gguf status
echo     python gan_vm_runtime.py SKELETON_NEURAL_F16_GAN_VM_F16.gguf bench
echo     python gan_vm_runtime.py SKELETON_NEURAL_F16_GAN_VM_F16.gguf train 1000
echo.
echo [*] Or open GAN_VM_MONITOR.html for web interface
echo.
pause
