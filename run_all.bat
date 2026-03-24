@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Self-contained package: ROOT = this folder (f). Same pipeline as parent hmwk used to run.
REM Optional env: SKIP_BUILD=1  SKIP_PERF_PY=1  SKIP_PIP_MPL=1  SKIP_ACCURACY=1  ACCURACY_LIMIT=N
REM SKIP_ROOT_LAUNCHER=1  skip building package-root local_llm.exe launcher at end
REM HMwkLauncher=1  set by package-root launcher; do not rebuild while it is running

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "LLAMA_DIR=%ROOT%\llama.cpp"
set "MODEL_DIR=%ROOT%\models"
set "OUT_DIR=%ROOT%\test_results"
set "BUILD_ROOT=%ROOT%\build"
set "BIN_RELEASE=%BUILD_ROOT%\bin\Release"
set "EXE=%BIN_RELEASE%\local_llm.exe"
set "BENCH=%BIN_RELEASE%\llama-bench.exe"
set "DATASET=%ROOT%\data\appointment_cert_dataset.jsonl"

set "GGUF_NAME=SmolLM2-135M-Instruct-Q4_0.gguf"
set "GGUF_URL=https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/%GGUF_NAME%"
set "MODEL_PATH=%MODEL_DIR%\%GGUF_NAME%"

if not defined LLAMA_REPO set "LLAMA_REPO=https://github.com/ggerganov/llama.cpp.git"

if exist "%ROOT%\local_llm.exe" for %%A in ("%ROOT%\local_llm.exe") do if %%~zA equ 0 del "%ROOT%\local_llm.exe"

echo [INFO] ROOT=%ROOT%
echo.

where git >nul 2>&1
if errorlevel 1 (
  echo [ERR] Git not in PATH.
  goto fail
)
where cmake >nul 2>&1
if errorlevel 1 (
  echo [ERR] CMake not in PATH.
  goto fail
)

if not exist "%LLAMA_DIR%\.git" (
  echo [1] git clone llama.cpp ...
  git clone --depth 1 "%LLAMA_REPO%" "%LLAMA_DIR%"
  if errorlevel 1 (
    echo [ERR] git clone failed.
    goto fail
  )
) else (
  echo [1] git pull llama.cpp ...
  pushd "%LLAMA_DIR%"
  git pull --ff-only
  popd
)

if /i "%SKIP_BUILD%"=="1" goto after_build
if exist "%BIN_RELEASE%\local_llm.exe" (
  echo [2] Found build\bin\Release\local_llm.exe, skip compile.
  set "EXE=%BIN_RELEASE%\local_llm.exe"
  set "BENCH=%BIN_RELEASE%\llama-bench.exe"
  goto after_build
)
if exist "%LLAMA_DIR%\build\bin\Release\local_llm.exe" (
  echo [2] Found llama.cpp\build\...\local_llm.exe, skip package CMake.
  goto after_build
)
if exist "%LLAMA_DIR%\build\bin\Release\llama-completion.exe" (
  echo [2] Found llama.cpp\build\...\llama-completion.exe, skip package CMake.
  goto after_build
)
echo [2] CMake package + build local_llm_exe ...
cmake -S "%ROOT%" -B "%BUILD_ROOT%"
if errorlevel 1 (
  echo [WARN] CMake failed, try llama.cpp\build exe.
  goto after_build
)
cmake --build "%BUILD_ROOT%" --config Release -t local_llm_exe --parallel %NUMBER_OF_PROCESSORS%
if errorlevel 1 (
  echo [WARN] Build failed, try llama.cpp\build exe.
)
set "EXE=%BIN_RELEASE%\local_llm.exe"
set "BENCH=%BIN_RELEASE%\llama-bench.exe"
:after_build

if not exist "%EXE%" set "EXE=%ROOT%\llama.cpp\build\bin\Release\local_llm.exe"
if not exist "%EXE%" set "EXE=%ROOT%\llama.cpp\build\bin\Release\llama-completion.exe"
if not exist "%BENCH%" set "BENCH=%ROOT%\llama.cpp\build\bin\Release\llama-bench.exe"

if not exist "%EXE%" (
  echo [ERR] No local_llm.exe or llama-completion.exe.
  goto fail
)

for %%F in ("%EXE%") do set "_EXENAME=%%~nxF"
for %%F in ("%EXE%") do set "_EXEDIR=%%~dpF"
if /i "%_EXENAME%"=="llama-completion.exe" (
  copy /Y "%EXE%" "%_EXEDIR%local_llm.exe" >nul
  set "EXE=%_EXEDIR%local_llm.exe"
)

echo [INFO] EXE=%EXE%

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if exist "%MODEL_PATH%" goto have_model
echo [3] Download GGUF ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%GGUF_URL%' -OutFile '%MODEL_PATH%' -UseBasicParsing"
if errorlevel 1 (
  echo [ERR] Download failed.
  goto fail
)
:have_model

if not exist "%ROOT%\data" mkdir "%ROOT%\data"
if not exist "%DATASET%" (
  echo [3b] Generate dataset ...
  where python >nul 2>&1
  if errorlevel 1 (
    echo [ERR] Python needed for gen_appointment_dataset.py
    goto fail
  )
  python "%ROOT%\tests\gen_appointment_dataset.py"
  if errorlevel 1 (
    echo [ERR] dataset generation failed.
    goto fail
  )
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
for /f "delims=" %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%t"
set "RUN_DIR=%OUT_DIR%\run_%TS%"
mkdir "%RUN_DIR%"
set "REPORT=%RUN_DIR%\test_report.txt"

echo [INFO] This run output folder: %RUN_DIR%

echo [4] Smoke + bench -^> %REPORT%

echo ============================================ > "%REPORT%"
echo f\run_all.bat >> "%REPORT%"
echo run_folder: %RUN_DIR% >> "%REPORT%"
echo Time: %DATE% %TIME% >> "%REPORT%"
echo EXE=%EXE% >> "%REPORT%"
echo MODEL=%MODEL_PATH% >> "%REPORT%"
echo ============================================ >> "%REPORT%"
echo. >> "%REPORT%"
echo [A] inference smoke >> "%REPORT%"
echo. >> "%REPORT%"

"%EXE%" -m "%MODEL_PATH%" -no-cnv -p "Hello" -n 32 -ngl 0 --perf >> "%REPORT%" 2>&1

echo. >> "%REPORT%"
echo [B] llama-bench >> "%REPORT%"
echo. >> "%REPORT%"

if exist "%BENCH%" (
  "%BENCH%" -m "%MODEL_PATH%" -ngl 0 -p 64 -n 64 -r 2 -o md >> "%REPORT%" 2>&1
) else (
  echo [SKIP] llama-bench.exe not found >> "%REPORT%"
)

if /i "%SKIP_PERF_PY%"=="1" goto no_py
where python >nul 2>&1
if errorlevel 1 (
  echo [WARN] Python not in PATH, skip perf_smoke_test.py >> "%REPORT%"
  goto no_py
)
echo [5] perf_smoke_test.py -^> %RUN_DIR%
echo. >> "%REPORT%"
if /i not "%SKIP_PIP_MPL%"=="1" (
  python -m pip install -q -r "%ROOT%\requirements.txt" 2>nul
)
python "%ROOT%\tests\perf_smoke_test.py" --root "%ROOT%" --exe "%EXE%" --model "%MODEL_PATH%" --out-dir "%RUN_DIR%" --append-text-report "%REPORT%"
:no_py

if /i "%SKIP_ACCURACY%"=="1" goto no_acc
where python >nul 2>&1
if errorlevel 1 (
  echo [WARN] Python not in PATH, skip accuracy_eval >> "%REPORT%"
  goto no_acc
)
echo [6] accuracy_eval.py ^(200 samples can take 30-60+ min; quick test: set ACCURACY_LIMIT=20^)
echo. >> "%REPORT%"
echo [D] accuracy_eval data\appointment_cert_dataset.jsonl >> "%REPORT%"
echo. >> "%REPORT%"
if defined ACCURACY_LIMIT (
  python "%ROOT%\tests\accuracy_eval.py" --exe "%EXE%" --model "%MODEL_PATH%" --dataset "%DATASET%" --limit %ACCURACY_LIMIT% --out-json "%RUN_DIR%\accuracy_full.json" --out-txt "%RUN_DIR%\accuracy_summary.txt"
) else (
  python "%ROOT%\tests\accuracy_eval.py" --exe "%EXE%" --model "%MODEL_PATH%" --dataset "%DATASET%" --out-json "%RUN_DIR%\accuracy_full.json" --out-txt "%RUN_DIR%\accuracy_summary.txt"
)
if exist "%RUN_DIR%\accuracy_summary.txt" type "%RUN_DIR%\accuracy_summary.txt" >> "%REPORT%"
:no_acc

if /i "%SKIP_ROOT_LAUNCHER%"=="1" goto skip_launcher
if /i "%HMwkLauncher%"=="1" goto skip_launcher
if not exist "%ROOT%\launcher\main.c" goto skip_launcher
echo [7] Build package-root local_llm.exe ^(double-click = re-run tests^) ...
set "LEX=%ROOT%\local_llm.exe"
set "VSW=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSW%" set "VSW=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSDIR="
if exist "%VSW%" for /f "usebackq delims=" %%i in (`"%VSW%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
  echo [WARN] vswhere / VS C++ not found; skip package-root launcher.
  goto skip_launcher
)
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" (
  echo [WARN] vcvars64.bat missing; skip package-root launcher.
  goto skip_launcher
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
pushd "%ROOT%\launcher"
cl /nologo /W3 /O2 /Fe:"%LEX%" main.c /link /SUBSYSTEM:CONSOLE
set "CLERR=%ERRORLEVEL%"
del /q main.obj 2>nul
popd
if not "%CLERR%"=="0" (
  echo [WARN] cl failed ^(code %CLERR%^); skip package-root launcher.
) else (
  echo [OK] Package-root launcher: %LEX%
)
:skip_launcher

echo.
echo ============================================================
echo [OK] Done. Output folder:
echo      %RUN_DIR%
echo      - test_report.txt
echo      - perf_smoke_*.json .md .png
echo      - accuracy_full.json accuracy_summary.txt  ^(if not SKIP_ACCURACY^)
echo      - local_llm.exe in this folder ^(launcher; with args = infer^)
echo ============================================================
goto end

:fail
echo.
echo [FAILED] See messages above.
pause
endlocal
exit /b 1

:end
echo.
pause
endlocal
exit /b 0
