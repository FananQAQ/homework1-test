Self-contained homework package (folder f).

Double-click run_all.bat here. Same workflow as parent hmwk\run_all.bat (it forwards to this folder).

CMake: cmake -S . -B build  then cmake --build build --config Release -t local_llm_exe

Tracked in git: tests/, data/, docs/, need.md, CMakeLists.txt, requirements.txt, launcher/, *.bat
Not tracked: llama.cpp (clone on first run), build/, models/*.gguf, test_results/
