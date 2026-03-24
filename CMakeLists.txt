# Standalone package under f/: builds llama.cpp and produces local_llm.exe
cmake_minimum_required(VERSION 3.14...3.28)
project(f_local_llm LANGUAGES C CXX)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/CMakeLists.txt")
  message(FATAL_ERROR
    "Missing llama.cpp/CMakeLists.txt. Clone first:\n"
    "  git clone --depth 1 https://github.com/ggerganov/llama.cpp.git llama.cpp")
endif()

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

if(NOT TARGET llama-completion)
  message(FATAL_ERROR "Target llama-completion not found; check llama.cpp tools build options.")
endif()

if(WIN32)
  set(_local_llm_name "local_llm.exe")
else()
  set(_local_llm_name "local_llm")
endif()

add_custom_target(
  local_llm_exe ALL
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "$<TARGET_FILE:llama-completion>"
          "$<TARGET_FILE_DIR:llama-completion>/${_local_llm_name}"
  DEPENDS llama-completion
  COMMENT "Copy llama-completion -> local_llm (homework deliverable name)")
