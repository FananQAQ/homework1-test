#define _CRT_SECURE_NO_WARNINGS
#include <io.h>
#include <process.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static void dirname_in_place(char *path)
{
    char *p = strrchr(path, '\\');
    if (p)
        *p = '\0';
}

static int file_exists(const char *p)
{
    return _access(p, 0) == 0;
}

static int find_infer_exe(const char *root, char *out, size_t outsz)
{
    static const char *suffixes[] = {
        "\\llama.cpp\\build\\bin\\Release\\local_llm.exe",
        "\\build\\bin\\Release\\local_llm.exe",
        "\\llama.cpp\\build\\bin\\Release\\llama-completion.exe",
    };
    size_t i;
    for (i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        int n = snprintf(out, outsz, "%s%s", root, suffixes[i]);
        if (n < 0 || (size_t)n >= outsz)
            continue;
        if (file_exists(out))
            return 0;
    }
    return -1;
}

int main(int argc, char **argv)
{
    char root[MAX_PATH];
    char cmd[4096];
    char infer[MAX_PATH];
    int n;

    if (!GetModuleFileNameA(NULL, root, MAX_PATH))
        return 1;
    dirname_in_place(root);

    if (argc < 2) {
        n = snprintf(cmd, sizeof(cmd),
            "cmd.exe /c \"cd /d \"%s\" && set HMwkLauncher=1&& call run_all.bat\"",
            root);
        if (n < 0 || (size_t)n >= sizeof(cmd))
            return 1;
        return system(cmd) == 0 ? 0 : 1;
    }

    if (find_infer_exe(root, infer, sizeof(infer)) != 0) {
        fprintf(stderr,
            "local_llm: could not find local_llm.exe or llama-completion.exe under build folders.\n"
            "Run run_all.bat once to build.\n");
        return 1;
    }

    {
        char **av = (char **)calloc((size_t)argc + 1, sizeof(char *));
        int i;
        if (!av)
            return 1;
        av[0] = infer;
        for (i = 1; i < argc; i++)
            av[i] = argv[i];
        av[argc] = NULL;
        _execvp(infer, (const char *const *)av);
        free(av);
    }

    perror("_execvp");
    return 1;
}
