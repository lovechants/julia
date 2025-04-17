try:
    from julia.core.backends.clang.compiler import ClangCompiler, JITCompileOpClang
    CLANG_AVAILABLE = True
except ImportError:
    CLANG_AVAILABLE = False
