Code style is good. It is even better if a tool can automatically
reformat the code.  ClangFormat is a good choice.  I configured it to
work with Emacs on my MacBook Pro.

1. Install clang-format.

        brew install clang-format

1. Setup Emacs plugin.  Add the following segment into `~/.emacs`:


        (load "/usr/local/Cellar/clang-format/2015-07-31/share/clang/clang-format.el")
        (global-set-key [C-M-tab] 'clang-format-buffer)

1. Choose the "Chromium" style by edting file `/usr/local/Cellar/clang-format/2015-07-31/share/clang/clang-format.el` to change the line

         (defcustom clang-format-style "file"

into

         (defcustom clang-format-style "Chromium"

Then, I am going to check the followings work:

1. Work with source files with the `.cu` extension.

   Yes, ClangFormat works with files with any extension.

1. Does only simple reformatting.

   Yes. Nothing with semantics changing.

1. Works with extended C++ syntax introduced by CUDA.

   TODO(yiwang): to confirm this.
