"""Entry point of the packaged desktop app (``GPU Relay.exe``).

Without arguments it opens the desktop app; with arguments it runs a CLI command (that is
how the app starts its controller: ``"GPU Relay.exe" up``).
"""

from gpurelay.desktop.app import main

if __name__ == "__main__":
    main()
