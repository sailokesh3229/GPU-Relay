"""AI fix (spec §20, plan A1-A3): a failed job's code fixed by the user's own AI subscription.

* :mod:`request`: what is sent (the code, the error and where it happened, the traceback) and
  how the answer becomes a per-cell change with a diff;
* :mod:`claude_code`: Claude Code (``claude -p``) on the user's computer, with no tools at all;
* :mod:`board`: the fix requests of one user's GPU Relay and the helper that answers them;
* :mod:`helper`: ``gpurelay helper``, the laptop side for the website.
"""
