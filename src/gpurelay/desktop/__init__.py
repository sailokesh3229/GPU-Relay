"""GPU Relay desktop app (plan D2-D4): a native window, a tray icon and notifications.

Two modes, chosen in the setup wizard:

* **This computer**: the app runs the controller (``gpurelay up``) and the web app locally.
* **My server**: the controller runs on an always-on server (``gpurelay serve``); the app
  shows that server's web app and notifies about its jobs.
"""
