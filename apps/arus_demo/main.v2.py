import logging
import os
import app_state as app
try:
    import backend
    import dashboard
except ImportError as e:
    msg = (
        "Arus demo requirements are not installed.\n\n"
        "Please install the demo extra packages as follows:\n\n"
        "  pip install arus[demo]\n\n"
        "  poetry add arus  --extras demo"
    )
    raise ImportError(str(e) + "\n\n" + msg)
import multiprocessing
import sys
import os

def start_app():
    logging.basicConfig(
        level=logging.DEBUG, format='[%(levelname)s]%(asctime)s <P%(process)d-%(threadName)s> %(message)s')
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    app_state = app.AppState.getInstance()
    app_state.initial_dataset = backend.load_initial_data()
    demo = dashboard.Dashboard(title='Arus Demo - Dashboard')
    demo.start()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    start_app()
    