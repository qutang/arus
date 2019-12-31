import logging
import os
import app_state as app
import backend
import dashboard

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format='[%(levelname)s]%(asctime)s <P%(process)d-%(threadName)s> %(message)s')
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    app_state = app.AppState.getInstance()
    app_state.initial_dataset = backend.load_initial_data()
    demo = dashboard.Dashboard(title='Arus Demo - Dashboard')
    demo.start()