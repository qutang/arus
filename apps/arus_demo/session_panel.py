import PySimpleGUI as sg
import app_state as app
import dashboard
import backend
import os
import logging
import traceback


def control_button(text, disabled, key=None):
    return sg.Button(button_text=text,
                     font=('Helvetica', 15), auto_size_button=True, size=(25, None), key=key, disabled=disabled)


class SessionSelectionPanel:
    def __init__(self):
        self._new_button = None
        self._continue_button = None

    def init_panel(self):
        self._new_button = control_button(
            'New session', disabled=False, key='_NEW_')
        self._continue_button = control_button(
            'Continue last session', disabled=not os.path.exists(app.AppState._snapshot_path), key='_CONTINUE_'
        )
        return sg.Window('Select a session to start', layout=[
            [self._new_button, self._continue_button]
        ], finalize=True)

    def start(self):
        window = self.init_panel()
        ready = False
        while True:
            event, _ = window.read()
            if event == self._continue_button.Key:
                logging.info('Restoring application status..')
                if app.AppState.restore():
                    app_state = app.AppState.getInstance()
                    ready = True
                else:
                    app_state = app.AppState.getInstance()
                    app_state.initial_dataset = backend.load_initial_data()
                    ready = True
            elif event == self._new_button.Key:
                app_state = app.AppState.getInstance()
                app_state.initial_dataset = backend.load_initial_data()
                ready = True
            elif event is None:
                break
            if ready:
                try:
                    demo = dashboard.Dashboard(
                        title='Arus Demo Session: ' + app_state.pid)
                    demo.start()
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                finally:
                    logging.info('Saving application status..')
                    app.AppState.snapshot()
        window.close()