import PySimpleGUI as sg
import app_state as app
import dashboard
import backend
import os
from loguru import logger
import traceback
import arus


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
                logger.info('Restoring application status..')
                file_path = sg.PopupGetFile('Select the pkl file to restore session', title='Continue a session',
                                            default_extension='.pkl', initial_folder=app.AppState._snapshot_path)
                if file_path is None:
                    continue
                if app.AppState.restore(file_path):
                    app_state = app.AppState.getInstance()
                    ready = True
                else:
                    app_state = app.AppState.getInstance()
                    app_state.origin_dataset = backend.load_origin_dataset()
                    ready = True
            elif event == self._new_button.Key:
                app.AppState.reset()
                app_state = app.AppState.getInstance()
                new_pid = sg.PopupGetText(
                    'Set new participant ID',
                    title='Create a new session',
                    default_text=app_state.pid,
                    keep_on_top=True
                )
                if new_pid is None:
                    continue
                app_state.pid = new_pid
                app_state.origin_dataset = backend.load_origin_dataset()
                ready = True
            elif event is None:
                break
            if ready:
                try:
                    log_file = os.path.join(
                        app.AppState._path, 'logs', app_state.pid + '.log')
                    os.makedirs(os.path.dirname(log_file), exist_ok=True)
                    logger.add(log_file)
                    demo = dashboard.Dashboard(
                        title='Arus Demo Session: ' + app_state.pid)
                    demo.start()
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                finally:
                    logger.info('Saving application status..')
                    app.AppState.snapshot()
        window.close()
