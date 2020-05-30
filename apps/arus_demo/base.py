import queue
import PySimpleGUI as sg
import enum
from loguru import logger


class BaseEvent(enum.Enum):
    CLOSE_WINDOW = enum.auto()


class BaseWindow:
    def __init__(self, title, column_width):
        self._title = title
        self._column_width = column_width
        self._events = queue.Queue()

    def start(self):
        self._state = self.init_states()
        self._window = self.init_views()
        while True:
            self.update_states_and_events()
            if self.dispatch_events():
                break
        self._window.close()

    def init_states(self):
        raise NotImplementedError(
            "Sub class must implements this method to initialize the app state before starting the window. This method should return an app.App_State object.")

    def init_views(self):
        raise NotImplementedError(
            "Sub class must implements this method to initialize the views of the window before starting the window. This method should return a sg.Window object.")

    def update_states_and_events(self):
        inherited_window_event, new_values = self._window.read(timeout=20)
        if inherited_window_event == sg.TIMEOUT_KEY:
            pass
        elif inherited_window_event is not None:
            self._events.put(inherited_window_event)

        if new_values is not None:
            self._update_states_and_events(inherited_window_event, new_values)

        if inherited_window_event is None:
            self._events.put(BaseEvent.CLOSE_WINDOW)

    def dispatch_events(self):
        for event in self._events.queue:
            logger.info('Dispatch event: ' + str(event))
            if event == BaseEvent.CLOSE_WINDOW:
                return True
            else:
                self._dispatch_events(event)
        self._events.queue.clear()
        return False

    def _update_states_and_events(self, event, values):
        raise NotImplementedError(
            'Sub class must implements this method to update states and events based on current app state and input events.')

    def _dispatch_events(self, event):
        raise NotImplementedError(
            "Sub class must implements this method to handle different events.")
