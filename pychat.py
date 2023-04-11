import sys
import os
from enum import Enum
from PyQt5.QtCore import Qt, QEvent, pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, \
    QLineEdit, QPushButton, QListWidget, QSizePolicy, QDialog, QLabel, QMessageBox
from PyQt5.QtGui import QPalette

from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain import PromptTemplate


prompt_template = """
You are a chatbot named C-3PIO. Answer the questions asked using the formatting guidelines below. Do not make things up. 
If you are unsure about an answer, reply with "I am not sure how to answer your question"

Guidelines: If there is code in the reply, format the code with HTML code tags as follows:
<pre style="color: #ffffff; padding: 10px; font-size: 14px;">
  <h3 style="color: #ffffff;">{heading}</h3>
  <code>
    {insert generated code here}
  </code>
</pre>
If there is an image in the reply, format the image url with an HTML <image> tag
If there is a URL in the reply format the reply with an <A> HTML tag. 

Start the conversation by introducing yourself.
"""

SessionListChangedEventType = Enum('SessionListChangedEventType', ['Add', 'Remove'])


class AnyActiveSessionsEvent(QEvent):
    Type = QEvent.Type(QEvent.User + 1)

    def __init__(self, state):
        super().__init__(AnyActiveSessionsEvent.Type)
        self.state = state


class SessionListChangedEvent(QEvent):
    Type = QEvent.Type(QEvent.User + 2)

    def __init__(self, change_type, name):
        super().__init__(SessionListChangedEvent.Type)
        self.change_type = change_type
        self.name = name


class OpenAIChat:
    def __init__(self, model_name: str, api_key, temp=0.7, conversation_buffer_token_limit=1000):
        self.__model_name = model_name
        self.__temperature = temp
        self.__api_key = api_key
        self.__conversation_buffer_token_limit = conversation_buffer_token_limit

        self.__llm_factory()

    def __llm_factory(self):
        self.__llm = OpenAI(temperature=self.__temperature,
                            openai_api_key=self.__api_key,
                            model_name=self.__model_name)

        self.__conversation_buffer = ConversationSummaryBufferMemory(
            llm=self.__llm,
            max_token_limit=self.__conversation_buffer_token_limit
        )
        self.__conversation_chain = ConversationChain(
            llm=self.__llm,
            memory=self.__conversation_buffer
        )

    @property
    def temperature(self)->float:
        return self.__temperature

    @temperature.setter
    def temperature(self, value):
        self.__temperature = value
        self.__llm_factory()

    @property
    def model_name(self)->str:
        return self.__model_name

    @model_name.setter
    def model_name(self, value):
        self.__model_name = value
        self.__llm_factory()

    @property
    def conversation_buffer(self)->ConversationSummaryBufferMemory:
        return self.__conversation_buffer

    @property
    def conversation_chain(self)->ConversationChain:
        return self.__conversation_chain

class SessionlistWidget(QWidget):
    def __init__(self, chat_widget, session_item_changed_handler):
        super().__init__()

        self.__chat_widget = chat_widget
        self.__session_item_changed_handler = session_item_changed_handler

        # create left panel
        self.left_panel_layout = QVBoxLayout()
        self.left_panel_layout.setAlignment(Qt.AlignTop)

        self.setLayout(self.left_panel_layout)

        session_button_layout = QHBoxLayout()
        session_button_layout.setAlignment(Qt.AlignTop)

        # Create the plus button to add new checkboxes
        add_button = QPushButton("+")
        add_button.setMaximumWidth(50)
        add_button.clicked.connect(self.add_session)
        session_button_layout.addWidget(add_button)

        # Create the delete button to remove checked checkboxes
        self.remove_button = QPushButton("-")
        self.remove_button.setMaximumWidth(50)
        self.remove_button.setDisabled(True)
        self.remove_button.clicked.connect(self.remove_session)
        session_button_layout.addWidget(self.remove_button)

        self.left_panel_layout.addLayout(session_button_layout)
        self.sessionList = QListWidget()
        self.sessionList.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.sessionList.itemSelectionChanged.connect(self.trigger_session_item_changed)
        self.left_panel_layout.addWidget(self.sessionList)

    def trigger_session_item_changed(self):
        self.__session_item_changed_handler(self.sessionList)

    def add_session(self):
        add_session_dialog = AddSessionDialog()
        if add_session_dialog.exec_() == QDialog.Accepted:
            # check if item exists
            matches = self.sessionList.findItems(add_session_dialog.session_name_box.text(), Qt.MatchExactly)
            if len(matches) > 0:
                self.infobox("That session name already exists! Try a different name.")
                return

            self.sessionList.addItem(add_session_dialog.session_name_box.text())

            # update UI states
            session_added = SessionListChangedEvent(SessionListChangedEventType.Add,
                                                    add_session_dialog.session_name_box.text())

            QApplication.postEvent(self.__chat_widget, session_added)

            self.remove_button.setDisabled(False)
            if self.sessionList.count() == 1:
                session_state_change = AnyActiveSessionsEvent(False)
                QApplication.postEvent(self.__chat_widget, session_state_change)

    def select_session(self, index):
        for item in self.sessionList.selectedItems():
            item.setSelected(False)

        # select newly created item
        self.sessionList.item(index).setSelected(True)

    def remove_session(self):
        for item in self.sessionList.selectedItems():
            idx = self.sessionList.row(item)
            session_removed = SessionListChangedEvent(SessionListChangedEventType.Remove,
                                                      self.sessionList.item(idx).text())
            QApplication.postEvent(self.__chat_widget, session_removed)
            self.sessionList.takeItem(idx)
            if self.sessionList.count() == 0:
                self.remove_button.setDisabled(True)
                session_state_change = AnyActiveSessionsEvent(True)
                QApplication.postEvent(self.__chat_widget, session_state_change)

    def infobox(self, msg):
        confirm_box = QMessageBox()
        confirm_box.setIcon(QMessageBox.Question)
        confirm_box.setWindowTitle('Information')
        confirm_box.setText(msg)
        confirm_box.setStandardButtons(QMessageBox.Ok)
        confirm_box.exec_()


class AddSessionDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        self.label = QLabel('Session name')
        self.layout.addWidget(self.label)

        self.session_name_box = QLineEdit()
        self.layout.addWidget(self.session_name_box)

        self.button_layout = QHBoxLayout()

        self.ok_button = QPushButton('Ok')
        self.ok_button.clicked.connect(self.session_dialog_ok_click)  # Accept the dialog on OK button click
        self.button_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.clicked.connect(
            self.session_dialog_cancel_click)  # Reject (close) the dialog on Cancel button click
        self.button_layout.addWidget(self.cancel_button)

        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)

    def session_dialog_ok_click(self):
        self.accept()

    def session_dialog_cancel_click(self):
        self.reject()

class SessionContainer:
    key: str
    ai_engine: OpenAIChat
    history: []

class ChatWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, session, message):
        super().__init__()

        self.__session = session
        self.__message = message

    def run(self):
        response = self.__session.conversation_chain.run(self.__message)
        self.finished.emit(response)

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.any_active_sessions_event_signal = pyqtSignal(bool)
        self.session_list_changed_event_signal = pyqtSignal(SessionListChangedEventType, str)

        self.__sessions = dict()
        self.__selected_session = None
        self.__message_worker = None


        # Set the window title and size
        self.setWindowTitle("Chat Bot")
        self.setGeometry(100, 100, 1000, 800)

        palette = self.palette()
        palette.setColor(QPalette.Window, palette.color(QPalette.Light))
        self.setPalette(palette)

        # Create the main layout and add it to the window
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Create the checklist widget
        self.checklist = SessionlistWidget(self, self.session_list_selection_changed_handler)
        layout.addWidget(self.checklist)

        # Create the main layout and add it to the window
        chat_layout = QVBoxLayout()

        # Create the chat history widget
        self.history = QTextEdit()
        self.history.setReadOnly(True)

        chat_layout.addWidget(self.history)

        # Create the message input widget
        self.input = QLineEdit()
        chat_layout.addWidget(self.input)
        self.input.setPalette(palette)

        # Create the send button widget
        self.send_button = QPushButton("Send")
        chat_layout.addWidget(self.send_button)

        # Connect the send button to the send_message() method
        self.send_button.clicked.connect(self.send_message)

        # Create the send button widget
        self.clear_button = QPushButton("Clear")
        chat_layout.addWidget(self.clear_button)
        self.clear_button.clicked.connect(self.clear_history)

        layout.addLayout(chat_layout)

        # Initialize the counter for new items
        self.item_counter = 1

    def clear_history(self):
        self.history.clear()
        self.__selected_session.history = ""

    def session_list_selection_changed_handler(self, list: QListWidget):
        print("session_list_selection_changed_handler invoked")
        selected_items = list.selectedItems()
        #if len(selected_items) == 0:
        #    self.__selected_session = None

        for item in selected_items:
            # save history window text
            old_session = None
            if self.__selected_session is not None:
                self.__sessions[self.__selected_session.key].history = self.history.toHtml()
                # self.__selected_session.history = self.history.toHtml()

            self.__selected_session = self.__sessions[item.text()]

            # restore session text
            #if len(self.__selected_session.history) != 0:
            self.history.setHtml(self.__selected_session.history)

    def event(self, event):
        if event.type() == AnyActiveSessionsEvent.Type:
            self.any_active_sessions_handler(event.state)
            return True
        elif event.type() == SessionListChangedEvent.Type:
            self.session_list_changed_handler(event.change_type, event.name)

        return super().event(event)

    def any_active_sessions_handler(self, state):
        self.set_chat_disabled(state)

    def session_list_changed_handler(self, change_type, name):
        if change_type == SessionListChangedEventType.Add:
            api_key = os.getenv('OPENAI_KEY')
            container = SessionContainer()
            container.key = name
            container.ai_engine = OpenAIChat("gpt-3.5-turbo", api_key, temp=0.5)
            container.history = ""
            self.__sessions[name] = container
            self.checklist.select_session(self.checklist.sessionList.count() - 1)

            # Send prompt through
            #self.add_user_message(prompt_template)
            #self.input.clear()
            self.set_message_controls_disabled(True)
            self.__message_worker = ChatWorker(self.__selected_session.ai_engine, prompt_template)
            self.__message_worker.finished.connect(self.handle_ai_message)
            self.__message_worker.start()

        else:
            self.__sessions.pop(name)

    def add_user_message(self, msg):
        self.history.append(
            """
            <span style="color:#c0c0c0;padding-bottom:3pm"><B>You: </B> {message}</span><br></br>
            """.format(message=msg)
        )
    def add_ai_message(self, msg):
        self.history.append(
            """
            <span style="color:#87CEEB;padding-bottom:3pm"><B>AI: </B>{message}</span>
            """.format(message=msg)
        )
        self.history.append("\n")

    def set_message_controls_disabled(self, state):
        self.input.setDisabled(state)
        self.send_button.setDisabled(state)
        self.clear_button.setDisabled(state)

    def handle_ai_message(self, response):
        self.set_message_controls_disabled(False)
        self.add_ai_message(response)

    def send_message(self):
        # Get the user input and clear the input widget
        message = self.input.text().strip()

        self.add_user_message(message)
        self.input.clear()
        self.set_message_controls_disabled(True)

        self.__message_worker = ChatWorker(self.__selected_session.ai_engine, message)
        self.__message_worker.finished.connect(self.handle_ai_message)
        self.__message_worker.start()

    def set_chat_disabled(self, state):
        self.history.setDisabled(state)
        self.input.setDisabled(state)
        self.send_button.setDisabled(state)
        self.clear_button.setDisabled(state)


if __name__ == "__main__":
    # Create the application and chat widget
    load_dotenv()
    app = QApplication(sys.argv)
    chat_widget = ChatWindow()
    chat_widget.set_chat_disabled(True)

    # Show the chat widget and start the application event loop
    chat_widget.show()
    sys.exit(app.exec_())
