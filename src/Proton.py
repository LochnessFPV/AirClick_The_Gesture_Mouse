"""Proton, the voice and text assistant front end for AirClick.

Run this module to launch the application: it starts the UI window, then
listens for typed or spoken commands.
"""

import datetime
import logging
import re
import sys
import time
import webbrowser
from pathlib import Path
from threading import Thread
from urllib.parse import quote_plus

import speech_recognition as sr
from pynput.keyboard import Controller, Key

import airclick_platform as system
import app
from airclick_settings import configure_logging, get_settings

LOGGER = logging.getLogger("airclick.assistant")

# -------------Object Initialization---------------
recognizer = sr.Recognizer()
recognizer.energy_threshold = 500
recognizer.dynamic_energy_threshold = False
recognizer.pause_threshold = 0.8

keyboard = Controller()
settings = get_settings()

WAKE_WORD = "proton"
BROWSE_ROOT = Path.home()

# ----------------Variables------------------------
file_exp_status = False
files = []
path = BROWSE_ROOT
pending_open = None  # file awaiting the user's confirmation


# ------------------Functions----------------------
def reply(audio):
    """Show a message in the UI and speak it."""
    app.ChatBot.addAppMsg(audio)
    LOGGER.info("%s", audio)
    system.speaker.say(audio)


def said(voice_data, *phrases):
    """True when one of 'phrases' appears as whole words.

    Substring matching used to fire 'search' on "research" and 'by' on almost
    anything, so commands are matched on word boundaries instead.
    """
    return any(
        re.search(r"\b" + re.escape(phrase) + r"\b", voice_data) for phrase in phrases
    )


def trailing_index(voice_data):
    """The 1-based item number at the end of a command, or None."""
    match = re.search(r"(\d+)\s*$", voice_data)
    if not match:
        return None
    return int(match.group(1))


def wish():
    hour = datetime.datetime.now().hour

    if hour < 12:
        reply("Good Morning!")
    elif hour < 18:
        reply("Good Afternoon!")
    else:
        reply("Good Evening!")

    reply("I am Proton, how may I help you?")


# Audio to String
def record_audio():
    """Listen once and return the recognised text in lower case."""
    try:
        with sr.Microphone() as source:
            audio = recognizer.listen(source, phrase_time_limit=5)
    except OSError:
        LOGGER.warning("No microphone is available; voice input is off")
        settings.set("modes", "voice_assistant", False)
        reply("I cannot find a microphone, so I have switched voice input off.")
        return ""

    try:
        return recognizer.recognize_google(audio).lower()
    except sr.RequestError:
        reply("Speech recognition is unreachable. Please check your internet connection.")
    except sr.UnknownValueError:
        LOGGER.debug("Speech was not recognised")
    return ""



# Executes Commands (input: string)
def respond(voice_data):
    """Handle one command. Returns False when the assistant should quit."""
    global file_exp_status, files, is_awake, path, pending_open

    voice_data = voice_data.replace(WAKE_WORD, "").strip()
    LOGGER.debug("Command: %s", voice_data)

    if not is_awake:
        if said(voice_data, "wake up"):
            is_awake = True
            wish()
        return True

    if pending_open is not None:
        return confirm_pending_open(voice_data)

    # STATIC CONTROLS
    if said(voice_data, "hello", "hi"):
        wish()

    elif said(voice_data, "what is your name"):
        reply("My name is Proton!")

    elif said(voice_data, "date"):
        reply(datetime.date.today().strftime("%B %d, %Y"))

    elif said(voice_data, "time"):
        reply(datetime.datetime.now().strftime("%H:%M:%S"))

    elif said(voice_data, "search"):
        query = voice_data.split("search", 1)[1].strip()
        if not query:
            reply("What would you like me to search for?")
        else:
            open_url("https://google.com/search?q=" + quote_plus(query),
                     "Searching for " + query)

    elif said(voice_data, "location"):
        reply("Which place are you looking for?")
        place = record_audio()
        if not place:
            reply("I did not catch that.")
        else:
            app.ChatBot.addUserMsg(place)
            open_url("https://www.google.com/maps/search/" + quote_plus(place),
                     "Locating " + place)

    elif said(voice_data, "bye", "goodbye"):
        reply("Good bye Sir! Have a nice day.")
        is_awake = False

    elif said(voice_data, "exit", "terminate", "quit"):
        app.stop_gesture()
        app.ChatBot.close()
        return False

    # DYNAMIC CONTROLS
    elif said(voice_data, "launch gesture recognition", "start gesture recognition"):
        ok, message = app.start_gesture()
        reply(message if not ok else "Launched successfully")

    elif said(voice_data, "stop gesture recognition", "top gesture recognition"):
        ok, message = app.stop_gesture()
        reply(message if not ok else "Gesture recognition stopped")

    elif said(voice_data, "copy"):
        with keyboard.pressed(Key.ctrl):
            keyboard.press("c")
            keyboard.release("c")
        reply("Copied")

    elif said(voice_data, "paste", "page", "pest"):
        with keyboard.pressed(Key.ctrl):
            keyboard.press("v")
            keyboard.release("v")
        reply("Pasted")

    # File Navigation
    elif said(voice_data, "list"):
        path = BROWSE_ROOT
        if list_directory(path):
            file_exp_status = True
            reply("These are the files in your home folder")

    elif file_exp_status:
        handle_file_navigation(voice_data)

    else:
        reply("I am not functioned to do this!")

    return True


def open_url(url, message):
    """Open a URL in the browser, reporting failures to the user."""
    reply(message)
    try:
        webbrowser.open(url)
    except webbrowser.Error:
        LOGGER.exception("Could not open the browser")
        reply("I could not open your browser.")


def list_directory(target):
    """Show the contents of 'target' in the chat window."""
    global files
    try:
        files = sorted(entry.name for entry in Path(target).iterdir())
    except OSError:
        LOGGER.warning("Could not read %s", target, exc_info=True)
        reply("I cannot read that folder.")
        return False

    if not files:
        reply("That folder is empty.")
        return True

    listing = "\n".join(
        "{}:  {}".format(number, name) for number, name in enumerate(files, start=1)
    )
    app.ChatBot.addAppMsg(listing)
    return True


def resolve_choice(voice_data):
    """Map a spoken item number onto an entry of the current listing."""
    index = trailing_index(voice_data)
    if index is None:
        reply("Please tell me the number of the item.")
        return None
    if not 1 <= index <= len(files):
        reply("There is no item number {} in this folder.".format(index))
        return None
    return Path(path) / files[index - 1]


def handle_file_navigation(voice_data):
    global file_exp_status, path, pending_open

    if said(voice_data, "open"):
        target = resolve_choice(voice_data)
        if target is None:
            return
        if target.is_dir():
            path = target
            if list_directory(path):
                reply("Opened successfully")
            return
        if settings.get("general", "confirm_file_open"):
            pending_open = target
            reply("Say yes to open {}, or no to cancel.".format(target.name))
            return
        open_file(target)

    elif said(voice_data, "back", "up"):
        if Path(path) == BROWSE_ROOT:
            reply("Sorry, this is the top folder.")
            return
        path = Path(path).parent
        if list_directory(path):
            reply("Ok")

    elif said(voice_data, "close", "stop browsing"):
        file_exp_status = False
        reply("Closed the file browser.")


def open_file(target):
    """Open a file, keeping it inside the folder being browsed."""
    if system.open_path(target, root=BROWSE_ROOT):
        reply("Opening {}".format(target.name))
    else:
        reply("I am not allowed to open that.")


def confirm_pending_open(voice_data):
    global pending_open, file_exp_status

    target, pending_open = pending_open, None
    if said(voice_data, "yes", "yeah", "confirm", "ok"):
        open_file(target)
        file_exp_status = False
    else:
        reply("Cancelled.")
    return True


# ------------------Driver Code--------------------


is_awake = True  # Bot status


def read_command():
    """Next command from the UI, or from the microphone when enabled."""
    if app.ChatBot.isUserInput():
        voice_data = app.ChatBot.popUserInput()
        app.ChatBot.addUserMsg(voice_data)
        return voice_data.lower()

    if not settings.get("modes", "voice_assistant"):
        time.sleep(0.2)
        return None

    voice_data = record_audio()
    if not voice_data:
        return None
    # Spoken commands need the wake word; typed ones are explicit already.
    if WAKE_WORD not in voice_data:
        LOGGER.debug("Ignoring speech without the wake word: %s", voice_data)
        return None
    app.ChatBot.addUserMsg(voice_data)
    return voice_data


def main():
    configure_logging()
    LOGGER.info("Starting AirClick")

    ui_thread = Thread(target=app.ChatBot.start, name="airclick-ui", daemon=True)
    ui_thread.start()

    if not app.ChatBot.ready.wait(timeout=30):
        LOGGER.error("The UI did not start in time")
        return 1

    wish()

    try:
        while app.ChatBot.started and ui_thread.is_alive():
            voice_data = read_command()
            if not voice_data:
                continue
            try:
                if not respond(voice_data):
                    break
            except Exception:
                LOGGER.exception("Command failed: %s", voice_data)
                reply("Something went wrong while doing that.")
    except KeyboardInterrupt:
        LOGGER.info("Interrupted")
    finally:
        app.stop_gesture()
        app.ChatBot.close()
        system.speaker.shutdown()
        LOGGER.info("AirClick closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

        


