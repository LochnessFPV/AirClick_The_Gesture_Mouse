"use strict";

// ---------------------------------------------------------------- chat view

function addMessage(msg, cssClass, animationClass) {
    const element = document.getElementById("messages");
    const bubble = document.createElement("div");
    bubble.className = cssClass + " ready " + animationClass;
    // textContent, never innerHTML: chat content includes raw speech and file
    // names, which must never be interpreted as markup.
    bubble.textContent = msg;
    element.appendChild(bubble);
    element.scrollTop = element.scrollHeight;
    setTimeout(() => { bubble.className = cssClass; }, 500);
}

function addUserMsg(msg) {
    addMessage(msg, "message from", "rtol");
}

function addAppMsg(msg) {
    addMessage(msg, "message to", "ltor");
}

function getUserInput() {
    const element = document.getElementById("userInput");
    const msg = element.value;
    if (msg.length !== 0) {
        element.value = "";
        eel.getUserInput(msg);
    }
}

document.getElementById("userInputButton").addEventListener("click", getUserInput);
document.getElementById("userInput").addEventListener("keyup", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        getUserInput();
    }
});

// -------------------------------------------------------------------- tabs

function selectTab(name) {
    const isChat = name === "chat";
    document.getElementById("tabChat").classList.toggle("active", isChat);
    document.getElementById("tabSettings").classList.toggle("active", !isChat);
    document.getElementById("tabChatButton").setAttribute("aria-selected", String(isChat));
    document.getElementById("tabSettingsButton").setAttribute("aria-selected", String(!isChat));
}

document.getElementById("tabChatButton").addEventListener("click", () => selectTab("chat"));
document.getElementById("tabSettingsButton").addEventListener("click", () => selectTab("settings"));

// ------------------------------------------------------------------ status

function updateStatus(status) {
    const dot = document.getElementById("statusDot");
    const text = document.getElementById("statusText");
    const line = document.getElementById("statusLine");

    dot.classList.toggle("on", Boolean(status.running) && !status.paused);
    dot.classList.toggle("paused", Boolean(status.paused));

    if (!status.running) {
        text.textContent = "Off";
    } else if (status.paused) {
        text.textContent = "Paused";
    } else {
        text.textContent = Math.round(status.fps || 0) + " fps";
    }

    const parts = [];
    if (status.message) { parts.push(status.message); }
    if (status.running && status.gesture && status.gesture !== "none") {
        parts.push("Gesture: " + status.gesture);
    }
    line.textContent = parts.join(" \u2014 ") || "Gesture control is off.";

    document.getElementById("startButton").disabled = Boolean(status.running);
    document.getElementById("stopButton").disabled = !status.running;
}

document.getElementById("startButton").addEventListener("click", async () => {
    document.getElementById("startButton").disabled = true;
    const result = await eel.startGesture()();
    if (!result.ok) { showError(result.message); }
});

document.getElementById("stopButton").addEventListener("click", async () => {
    document.getElementById("stopButton").disabled = true;
    await eel.stopGesture()();
});

eel.expose(addUserMsg);
eel.expose(addAppMsg);
eel.expose(updateStatus);

// ---------------------------------------------------------------- settings

let schema = [];
let currentValues = {};
const pendingWrites = {};

function showError(message) {
    const box = document.getElementById("settingsError");
    box.textContent = message || "";
    if (message) {
        setTimeout(() => {
            if (box.textContent === message) { box.textContent = ""; }
        }, 6000);
    }
}

function formatValue(option, value) {
    if (option.kind === "float") {
        const decimals = option.step && option.step < 0.1 ? 2 : 1;
        return Number(value).toFixed(decimals);
    }
    return String(value);
}

/** Writes are debounced so dragging a slider does not spam the back end. */
function queueWrite(section, key, value) {
    const id = section + "." + key;
    clearTimeout(pendingWrites[id]);
    pendingWrites[id] = setTimeout(async () => {
        const result = await eel.setSetting(section, key, value)();
        if (!result.ok) {
            showError(result.error);
        } else {
            currentValues[section][key] = result.value;
        }
    }, 120);
}

function buildControl(section, option) {
    const wrapper = document.createElement("div");
    wrapper.className = "setting";

    const label = document.createElement("label");
    const name = document.createElement("span");
    name.textContent = option.label;
    label.appendChild(name);

    const value = currentValues[section.id][option.id];

    if (option.kind === "bool") {
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = Boolean(value);
        input.addEventListener("change", () => queueWrite(section.id, option.id, input.checked));
        label.appendChild(input);
        wrapper.appendChild(label);
    } else if (option.kind === "choice") {
        wrapper.appendChild(label);
        const select = document.createElement("select");
        select.id = "setting-" + section.id + "-" + option.id;
        option.choices.forEach((choice) => {
            const item = document.createElement("option");
            item.value = String(choice);
            item.textContent = String(choice);
            item.selected = String(choice) === String(value);
            select.appendChild(item);
        });
        select.addEventListener("change", () => queueWrite(section.id, option.id, select.value));
        wrapper.appendChild(select);
    } else {
        const readout = document.createElement("span");
        readout.className = "value";
        readout.textContent = formatValue(option, value);
        label.appendChild(readout);
        wrapper.appendChild(label);

        const input = document.createElement("input");
        input.type = "range";
        input.min = option.minimum;
        input.max = option.maximum;
        input.step = option.step || (option.kind === "int" ? 1 : 0.1);
        input.value = value;
        input.addEventListener("input", () => {
            readout.textContent = formatValue(option, input.value);
            queueWrite(section.id, option.id, input.value);
        });
        wrapper.appendChild(input);
    }

    if (option.help) {
        const help = document.createElement("small");
        help.className = "help";
        help.textContent = option.help;
        wrapper.appendChild(help);
    }
    return wrapper;
}

function renderSettings() {
    const body = document.getElementById("settingsBody");
    body.textContent = "";
    schema.forEach((section, index) => {
        const details = document.createElement("details");
        details.className = "settings-section";
        details.open = index < 2;
        const summary = document.createElement("summary");
        summary.textContent = section.label;
        details.appendChild(summary);
        section.options.forEach((option) => details.appendChild(buildControl(section, option)));
        body.appendChild(details);
    });
}

function renderProfiles(profiles, active) {
    const select = document.getElementById("profileSelect");
    select.textContent = "";
    profiles.forEach((profile) => {
        const item = document.createElement("option");
        item.value = profile;
        item.textContent = profile;
        item.selected = profile === active;
        select.appendChild(item);
    });
}

async function loadSettings() {
    schema = await eel.getSettingsSchema()();
    const state = await eel.getSettings()();
    currentValues = state.values;
    renderProfiles(state.profiles, state.activeProfile);
    renderSettings();
}

document.getElementById("resetAll").addEventListener("click", async () => {
    const result = await eel.resetSettings()();
    if (result.ok) {
        currentValues = result.values;
        renderSettings();
    } else {
        showError(result.error);
    }
});

document.getElementById("refreshCameras").addEventListener("click", async () => {
    const cameras = await eel.listCameras()();
    if (!cameras.length) {
        showError("No cameras were detected.");
        return;
    }
    showError("Cameras detected: " + cameras.join(", "));
});

document.getElementById("profileSelect").addEventListener("change", async (event) => {
    const result = await eel.activateProfile(event.target.value)();
    if (result.ok) {
        currentValues = result.values;
        renderSettings();
    } else {
        showError(result.error);
    }
});

document.getElementById("profileSave").addEventListener("click", async () => {
    const input = document.getElementById("profileName");
    const result = await eel.saveProfile(input.value)();
    if (result.ok) {
        input.value = "";
        renderProfiles(result.profiles, result.activeProfile);
    } else {
        showError(result.error);
    }
});

document.getElementById("profileDelete").addEventListener("click", async () => {
    const select = document.getElementById("profileSelect");
    const result = await eel.deleteProfile(select.value)();
    if (result.ok) {
        renderProfiles(result.profiles, result.activeProfile);
        const state = await eel.getSettings()();
        currentValues = state.values;
        renderSettings();
    } else {
        showError(result.error);
    }
});

window.addEventListener("load", async () => {
    try {
        await loadSettings();
        updateStatus(await eel.getStatus()());
    } catch (error) {
        showError("Could not load settings: " + error);
    }
});
