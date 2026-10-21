document.addEventListener("DOMContentLoaded", () => {
  const messagesContainer = document.getElementById("messages-container");
  const messageForm = document.getElementById("message-form");
  const messageInput = document.getElementById("message-input");
  const micButton = document.getElementById("mic-button");

  // Helper to append message bubbles
  const addMessage = (message, role, imgSrc, audioUrl = null) => {
    const messageElement = document.createElement("div");
    messageElement.className = `message ${role} d-flex align-items-center my-2`;

    const imgElement = document.createElement("img");
    imgElement.src = imgSrc;
    imgElement.style.width = "35px";
    imgElement.style.height = "35px";
    imgElement.className = "mr-2 rounded-circle";

    const textElement = document.createElement("p");
    textElement.className = "mb-0 p-2 border rounded bg-light";
    textElement.innerText = message;

    messageElement.appendChild(imgElement);
    messageElement.appendChild(textElement);

    // If response contains an audio URL, embed audio player & auto-play
    if (audioUrl) {
      const audioElement = document.createElement("audio");
      audioElement.src = audioUrl;
      audioElement.autoplay = true;
      audioElement.controls = true;
      audioElement.className = "ml-2";
      messageElement.appendChild(audioElement);
    }

    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  };

  // 1. Text Form Submission Handler
  messageForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = messageInput.value.trim();
    if (!text) return;

    messageInput.value = "";
    addMessage(text, "user", "/static/user.jpeg");

    try {
      const response = await fetch("/api/text-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text }),
      });

      const data = await response.json();

      if (data.error) {
        addMessage(data.error, "error", "/static/Error.png");
      } else {
        addMessage(data.response, "aibot", "/static/Bot_logo.png", data.audio_url);
      }
    } catch (err) {
      addMessage("Failed to connect to backend server.", "error", "/static/Error.png");
    }
  });

  // 2. Microphone Audio Recording Handler (Web Audio API)
  let mediaRecorder;
  let audioChunks = [];

  if (micButton) {
    micButton.addEventListener("click", async () => {
      if (!mediaRecorder || mediaRecorder.state === "inactive") {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          mediaRecorder = new MediaRecorder(stream);
          audioChunks = [];

          mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);

          mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            const formData = new FormData();
            formData.append("audio", audioBlob, "mic_recording.wav");

            addMessage("🎤 Processing voice input...", "user", "/static/user.jpeg");

            try {
              const response = await fetch("/api/speech-chat", {
                method: "POST",
                body: formData,
              });
              const data = await response.json();

              if (data.error) {
                addMessage(data.error, "error", "/static/Error.png");
              } else {
                addMessage(`"${data.user_text}"`, "user", "/static/user.jpeg");
                addMessage(data.response, "aibot", "/static/Bot_logo.png", data.audio_url);
              }
            } catch (err) {
              addMessage("Error processing voice file.", "error", "/static/Error.png");
            }
          };

          mediaRecorder.start();
          micButton.classList.replace("btn-danger", "btn-warning");
          micButton.innerHTML = '<i class="fa fa-stop"></i> Listening...';
        } catch (err) {
          alert("Microphone access denied or not available.");
        }
      } else if (mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        micButton.classList.replace("btn-warning", "btn-danger");
        micButton.innerHTML = '<i class="fa fa-microphone"></i>';
      }
    });
  }
});
