# 🌍 Environment Conservation using IoT

An IoT-based project developed with **Raspberry Pi** to monitor forest environments in real-time and assist in conservation efforts. The system continuously tracks environmental conditions, detects anomalies, and sends alerts to authorities for rapid action.

---

## 🎯 Project Overview
This project is designed to **prevent illegal activities and forest hazards** by leveraging IoT and AI. It integrates multiple sensors and ML-based audio classification to provide early detection and reporting.

- 📍 **GPS Module** → Tracks exact location of incidents.
- 🔥 **Smoke Sensor** → Detects fire or smoke presence in the forest.
- 🎤 **Microphone + YAMNet (Audio Classification)** → Identifies suspicious sounds such as:
  - Gunfire 🔫
  - Vehicle sounds 🚙
  - Other abnormal noises ⚠️
- 📧 **Email Alert System** → Sends real-time alerts with location details to authorities.

---

## 🛠️ System Architecture
1. **Data Collection**: Sensors (GPS, smoke, microphone) gather real-time data.
2. **Processing**: Raspberry Pi processes sensor inputs and runs **YAMNet** for sound classification.
3. **Detection**: Identifies events like fire (smoke) or suspicious sounds.
4. **Alerting**: Sends an automated **email notification with GPS coordinates** to forest officials.

---

## 💡 Key Features
- 🌲 Continuous monitoring of forest surroundings.
- 🔍 Real-time **audio classification** using Google’s **YAMNet**.
- 📡 Automatic alerts via **email with live GPS coordinates**.
- 🚨 Early detection of **fire (smoke)** to minimize forest damage.
- 🛡️ Helps curb **illegal activities** such as poaching and deforestation.

---

## 🧩 Components Used
- **Hardware**:
  - Raspberry Pi (any model with sufficient processing power)
  - GPS Module
  - Smoke Sensor
  - Microphone
- **Software/ML**:
  - Python
  - YAMNet (for audio classification)
  - SMTP (for email notifications)
  - Raspberry Pi OS

---

## 📊 Workflow Diagram
```
[ Sensors ] → [ Raspberry Pi Processing ] → [ Detection (YAMNet + Smoke) ] → [ Email Alert w/ GPS ]
```

---

## 🚀 Future Enhancements
- Integration with **LoRaWAN/5G** for better connectivity.
- Cloud dashboard for live monitoring & visualization.
- AI-based wildfire prediction models.
- SMS/WhatsApp alerts in addition to email.

---


## 🔮 Conclusion
This project demonstrates the power of **IoT + AI for environmental conservation**. By combining **sensors, GPS, and audio classification**, it provides a proactive solution to detect **forest fires and illegal activities**, enabling faster response and safeguarding biodiversity.
