# Speech Naturalness Evaluation – MOS Listening Test

This repository contains a lightweight web-based listening test designed to evaluate the **naturalness of synthesized speech** using a **Mean Opinion Score (MOS)** protocol.

The test was developed in the context of a research project on **text-to-speech (TTS) synthesis** and is intended to be shared with non-technical participants through a simple web link.

---

## Overview

Participants are asked to listen to short audio samples and rate the **perceived naturalness** of each sample on a **5-point MOS scale**.

The test is **comparative**:
- Multiple audio samples are presented **within the same question**
- Participants can freely listen and compare before rating
- This design helps reduce contextual and memory bias compared to isolated MOS questions

---

## MOS Rating Scale

The following scale is used throughout the test:

- **1 — Completely artificial**  
- **2 — Mostly artificial**  
- **3 — Uncertain / ambiguous**  
- **4 — Mostly natural**  
- **5 — Completely natural**

Only the numeric values (1–5) are stored for analysis.

---

## Test Structure

The evaluation consists of two parts:

### Test A — Temperature
- Each question presents **4 audio samples**
- Samples differ by the **temperature parameter** used during synthesis
- The number of inference steps is fixed

### Test B — Number of Steps
- Each question presents **3 audio samples**
- Samples differ by the **number of inference steps**
- The temperature is fixed

The order of questions and the order of audio samples within each question are **randomized** to reduce bias.

---

## Technical Details

- The test is implemented as a **static web page** (HTML / CSS / JavaScript)
- Audio playback is handled directly in the browser
- Responses are **automatically submitted** to a Google Apps Script endpoint
- Results are stored in a Google Sheet and aggregated statistics are computed automatically

No account, login, or installation is required for participants.

---

## Repository Structure

