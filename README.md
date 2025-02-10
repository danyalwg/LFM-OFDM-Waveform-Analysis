# LFM + OFDM Waveform Generation and Analysis

Welcome to the **LFM + OFDM Waveform Generation and Analysis** repository. This project is a comprehensive PyQt5 application designed for generating, analyzing, and transmitting combined LFM (Linear Frequency Modulated) and OFDM (Orthogonal Frequency Division Multiplexing) waveforms. It is built for both research and real-world testing scenarios and integrates with the ADALM-Pluto SDR for actual transmission. This README is extremely detailed to provide a complete understanding of the project for clients and end-users.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Cloning the Repository](#cloning-the-repository)
  - [Installing Dependencies](#installing-dependencies)
  - [Pluto SDR Driver Installation](#pluto-sdr-driver-installation)
- [Usage](#usage)
  - [Launching the Application](#launching-the-application)
  - [Application Walkthrough](#application-walkthrough)
    - [Waveform Generation Tab](#waveform-generation-tab)
    - [Performance Analysis Tab](#performance-analysis-tab)
    - [Real-Time Transmission via ADALM-Pluto SDR](#real-time-transmission-via-adalm-pluto-sdr)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting and FAQ](#troubleshooting-and-faq)
- [Contributing](#contributing)
- [License](#license)
- [Changelog](#changelog)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Overview
The **LFM + OFDM Waveform Generation and Analysis** application provides an intuitive graphical interface that allows users to:
- **Generate LFM Waveforms:** Configure parameters such as start frequency, chirp rate, pulse width, and chirp direction (up or down).
- **Generate OFDM Waveforms:** Define the number of subcarriers, symbols per second, cyclic prefix percentage, and modulation scheme (BPSK, QPSK, or 16QAM).
- **Combine Waveforms:** Seamlessly merge an LFM prefix with OFDM data (after cyclic prefix removal) to create a joint waveform for improved synchronization and performance.
- **Analyze Performance:** Visualize time-domain plots, autocorrelation (range profile), 3D ambiguity functions (delay-Doppler analysis), and key performance indicators (KPIs) such as SNR, PSLR, data rate, BER, and detection probability.
- **Real-Time SDR Transmission:** Perform simulated transmissions (burst or continuous modes) and real-world transmission using ADALM-Pluto SDR.

This tool is ideal for researchers, engineers, and industry professionals working in radar systems, wireless communications, and signal processing.

## Features
- **Customizable Waveform Generation:**  
  - **LFM Waveform:** Adjustable start frequency, chirp rate, pulse width, and chirp type.
  - **OFDM Waveform:** Configurable number of subcarriers, symbol rate, cyclic prefix percentage, and modulation type.
- **Joint Waveform Combination:**  
  - Combines the strengths of LFM and OFDM by using an LFM prefix to improve synchronization and signal detection.
- **Advanced Performance Metrics:**  
  - Evaluate signal quality with metrics like Signal-to-Noise Ratio (SNR), Peak-to-Side Lobe Ratio (PSLR), Data Rate, Bit Error Rate (BER), and Detection Probability.
- **Real-Time SDR Integration:**  
  - Transmit and receive signals using the ADALM-Pluto SDR.
- **User-Friendly Interface:**  
  - Developed with PyQt5, offering multiple tabs for different functionalities.
  - Enhanced visual appearance using qdarkstyle.
- **Detailed Visualization:**  
  - Provides multiple plots including time-domain waveforms, range profiles, 3D ambiguity functions, and KPI bar charts.

## Project Structure
- **joint_waveform_gui _v3.py:**  
  Contains the complete PyQt5 application code, including waveform generation, combination, performance analysis, and SDR transmission.

## Prerequisites
- **Python 3.7 or Higher**
- **pip3** (Python package manager)
- **ADALM-Pluto SDR** hardware (for real-time transmission)
- **Windows OS** is recommended for Pluto SDR driver compatibility (see driver installation instructions).

## Installation

### Cloning the Repository
Clone this repository to your local machine using Git. Open a terminal (or Command Prompt) and run:

```
git clone https://github.com/yourusername/LFM-OFDM-Waveform-Analysis.git
```

Navigate to the project directory:

```
cd LFM-OFDM-Waveform-Analysis
```

### Installing Dependencies
It is recommended to create a virtual environment. Then install the required libraries using:

```
pip3 install numpy matplotlib scipy PyQt5 qdarkstyle adi iio
```


### Pluto SDR Driver Installation
For the ADALM-Pluto SDR integration, you must install the appropriate drivers. Please follow the official instructions for Windows:

[Pluto SDR Drivers for Windows](https://wiki.analog.com/university/tools/pluto/drivers/windows)

Make sure to check system requirements and installation guidelines on the official page to ensure proper driver setup.

## Usage

### Launching the Application
After completing the installation, launch the application by executing:

```
python "joint_waveform_gui _v3.py"
```

This will open the main window of the application, giving you access to waveform generation, performance analysis, and SDR transmission functionalities.

### Application Walkthrough

#### Waveform Generation Tab
- **LFM Parameters:**
  - **Start Frequency (Hz):** Input the initial frequency of the chirp signal.
  - **Chirp Rate (Hz/s):** Define the rate of frequency change.
  - **Pulse Width (s):** Set the duration of the LFM pulse.
  - **Chirp Type:** Select either "up" (increasing frequency) or "down" (decreasing frequency).
  - **Computed Stop Frequency (Hz):** Automatically computed based on your inputs.
  
- **OFDM Parameters:**
  - **OFDM Subcarriers:** Specify how many subcarriers to use.
  - **Symbols per Second:** Set the rate at which symbols are transmitted.
  - **Cyclic Prefix (%):** Define the percentage of the symbol to be used as a cyclic prefix.
  - **Modulation Scheme:** Choose between BPSK, QPSK, or 16QAM.
  
- **Transmission Mode:**
  - **Burst Mode:** For a single, one-time transmission.
  - **Continuous Mode:** For repeated transmissions at regular intervals.
  
- **ADALM-Pluto SDR Parameters:**
  - **Sample Rate (Hz):** Set the SDR sample rate.
  - **Center Frequency (Hz):** Input the operating frequency.
  - **TX Gain (dB):** Adjust the transmission gain.
  
- **Control Buttons:**
  - **Clear Results:** Clears all previous plots and outputs.
  - **Transmit (Simulated):** Generates and displays the waveform without hardware transmission.
  - **Transmit via ADALM Pluto:** Initiates real-time transmission using the Pluto SDR.
  - **Stop Transmission:** Stops any ongoing continuous transmission.

#### Performance Analysis Tab
- **Range Profile:**  
  - Visualizes the autocorrelation of the transmitted waveform to evaluate range resolution.
- **Ambiguity Function:**  
  - Displays a 3D surface plot showing delay-Doppler characteristics.
- **Key Performance Indicators (KPIs):**
  - **Signal-to-Noise Ratio (SNR):** Measured in decibels (dB).
  - **Peak-to-Side Lobe Ratio (PSLR):** Indicates the ratio between the main peak and sidelobes.
  - **Data Rate:** Calculated based on the number of subcarriers, modulation type, and symbol rate.
  - **Bit Error Rate (BER):** Estimated from the SNR.
  - **Detection Probability:** The probability of successfully detecting a target.

#### Real-Time Transmission via ADALM-Pluto SDR
- **Setup:**
  - Ensure your ADALM-Pluto SDR is connected to your computer.
  - Confirm that the Pluto SDR drivers are properly installed.
- **Operation:**
  - Click the "Transmit via ADALM Pluto" button.
  - The application will configure the SDR according to the parameters set in the GUI.
  - The joint waveform (combining LFM and OFDM) will be transmitted.
  - Received signals are processed and displayed in the "Pluto Received" tab for further analysis.

## Advanced Configuration
For power users and developers, several advanced options are available:
- **Waveform Scaling:**  
  - Modify the scaling factors for LFM and OFDM signals directly in `main.py` to fine-tune the amplitude.
- **Cyclic Prefix Management:**  
  - Adjust the algorithms for adding and removing the cyclic prefix as needed.
- **Custom Performance Metrics:**  
  - Update or replace the methods for calculating SNR, PSLR, BER, and detection probability.
- **SDR Configuration:**  
  - Tweak additional SDR parameters such as RF bandwidth and hardware gain within the code.
- **Source Code Exploration:**  
  - Developers can review the detailed implementation in `main.py` to understand signal generation, processing, and GUI integration.

## Troubleshooting and FAQ

### Common Issues
- **Invalid Parameter Inputs:**  
  - Double-check that all numerical inputs are entered in the correct format (e.g., decimal numbers for frequencies and time durations).
- **Data Reshaping Errors:**  
  - Ensure that the chosen number of subcarriers, symbols, and modulation scheme bits are consistent. Mismatches can cause errors during data reshaping for OFDM modulation.
- **Pluto SDR Not Detected:**  
  - Verify that your ADALM-Pluto SDR is properly connected, powered, and that the drivers are installed.
  - Check your system’s device manager for proper detection.
- **Application Freezing or Crashing:**  
  - Review error messages in the terminal; common issues are related to incorrect parameter values or driver problems.

### Frequently Asked Questions
- **Q: Can this application run on operating systems other than Windows?**  
  **A:** Yes, the core functionalities are cross-platform. However, the ADALM-Pluto SDR integration is optimized for Windows due to driver support.
  
- **Q: How can I modify the waveform parameters?**  
  **A:** Use the GUI to adjust parameters. For advanced modifications, directly edit the source code in `main.py`.
  
- **Q: Where can I find more detailed information about Pluto SDR?**  
  **A:** Visit the official [Pluto SDR Wiki](https://wiki.analog.com/university/tools/pluto) for comprehensive documentation and updates.
  
- **Q: What should I do if I encounter an error during installation?**  
  **A:** Ensure all dependencies are correctly installed. Consult the troubleshooting section above and verify your Python and driver installations.
