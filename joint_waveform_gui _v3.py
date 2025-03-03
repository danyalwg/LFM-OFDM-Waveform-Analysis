import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.fft import ifft
from scipy.special import erfc

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import qdarkstyle

# For 3D plotting
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.amb_colorbar = None
        self.setWindowTitle("LFM + OFDM Waveform Generation and Analysis")
        self.lfm_time = None
        self.lfm_waveform = None
        self.ofdm_waveform = None
        self.joint_waveform = None
        self.pluto_received = None  # To store Pluto SDR received data

        # QTimer for continuous transmission mode (simulation)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.continuous_transmission)

        self.setup_ui()

    def setup_ui(self):
        # Central widget and main layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Main tab widget: Waveform Generation and Performance Analysis
        self.tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Tab 1: Waveform Generation
        self.tab_waveform = QtWidgets.QWidget()
        self.tab_widget.addTab(self.tab_waveform, "Waveform Generation")
        self.setup_waveform_tab()

        # Tab 2: Performance Analysis
        self.tab_performance = QtWidgets.QWidget()
        self.tab_widget.addTab(self.tab_performance, "Performance Analysis")
        self.setup_performance_tab()

    def setup_waveform_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_waveform)
        grid = QtWidgets.QGridLayout()

        # --- LFM Parameters Group ---
        lfm_group = QtWidgets.QGroupBox("LFM Parameters")
        lfm_layout = QtWidgets.QGridLayout(lfm_group)
        lfm_layout.addWidget(QtWidgets.QLabel("LFM Start Frequency (Hz):"), 0, 0)
        self.edit_lfm_start = QtWidgets.QLineEdit("1e6")
        lfm_layout.addWidget(self.edit_lfm_start, 0, 1)
        lfm_layout.addWidget(QtWidgets.QLabel("Chirp Rate (Hz/s):"), 1, 0)
        self.edit_lfm_chirp_rate = QtWidgets.QLineEdit("4e9")
        lfm_layout.addWidget(self.edit_lfm_chirp_rate, 1, 1)
        lfm_layout.addWidget(QtWidgets.QLabel("Pulse Width (s):"), 2, 0)
        self.edit_lfm_pulse_width = QtWidgets.QLineEdit("1e-3")
        lfm_layout.addWidget(self.edit_lfm_pulse_width, 2, 1)
        lfm_layout.addWidget(QtWidgets.QLabel("Chirp Type:"), 3, 0)
        self.combo_lfm_type = QtWidgets.QComboBox()
        self.combo_lfm_type.addItems(["up", "down"])
        lfm_layout.addWidget(self.combo_lfm_type, 3, 1)
        lfm_layout.addWidget(QtWidgets.QLabel("Computed Stop Frequency (Hz):"), 4, 0)
        self.edit_lfm_stop = QtWidgets.QLineEdit("5e6")
        self.edit_lfm_stop.setReadOnly(True)
        lfm_layout.addWidget(self.edit_lfm_stop, 4, 1)
        grid.addWidget(lfm_group, 0, 0)

        # --- OFDM Parameters Group ---
        ofdm_group = QtWidgets.QGroupBox("OFDM Parameters")
        ofdm_layout = QtWidgets.QGridLayout(ofdm_group)
        ofdm_layout.addWidget(QtWidgets.QLabel("OFDM Subcarriers:"), 0, 0)
        self.edit_ofdm_subcarriers = QtWidgets.QLineEdit("64")
        ofdm_layout.addWidget(self.edit_ofdm_subcarriers, 0, 1)
        ofdm_layout.addWidget(QtWidgets.QLabel("Symbols per Second:"), 1, 0)
        self.edit_ofdm_symbols = QtWidgets.QLineEdit("10")
        ofdm_layout.addWidget(self.edit_ofdm_symbols, 1, 1)
        ofdm_layout.addWidget(QtWidgets.QLabel("Cyclic Prefix (%):"), 2, 0)
        self.edit_ofdm_cp = QtWidgets.QLineEdit("10")
        ofdm_layout.addWidget(self.edit_ofdm_cp, 2, 1)
        ofdm_layout.addWidget(QtWidgets.QLabel("Modulation Scheme:"), 3, 0)
        self.combo_ofdm_mod = QtWidgets.QComboBox()
        self.combo_ofdm_mod.addItems(["BPSK", "QPSK", "16QAM"])
        ofdm_layout.addWidget(self.combo_ofdm_mod, 3, 1)
        grid.addWidget(ofdm_group, 0, 1)

        # --- Transmission Mode Group ---
        tx_mode_group = QtWidgets.QGroupBox("Transmission Mode")
        tx_mode_layout = QtWidgets.QHBoxLayout(tx_mode_group)
        self.combo_trans_mode = QtWidgets.QComboBox()
        self.combo_trans_mode.addItems(["Burst", "Continuous"])
        tx_mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        tx_mode_layout.addWidget(self.combo_trans_mode)
        grid.addWidget(tx_mode_group, 1, 0)

        # --- ADALM-Pluto SDR Parameters Group ---
        pluto_group = QtWidgets.QGroupBox("ADALM-Pluto SDR Parameters")
        pluto_layout = QtWidgets.QFormLayout(pluto_group)
        self.edit_pluto_sample_rate = QtWidgets.QLineEdit("12000000")
        self.edit_pluto_center_freq = QtWidgets.QLineEdit("500000000")
        self.edit_pluto_tx_gain = QtWidgets.QLineEdit("0")
        pluto_layout.addRow("Sample Rate (Hz):", self.edit_pluto_sample_rate)
        pluto_layout.addRow("Center Frequency (Hz):", self.edit_pluto_center_freq)
        pluto_layout.addRow("TX Gain (dB):", self.edit_pluto_tx_gain)
        grid.addWidget(pluto_group, 1, 1)

        layout.addLayout(grid)

        # --- Buttons for Transmission and Clearing Results ---
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_clear = QtWidgets.QPushButton("Clear Results")
        self.btn_clear.clicked.connect(self.clear_results)
        btn_layout.addWidget(self.btn_clear)
        self.btn_transmit = QtWidgets.QPushButton("Transmit (Simulated)")
        self.btn_transmit.clicked.connect(self.transmit_waveform)
        btn_layout.addWidget(self.btn_transmit)
        self.btn_pluto = QtWidgets.QPushButton("Transmit via ADALM Pluto")
        self.btn_pluto.clicked.connect(self.transmit_pluto)
        btn_layout.addWidget(self.btn_pluto)
        self.btn_stop = QtWidgets.QPushButton("Stop Transmission")
        self.btn_stop.clicked.connect(self.stop_transmission)
        btn_layout.addWidget(self.btn_stop)
        self.btn_analyze = QtWidgets.QPushButton("Analyze Performance")
        self.btn_analyze.clicked.connect(self.analyze_performance)
        btn_layout.addWidget(self.btn_analyze)
        layout.addLayout(btn_layout)

        # --- Tab widget for embedded waveform plots ---
        self.waveform_plot_tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.waveform_plot_tabs)

        # LFM Waveform Plot Tab
        self.tab_lfm_plot = QtWidgets.QWidget()
        self.waveform_plot_tabs.addTab(self.tab_lfm_plot, "LFM Waveform")
        self.fig_lfm, self.ax_lfm = plt.subplots(figsize=(5, 3))
        self.canvas_lfm = FigureCanvas(self.fig_lfm)
        lfm_plot_layout = QtWidgets.QVBoxLayout(self.tab_lfm_plot)
        lfm_plot_layout.addWidget(self.canvas_lfm)

        # OFDM Waveform Plot Tab
        self.tab_ofdm_plot = QtWidgets.QWidget()
        self.waveform_plot_tabs.addTab(self.tab_ofdm_plot, "OFDM Waveform")
        self.fig_ofdm, self.ax_ofdm = plt.subplots(figsize=(5, 3))
        self.canvas_ofdm = FigureCanvas(self.fig_ofdm)
        ofdm_plot_layout = QtWidgets.QVBoxLayout(self.tab_ofdm_plot)
        ofdm_plot_layout.addWidget(self.canvas_ofdm)

        # Joint Waveform Plot Tab
        self.tab_joint_plot = QtWidgets.QWidget()
        self.waveform_plot_tabs.addTab(self.tab_joint_plot, "Joint Waveform")
        self.fig_joint, self.ax_joint = plt.subplots(figsize=(5, 3))
        self.canvas_joint = FigureCanvas(self.fig_joint)
        joint_plot_layout = QtWidgets.QVBoxLayout(self.tab_joint_plot)
        joint_plot_layout.addWidget(self.canvas_joint)

        # Pluto Received Signal Plot Tab
        self.tab_pluto_received = QtWidgets.QWidget()
        self.waveform_plot_tabs.addTab(self.tab_pluto_received, "Pluto Received")
        self.fig_pluto, self.ax_pluto = plt.subplots(figsize=(5, 3))
        self.canvas_pluto = FigureCanvas(self.fig_pluto)
        pluto_received_layout = QtWidgets.QVBoxLayout(self.tab_pluto_received)
        pluto_received_layout.addWidget(self.canvas_pluto)

    def setup_performance_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_performance)
        self.perf_tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.perf_tabs)

        # Range Profile Tab
        self.tab_range = QtWidgets.QWidget()
        self.perf_tabs.addTab(self.tab_range, "Range Profile")
        self.fig_range, self.ax_range = plt.subplots(figsize=(5, 3))
        self.canvas_range = FigureCanvas(self.fig_range)
        range_layout = QtWidgets.QVBoxLayout(self.tab_range)
        range_layout.addWidget(self.canvas_range)

        # Ambiguity Function Tab (3D Plot)
        self.tab_ambiguity = QtWidgets.QWidget()
        self.perf_tabs.addTab(self.tab_ambiguity, "Ambiguity Function")
        self.fig_amb = plt.figure(figsize=(5, 3))
        self.ax_amb = self.fig_amb.add_subplot(111, projection='3d')
        self.canvas_amb = FigureCanvas(self.fig_amb)
        amb_layout = QtWidgets.QVBoxLayout(self.tab_ambiguity)
        amb_layout.addWidget(self.canvas_amb)

        # KPI Tab with grid layout for all KPIs
        self.tab_kpis = QtWidgets.QWidget()
        self.perf_tabs.addTab(self.tab_kpis, "KPIs")
        kpi_grid = QtWidgets.QGridLayout(self.tab_kpis)

        # Signal-to-Noise Ratio (SNR)
        snr_widget = QtWidgets.QWidget()
        snr_layout = QtWidgets.QVBoxLayout(snr_widget)
        self.snr_label = QtWidgets.QLabel("")
        self.snr_label.setStyleSheet("font-size: 20pt;")
        self.snr_label.setMaximumHeight(40)
        snr_layout.addWidget(self.snr_label)
        self.fig_snr, self.ax_snr = plt.subplots(figsize=(4, 3))
        self.snr_canvas = FigureCanvas(self.fig_snr)
        snr_layout.addWidget(self.snr_canvas)
        kpi_grid.addWidget(snr_widget, 0, 0)

        # Peak-to-Side Lobe Ratio (PSLR)
        pslr_widget = QtWidgets.QWidget()
        pslr_layout = QtWidgets.QVBoxLayout(pslr_widget)
        self.pslr_label = QtWidgets.QLabel("")
        self.pslr_label.setStyleSheet("font-size: 20pt;")
        self.pslr_label.setMaximumHeight(40)
        pslr_layout.addWidget(self.pslr_label)
        self.fig_pslr, self.ax_pslr = plt.subplots(figsize=(4, 3))
        self.pslr_canvas = FigureCanvas(self.fig_pslr)
        pslr_layout.addWidget(self.pslr_canvas)
        kpi_grid.addWidget(pslr_widget, 0, 1)

        # Data Rate
        dr_widget = QtWidgets.QWidget()
        dr_layout = QtWidgets.QVBoxLayout(dr_widget)
        self.data_rate_label = QtWidgets.QLabel("")
        self.data_rate_label.setStyleSheet("font-size: 20pt;")
        self.data_rate_label.setMaximumHeight(40)
        dr_layout.addWidget(self.data_rate_label)
        self.fig_data_rate, self.ax_data_rate = plt.subplots(figsize=(4, 3))
        self.data_rate_canvas = FigureCanvas(self.fig_data_rate)
        dr_layout.addWidget(self.data_rate_canvas)
        kpi_grid.addWidget(dr_widget, 1, 0)

        # Bit Error Rate (BER)
        ber_widget = QtWidgets.QWidget()
        ber_layout = QtWidgets.QVBoxLayout(ber_widget)
        self.ber_label = QtWidgets.QLabel("")
        self.ber_label.setStyleSheet("font-size: 20pt;")
        self.ber_label.setMaximumHeight(40)
        ber_layout.addWidget(self.ber_label)
        self.fig_ber, self.ax_ber = plt.subplots(figsize=(4, 3))
        self.ber_canvas = FigureCanvas(self.fig_ber)
        ber_layout.addWidget(self.ber_canvas)
        kpi_grid.addWidget(ber_widget, 1, 1)

        # Detection Probability (DP)
        dp_widget = QtWidgets.QWidget()
        dp_layout = QtWidgets.QVBoxLayout(dp_widget)
        self.dp_label = QtWidgets.QLabel("")
        self.dp_label.setStyleSheet("font-size: 20pt;")
        self.dp_label.setMaximumHeight(40)
        dp_layout.addWidget(self.dp_label)
        self.fig_dp, self.ax_dp = plt.subplots(figsize=(4, 3))
        self.dp_canvas = FigureCanvas(self.fig_dp)
        dp_layout.addWidget(self.dp_canvas)
        kpi_grid.addWidget(dp_widget, 2, 0, 1, 2)

    def modulate_data(self, data, mod_scheme):
        if mod_scheme == "BPSK":
            return 2 * data - 1
        elif mod_scheme == "QPSK":
            symbols = (2 * data[0::2] - 1) + 1j * (2 * data[1::2] - 1)
            return symbols
        elif mod_scheme == "16QAM":
            real_part = (2 * (data[0::4] + 2 * data[1::4]) - 3)
            imag_part = (2 * (data[2::4] + 2 * data[3::4]) - 3)
            return real_part + 1j * imag_part
        return data

    def generate_ofdm_waveform(self, subcarriers, symbols, mod_scheme):
        bits_per_symbol = {"BPSK": 1, "QPSK": 2, "16QAM": 4}[mod_scheme]
        total_bits = subcarriers * symbols * bits_per_symbol
        data = np.random.randint(0, 2, total_bits)
        mod_data = self.modulate_data(data, mod_scheme)
        try:
            ofdm_symbols = mod_data.reshape((subcarriers, symbols))
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", "Data reshaping failed. Check input parameters.")
            return None
        cp_percent_val = float(self.edit_ofdm_cp.text()) / 100.0
        cp_length = int(cp_percent_val * subcarriers)
        waveform_list = []
        for i in range(symbols):
            symbol_time = ifft(ofdm_symbols[:, i])
            symbol_time = symbol_time / np.max(np.abs(symbol_time))
            cp = symbol_time[-cp_length:]
            symbol_with_cp = np.concatenate((cp, symbol_time))
            waveform_list.append(symbol_with_cp)
        waveform = np.concatenate(waveform_list)
        return waveform

    def combine_waveforms(self, lfm_waveform, ofdm_waveform, cp_percent):
        subcarriers = int(self.edit_ofdm_subcarriers.text())
        cp_length = int(float(self.edit_ofdm_cp.text()) / 100 * subcarriers)
        symbols = len(ofdm_waveform) // (subcarriers + cp_length)
        ofdm_no_cp_list = []
        for i in range(symbols):
            start = i * (subcarriers + cp_length) + cp_length
            end = i * (subcarriers + cp_length) + (subcarriers + cp_length)
            ofdm_no_cp_list.append(ofdm_waveform[start:end])
        ofdm_no_cp = np.concatenate(ofdm_no_cp_list)
        lfm_prefix = lfm_waveform[:cp_length]
        joint_waveform = np.concatenate((lfm_prefix, ofdm_no_cp))
        return joint_waveform

    def generate_lfm_waveform(self, start_freq, chirp_rate, pulse_width, chirp_type, fs=20e6):
        t = np.arange(0, pulse_width, 1/fs)
        if chirp_type == "up":
            stop_freq = start_freq + chirp_rate * pulse_width
        else:
            stop_freq = start_freq - chirp_rate * pulse_width
        self.edit_lfm_stop.setText(f"{stop_freq:.2f}")
        waveform = chirp(t, f0=start_freq, t1=pulse_width, f1=stop_freq)
        waveform = waveform / np.max(np.abs(waveform))
        return t, waveform

    def generate_and_combine(self):
        try:
            start_freq = float(self.edit_lfm_start.text())
            chirp_rate = float(self.edit_lfm_chirp_rate.text())
            pulse_width = float(self.edit_lfm_pulse_width.text())
            chirp_type = self.combo_lfm_type.currentText()
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid LFM parameters.")
            return
        try:
            subcarriers = int(self.edit_ofdm_subcarriers.text())
            symbols = int(self.edit_ofdm_symbols.text())
            mod_scheme = self.combo_ofdm_mod.currentText()
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid OFDM parameters.")
            return

        fs = 20e6
        self.lfm_time, self.lfm_waveform = self.generate_lfm_waveform(start_freq, chirp_rate, pulse_width, chirp_type, fs=fs)
        self.ofdm_waveform = self.generate_ofdm_waveform(subcarriers, symbols, mod_scheme)
        if self.ofdm_waveform is None:
            return
        lfm_scaled = 0.5 * self.lfm_waveform
        ofdm_scaled = 0.5 * self.ofdm_waveform
        self.joint_waveform = self.combine_waveforms(lfm_scaled, ofdm_scaled, float(self.edit_ofdm_cp.text()))
        self.clear_results()
        self.update_waveform_plots()
        if self.pluto_received is not None:
            self.joint_waveform = self.pluto_received

    def update_waveform_plots(self):
        # LFM waveform plot (first 1000 samples)
        self.ax_lfm.cla()
        self.ax_lfm.plot(self.lfm_time[:1000], self.lfm_waveform[:1000], label="LFM")
        self.ax_lfm.set_title("LFM Waveform")
        self.ax_lfm.set_xlabel("Time (s)")
        self.ax_lfm.set_ylabel("Amplitude")
        self.ax_lfm.grid(True)
        self.ax_lfm.legend()
        self.canvas_lfm.draw()

        # OFDM waveform plot: highlight cyclic prefix (red) and data (blue)
        subcarriers = int(self.edit_ofdm_subcarriers.text())
        cp_percent_val = float(self.edit_ofdm_cp.text())
        cp_length = int(cp_percent_val / 100 * subcarriers)
        first_symbol_length = subcarriers + cp_length
        ofdm_first_symbol = self.ofdm_waveform[:first_symbol_length]
        cp_portion = ofdm_first_symbol[:cp_length]
        data_portion = ofdm_first_symbol[cp_length:]
        self.ax_ofdm.cla()
        self.ax_ofdm.plot(np.real(cp_portion), color='red', label="Cyclic Prefix")
        self.ax_ofdm.plot(np.real(data_portion), color='blue', label="OFDM Symbol")
        self.ax_ofdm.set_title("OFDM Waveform (First Symbol Highlighted)")
        self.ax_ofdm.set_xlabel("Sample Index")
        self.ax_ofdm.set_ylabel("Amplitude")
        self.ax_ofdm.legend()
        self.ax_ofdm.grid(True)
        self.canvas_ofdm.draw()

        # Joint waveform plot: LFM prefix (green) and OFDM data (blue)
        joint = self.joint_waveform
        lfm_part = joint[:cp_length]
        ofdm_part = joint[cp_length:]
        self.ax_joint.cla()
        self.ax_joint.plot(np.real(lfm_part), color='green', label="LFM Prefix")
        self.ax_joint.plot(np.real(ofdm_part), color='blue', label="OFDM Data")
        self.ax_joint.set_title("Joint Waveform (LFM Prefix + OFDM Data)")
        self.ax_joint.set_xlabel("Sample Index")
        self.ax_joint.set_ylabel("Amplitude")
        self.ax_joint.legend()
        self.ax_joint.grid(True)
        self.canvas_joint.draw()

    def clear_results(self):
        # Clear waveform plots
        self.ax_lfm.cla(); self.canvas_lfm.draw()
        self.ax_ofdm.cla(); self.canvas_ofdm.draw()
        self.ax_joint.cla(); self.canvas_joint.draw()
        self.ax_pluto.cla(); self.canvas_pluto.draw()
        # Clear performance plots
        self.ax_range.cla(); self.canvas_range.draw()
        self.ax_amb.cla(); self.canvas_amb.draw()
        self.ax_snr.cla(); self.snr_canvas.draw()
        self.ax_pslr.cla(); self.pslr_canvas.draw()
        self.ax_data_rate.cla(); self.data_rate_canvas.draw()
        self.ax_ber.cla(); self.ber_canvas.draw()
        self.ax_dp.cla(); self.dp_canvas.draw()
        # Clear KPI text labels
        self.snr_label.setText("")
        self.pslr_label.setText("")
        self.data_rate_label.setText("")
        self.ber_label.setText("")
        self.dp_label.setText("")

    def transmit_waveform(self):
        self.clear_results()
        self.pluto_received = None  # Clear previous Pluto data
        mode = self.combo_trans_mode.currentText()
        if mode == "Burst":
            self.generate_and_combine()
        elif mode == "Continuous":
            self.timer.start(5000)  # every 5000 ms (5 seconds)
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "Unknown transmission mode.")

    def continuous_transmission(self):
        self.generate_and_combine()
        if self.combo_trans_mode.currentText() == "Continuous":
            self.analyze_performance()

    def stop_transmission(self):
        self.timer.stop()

    def compute_ambiguity_function(self, t, waveform, fs, analysis_length=1000):
        t_analysis = t[:analysis_length]
        waveform_analysis = waveform[:analysis_length]
        N = len(waveform_analysis)
        delays = np.arange(-N + 1, N) / fs
        doppler_shifts = np.linspace(-5000, 5000, 50)
        ambg = np.zeros((len(doppler_shifts), len(delays)))
        for i, fd in enumerate(doppler_shifts):
            waveform_fd = waveform_analysis * np.exp(1j * 2 * np.pi * fd * t_analysis)
            corr = np.abs(np.correlate(waveform_fd, waveform_analysis, mode='full'))
            ambg[i, :] = corr / np.max(corr)
        return doppler_shifts, delays, ambg

    def analyze_performance(self):
        # Use Pluto received data if available, otherwise simulated joint waveform
        if self.pluto_received is not None:
            analysis_waveform = self.pluto_received
        else:
            analysis_waveform = self.joint_waveform

        # --- Range Profile (Autocorrelation) ---
        autocorr = np.correlate(analysis_waveform, analysis_waveform, mode='full')
        lags = np.arange(-len(analysis_waveform) + 1, len(analysis_waveform))
        self.ax_range.cla()
        self.ax_range.plot(lags, autocorr)
        self.ax_range.set_title("Range Profile (Autocorrelation)")
        self.ax_range.set_xlabel("Lag")
        self.ax_range.set_ylabel("Correlation")
        self.ax_range.grid(True)
        self.canvas_range.draw()

        # --- Ambiguity Function (3D Surface Plot) ---
        fs = 20e6
        doppler_shifts, delays, ambg = self.compute_ambiguity_function(
            self.lfm_time, self.lfm_waveform, fs, analysis_length=1000)
        # Clear the existing ambiguity figure instead of removing the tab
        self.fig_amb.clf()
        self.ax_amb = self.fig_amb.add_subplot(111, projection='3d')
        D, F = np.meshgrid(delays, doppler_shifts)
        # Plot only the real part to avoid ComplexWarnings
        surf = self.ax_amb.plot_surface(D, F, np.real(ambg), cmap='viridis')
        self.ax_amb.set_title("Ambiguity Function")
        self.ax_amb.set_xlabel("Delay (s)")
        self.ax_amb.set_ylabel("Doppler (Hz)")
        self.ax_amb.set_zlabel("Correlation")
        # Remove any existing colorbar robustly
        if self.amb_colorbar is not None:
            try:
                if hasattr(self.amb_colorbar, 'ax') and self.amb_colorbar.ax is not None:
                    self.fig_amb.delaxes(self.amb_colorbar.ax)
                self.amb_colorbar = None
            except Exception as e:
                print("Error removing colorbar:", e)
                self.amb_colorbar = None
        self.amb_colorbar = self.fig_amb.colorbar(surf, ax=self.ax_amb, shrink=0.5, aspect=10)
        self.canvas_amb.draw()

        # --- KPIs ---
        noise_variance = 0.01
        signal_power = np.mean(np.abs(analysis_waveform)**2)
        snr_linear = signal_power / noise_variance
        snr_db = 10 * np.log10(snr_linear)
        center_index = len(autocorr) // 2
        main_peak = autocorr[center_index]
        sidelobes = np.delete(autocorr, center_index)
        max_sidelobe = np.max(sidelobes)
        pslr_db = 20 * np.log10(main_peak / max_sidelobe) if max_sidelobe != 0 else np.inf
        bits_per_symbol = {"BPSK": 1, "QPSK": 2, "16QAM": 4}[self.combo_ofdm_mod.currentText()]
        subcarriers = int(self.edit_ofdm_subcarriers.text())
        symbols_per_sec = float(self.edit_ofdm_symbols.text())
        data_rate = subcarriers * bits_per_symbol * symbols_per_sec
        ber = 0.5 * erfc(np.sqrt(snr_linear))
        detection_prob = 1 - np.exp(-snr_linear / 10)

        self.snr_label.setText(f"{snr_db:.2f} dB")
        self.ax_snr.cla()
        self.ax_snr.bar(["Signal-to-Noise Ratio"], [snr_db], color='skyblue')
        self.ax_snr.set_ylim(0, snr_db * 1.2)
        self.ax_snr.set_ylabel("dB")
        self.ax_snr.set_title("Signal-to-Noise Ratio")
        self.snr_canvas.draw()

        self.pslr_label.setText(f"{pslr_db:.2f} dB")
        self.ax_pslr.cla()
        self.ax_pslr.bar(["Peak-to-Side Lobe Ratio"], [pslr_db], color='salmon')
        self.ax_pslr.set_ylim(0, pslr_db * 1.2 if pslr_db != np.inf else 10)
        self.ax_pslr.set_ylabel("dB")
        self.ax_pslr.set_title("Peak-to-Side Lobe Ratio")
        self.pslr_canvas.draw()

        self.data_rate_label.setText(f"{data_rate:.2f} bits/s")
        self.ax_data_rate.cla()
        self.ax_data_rate.bar(["Data Rate"], [data_rate], color='limegreen')
        self.ax_data_rate.set_ylabel("bits/s")
        self.ax_data_rate.set_title("Data Rate")
        self.data_rate_canvas.draw()

        self.ber_label.setText(f"{ber:.2e}")
        self.ax_ber.cla()
        self.ax_ber.bar(["Bit Error Rate"], [ber], color='violet')
        self.ax_ber.set_ylabel("BER")
        self.ax_ber.set_title("Bit Error Rate")
        self.ber_canvas.draw()

        self.dp_label.setText(f"{detection_prob:.2f}")
        self.ax_dp.cla()
        self.ax_dp.bar(["Detection Probability"], [detection_prob], color='orange')
        self.ax_dp.set_ylim(0, 1)
        self.ax_dp.set_ylabel("Probability")
        self.ax_dp.set_title("Detection Probability")
        self.dp_canvas.draw()


    def transmit_pluto(self):
        try:
            import adi
            import iio
            self.clear_results()  # Clear previous plots
            self.generate_and_combine()
            iq_signal = self.joint_waveform.astype(complex)
            contexts = iio.scan_contexts()
            pluto_uri = None
            for uri, desc in contexts.items():
                if "PlutoSDR" in desc:
                    pluto_uri = uri
                    break
            if pluto_uri is None:
                QtWidgets.QMessageBox.critical(self, "Error", "No ADALM Pluto device found.")
                return

            sdr = adi.Pluto(uri=pluto_uri)
            sdr.sample_rate = int(self.edit_pluto_sample_rate.text())
            center_freq = int(self.edit_pluto_center_freq.text())
            sdr.tx_lo = center_freq
            sdr.rx_lo = center_freq
            sdr.tx_hardwaregain_chan0 = int(self.edit_pluto_tx_gain.text())
            sdr.rx_rf_bandwidth = 6000000
            sdr.tx_cyclic_buffer = True

            sdr.tx(iq_signal)
            QtCore.QThread.sleep(1)
            received_data = sdr.rx()
            sdr.tx_destroy_buffer()
            del sdr

            processed = np.convolve(np.real(received_data), np.ones(10)/10, mode='same')
            self.pluto_received = processed
            self.ax_pluto.cla()
            self.ax_pluto.plot(processed, label="Processed Received Signal", color='blue')
            self.ax_pluto.set_title("ADALM Pluto Received Signal")
            self.ax_pluto.set_xlabel("Sample Index")
            self.ax_pluto.set_ylabel("Amplitude")
            self.ax_pluto.legend()
            self.canvas_pluto.draw()

            if self.combo_trans_mode.currentText() == "Continuous":
                QtCore.QTimer.singleShot(5000, self.transmit_pluto)
                self.analyze_performance()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
