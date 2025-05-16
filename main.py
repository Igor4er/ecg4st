import wfdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.style.use("dark_background")

class ECGViewer:
    # Configuration constants
    RECORD_PATH = 'mitdb/101'
    ANNOTATION_TYPE = 'atr'
    CHANNEL_INDEX = 0

    # QRS detection settings
    QRS_BOX_WIDTH = 0.025  # seconds (half-width)
    QRS_BOX_COLOR = 'red'  # Annotated QRS
    CALC_QRS_BOX_COLOR = 'blue'  # Calculated QRS
    QRS_BOX_ALPHA = 0.3
    # QRS box vertical positioning
    QRS_TOP_POS = (0.5, 1.0)  # (ymin, ymax) for annotated QRS boxes (top half)
    QRS_BOTTOM_POS = (0.0, 0.5)  # (ymin, ymax) for calculated QRS boxes (bottom half)

    # Time navigation settings
    SLIDER_STEP = 1  # seconds
    DEFAULT_WINDOW_SIZE = 5  # seconds
    MIN_WINDOW_SIZE = 1  # seconds
    MAX_WINDOW_SIZE = 10  # seconds

    def __init__(self):
        # Load ECG data and annotations
        self.record = wfdb.rdrecord(self.RECORD_PATH)
        self.annotation = wfdb.rdann(self.RECORD_PATH, self.ANNOTATION_TYPE)
        self.signal = self.record.p_signal[:, self.CHANNEL_INDEX]
        self.fs = self.record.fs
        self.qrs_locs = self.annotation.sample
        self.calc_qrs_locs = self.calculate_qrs()  # Calculate QRS using our algorithm
        self.total_secs = len(self.signal) // self.fs

        # Current view settings
        self.current_window_size = self.DEFAULT_WINDOW_SIZE
        self.current_start_sec = 0

        # Plot elements
        self.fig = None
        self.ax_ecg = None
        self.ax_hr = None  # Heart rate display axis
        self.line = None
        self.slider = None
        self.win_slider = None
        self.qrs_boxes = []
        self.calc_qrs_boxes = []  # Storage for calculated QRS boxes

        # Heart rate display elements
        self.anno_hr_text = None  # Annotated heart rate display
        self.calc_hr_text = None  # Calculated heart rate display
        self.hr_min_text = None   # Minimum heart rate
        self.hr_max_text = None   # Maximum heart rate

        self.setup_plot()

    def setup_plot(self):
        """Initialize and configure the plot and controls"""
        # Setup the plot and controls"""
        self.fig, (self.ax_ecg, self.ax_hr, ax_slider, ax_window) = plt.subplots(
            4, 1, figsize=(15, 9),
            gridspec_kw={'height_ratios': [6, 1, 0.4, 0.4]},
            facecolor="#222222"
        )
        plt.subplots_adjust(bottom=0, hspace=0.5)

        # Setup initial ECG plot
        start = int(self.current_start_sec * self.fs)
        end = start + int(self.current_window_size * self.fs)
        time = self.get_time_segment(start, end)
        segment = self.get_signal_segment(start, end)

        self.line, = self.ax_ecg.plot(time, segment, label="ECG", color="lightblue")
        self.plot_qrs_boxes(start, end)

        self.ax_ecg.set_title(f"ЕКГ від {self.current_start_sec:.1f} до {self.current_start_sec + self.current_window_size:.1f} секунд")
        self.ax_ecg.set_xlabel("Час (с)")
        self.ax_ecg.set_ylabel("Амплітуда (мВ)")
        self.ax_ecg.set_facecolor("#202020")
        self.ax_ecg.grid(True, color="#229922")

        # Configure heart rate display area
        self.ax_hr.set_facecolor("#101010")
        self.ax_hr.set_xticks([])
        self.ax_hr.set_yticks([])
        self.ax_hr.set_frame_on(True)
        self.ax_hr.set_title("Інформація про сердечний ритм", color="white", fontsize=12, pad=2)

        # Create a background rectangle for heart rate section
        self.ax_hr.add_patch(plt.Rectangle(
            (0.01, 0.05), 0.98, 0.9,
            facecolor="#202020",
            edgecolor="#555555",
            alpha=0.8,
            transform=self.ax_hr.transAxes
        ))

        # Add heart rate text displays to the dedicated heart rate axis
        # First row - Main heart rates
        # Annotated heart rate
        self.anno_hr_text = self.ax_hr.text(
            0.05, 0.7, "Анотована ЧСС: -- уд/хв",
            transform=self.ax_hr.transAxes,
            fontsize=12, color=self.QRS_BOX_COLOR,
            fontweight='bold',
            verticalalignment='center'
        )

        # Calculated heart rate
        self.calc_hr_text = self.ax_hr.text(
            0.4, 0.7, "Вирахувана ЧСС: -- уд/хв",
            transform=self.ax_hr.transAxes,
            fontsize=12, color=self.CALC_QRS_BOX_COLOR,
            fontweight='bold',
            verticalalignment='center'
        )

        # Second row - Additional statistics
        # Min heart rate
        self.hr_min_text = self.ax_hr.text(
            0.05, 0.3, "Мінімальна ЧСС: -- уд/хв",
            transform=self.ax_hr.transAxes,
            fontsize=11, color="#8a8a8a",
            verticalalignment='center'
        )

        # Max heart rate
        self.hr_max_text = self.ax_hr.text(
            0.4, 0.3, "Максимальна ЧСС: -- уд/хв",
            transform=self.ax_hr.transAxes,
            fontsize=11, color="#8a8a8a",
            verticalalignment='center'
        )

        # Setup time navigation slider
        ax_slider.set_position([0.15, 0.1, 0.7, 0.03])
        self.slider = Slider(
            ax_slider, 'Початок (с)', 0, self.total_secs - self.current_window_size,
            valinit=self.current_start_sec, valstep=self.SLIDER_STEP,
            color="#34a334"
        )
        self.slider.on_changed(self.on_slider_change)

        # Setup window size slider
        ax_window.set_position([0.15, 0.05, 0.7, 0.03])
        self.win_slider = Slider(
            ax_window, 'Розмір вікна перегляду (с)', self.MIN_WINDOW_SIZE, self.MAX_WINDOW_SIZE,
            valinit=self.current_window_size, valstep=1,
            color="#a33434"
        )
        self.win_slider.on_changed(self.on_window_size_change)

        # Setup keyboard navigation
        # Add legend for QRS boxes
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.QRS_BOX_COLOR, alpha=self.QRS_BOX_ALPHA, label='Анотовані QRS (верх)'),
            Patch(facecolor=self.CALC_QRS_BOX_COLOR, alpha=self.QRS_BOX_ALPHA, label='Вирахувані QRS (низ)')
        ]
        self.ax_ecg.legend(handles=legend_elements, loc='upper right')

        plt.figtext(
            0.1, 0.01,
            "Аналіз електрокардіографічного (ЕКГ) сигналу - виконав студент групи ФЕМ-21, Чередніченко Ігор",
            wrap=True, horizontalalignment='left'
        )
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def get_time_segment(self, start, end):
        """Get time array for the specified sample range"""
        return [i / self.fs for i in range(start, end)]

    def get_signal_segment(self, start, end):
        """Get signal segment for the specified sample range"""
        return self.signal[start:end]

    def plot_qrs_boxes(self, start_sample, end_sample):
        """Plot QRS detection boxes within the current view"""
        # Remove existing boxes
        for box in self.qrs_boxes:
            box.remove()
        self.qrs_boxes.clear()

        for box in self.calc_qrs_boxes:
            box.remove()
        self.calc_qrs_boxes.clear()

        # Add new boxes for annotated QRS complexes in the current view (top half)
        for qrs in self.qrs_locs:
            if start_sample <= qrs < end_sample:
                qrs_time = qrs / self.fs
                box = self.ax_ecg.axvspan(
                    qrs_time - self.QRS_BOX_WIDTH,
                    qrs_time + self.QRS_BOX_WIDTH,
                    ymin=self.QRS_TOP_POS[0],
                    ymax=self.QRS_TOP_POS[1],
                    color=self.QRS_BOX_COLOR,
                    alpha=self.QRS_BOX_ALPHA
                )
                self.qrs_boxes.append(box)

        # Add boxes for calculated QRS complexes (bottom half)
        for qrs in self.calc_qrs_locs:
            if start_sample <= qrs < end_sample:
                qrs_time = qrs / self.fs
                box = self.ax_ecg.axvspan(
                    qrs_time - self.QRS_BOX_WIDTH,
                    qrs_time + self.QRS_BOX_WIDTH,
                    ymin=self.QRS_BOTTOM_POS[0],
                    ymax=self.QRS_BOTTOM_POS[1],
                    color=self.CALC_QRS_BOX_COLOR,
                    alpha=self.QRS_BOX_ALPHA
                )
                self.calc_qrs_boxes.append(box)

    def update_plot(self):
        """Update the plot with current settings"""
        start_sample = int(self.current_start_sec * self.fs)
        end_sample = start_sample + int(self.current_window_size * self.fs)

        # Ensure we don't go beyond the signal boundaries
        if end_sample > len(self.signal):
            end_sample = len(self.signal)
            start_sample = end_sample - int(self.current_window_size * self.fs)
            self.current_start_sec = start_sample / self.fs

        segment = self.get_signal_segment(start_sample, end_sample)
        time = self.get_time_segment(start_sample, end_sample)

        self.line.set_data(time, segment)
        self.ax_ecg.set_xlim(time[0], time[-1])
        self.ax_ecg.set_title(f"ЕКГ від {self.current_start_sec:.1f} до {self.current_start_sec + self.current_window_size:.1f} секунд")

        self.plot_qrs_boxes(start_sample, end_sample)

        # Update heart rate displays
        anno_results = self.calculate_heart_rate(start_sample, end_sample, use_annotated=True)
        calc_results = self.calculate_heart_rate(start_sample, end_sample, use_annotated=False)

        # Unpack results
        anno_hr, anno_inst_hr, anno_min_hr, anno_max_hr, anno_hrv = anno_results
        calc_hr, calc_inst_hr, calc_min_hr, calc_max_hr, calc_hrv = calc_results

        # Update annotated heart rate
        if anno_hr is not None:
            self.anno_hr_text.set_text(f"Анотовані ЧСС: {anno_hr:.1f} уд/хв")
            self.anno_hr_text.set_color("#ff5555")  # Brighten color when value is present
        else:
            self.anno_hr_text.set_text("Анотовані ЧСС: -- уд/хв")
            self.anno_hr_text.set_color(self.QRS_BOX_COLOR)  # Reset to default color

        # Update calculated heart rate
        if calc_hr is not None:
            self.calc_hr_text.set_text(f"Вирахувані ЧСС: {calc_hr:.1f} уд/хв")
            self.calc_hr_text.set_color("#5555ff")  # Brighten color when value is present
        else:
            self.calc_hr_text.set_text("Вирахувані ЧСС: -- уд/хв")
            self.calc_hr_text.set_color(self.CALC_QRS_BOX_COLOR)  # Reset to default color

        # Update additional heart rate statistics - prefer annotated when available
        # Min heart rate
        min_hr = anno_min_hr if anno_min_hr is not None else calc_min_hr
        if min_hr is not None:
            self.hr_min_text.set_text(f"Мін. ЧСС: {min_hr:.1f} уд/хв")
            self.hr_min_text.set_color("#cccccc")
        else:
            self.hr_min_text.set_text("Мін. ЧСС: -- уд/хв")
            self.hr_min_text.set_color("#8a8a8a")

        # Max heart rate
        max_hr = anno_max_hr if anno_max_hr is not None else calc_max_hr
        if max_hr is not None:
            self.hr_max_text.set_text(f"Макс. ЧСС: {max_hr:.1f} уд/хв")
            self.hr_max_text.set_color("#cccccc")
        else:
            self.hr_max_text.set_text("Макс. ЧСС: -- уд/хв")
            self.hr_max_text.set_color("#8a8a8a")

        self.fig.canvas.draw_idle()

    def on_slider_change(self, val):
        """Handle time slider changes"""
        self.current_start_sec = val
        self.update_plot()

    def on_window_size_change(self, val):
        """Handle window size slider changes"""
        self.current_window_size = int(val)

        # Update the time slider's maximum value
        max_start = self.total_secs - self.current_window_size
        self.slider.valmax = max(0, max_start)
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)

        # If current position is beyond new max, adjust it
        if self.current_start_sec > max_start:
            self.current_start_sec = max_start
            self.slider.set_val(max_start)
        else:
            self.update_plot()

    def on_key_press(self, event):
        """Handle keyboard navigation"""
        if event.key == 'right':
            new_val = min(self.current_start_sec + self.SLIDER_STEP, self.slider.valmax)
            self.slider.set_val(new_val)
        elif event.key == 'left':
            new_val = max(self.current_start_sec - self.SLIDER_STEP, self.slider.valmin)
            self.slider.set_val(new_val)
        elif event.key == 'home':
            self.slider.set_val(self.slider.valmin)
        elif event.key == 'end':
            self.slider.set_val(self.slider.valmax)

    def calculate_qrs(self):
        """Simple QRS detection algorithm (improved Pan-Tompkins)"""
        signal = self.signal.copy()

        # Step 1: Bandpass filter - use simple moving average for low-pass
        # and differentiation for high-pass effects

        # Differentiate to emphasize QRS slopes - improved to catch more subtle changes
        diff_signal = np.zeros_like(signal)
        for i in range(4, len(signal)):
            # Use a wider window for differentiation to catch more subtle changes
            diff_signal[i] = 1.5 * (signal[i] - signal[i-1]) + 1.0 * (signal[i-1] - signal[i-3])

        # Square to make all values positive and emphasize larger differences
        squared_signal = diff_signal ** 2

        # Moving average integration with improved sensitivity
        integration_window = int(0.12 * self.fs)  # 120ms window - wider to catch more peaks
        integrated_signal = np.zeros_like(squared_signal)

        # Enhanced integration with emphasis on recent samples
        for i in range(len(squared_signal)):
            start_idx = max(0, i - integration_window)
            window_samples = squared_signal[start_idx:i+1]
            # Weight recent samples more heavily
            if len(window_samples) > 1:
                weights = np.linspace(0.5, 1.0, len(window_samples))
                integrated_signal[i] = np.average(window_samples, weights=weights)
            else:
                integrated_signal[i] = squared_signal[i]

        # Step 2: Adaptive thresholding
        # Use a lower percentile threshold to catch more potential QRS complexes
        threshold = np.percentile(integrated_signal, 85)

        # Step 3: Identify peaks above threshold
        qrs_indices = []
        # Reduced minimum distance between QRS complexes (refractory period)
        # to catch more closely spaced beats
        refractory = int(0.15 * self.fs)  # 150ms

        i = 0
        while i < len(integrated_signal):
            if integrated_signal[i] > threshold:
                # Found potential QRS - find the highest point in this region
                window_start = i
                window_end = min(i + refractory, len(integrated_signal))

                # Find actual peak location
                peak_idx = window_start + np.argmax(integrated_signal[window_start:window_end])

                # Dynamically adjust threshold after each peak to improve detection
                # Lower the threshold if we haven't found many peaks yet
                if len(qrs_indices) < 5:
                    threshold = max(np.percentile(integrated_signal, 80), integrated_signal[peak_idx] * 0.5)

                # Find the corresponding peak in the original signal
                # Look for the actual R wave peak near this location
                r_search_window = int(0.08 * self.fs)  # 80ms window to look for R peak
                r_start = max(0, peak_idx - r_search_window)
                r_end = min(len(signal), peak_idx + r_search_window)

                # Find the R peak in the original signal
                r_peak = r_start + np.argmax(signal[r_start:r_end])

                qrs_indices.append(r_peak)

                # Skip ahead past refractory period
                i = peak_idx + refractory
            else:
                i += 1

        return np.array(qrs_indices)

    def calculate_heart_rate(self, start_sample, end_sample, use_annotated=True):
        """Calculate heart rate based on QRS intervals in the current view

        Args:
            start_sample: Start sample of current view
            end_sample: End sample of current view
            use_annotated: If True, use annotated QRS locations, otherwise use calculated ones

        Returns:
            tuple: (average_hr, instantaneous_hr, min_hr, max_hr, hr_std) or (None, None, None, None, None)
                   if not enough QRS complexes
        """
        # Select QRS source based on parameter
        qrs_source = self.qrs_locs if use_annotated else self.calc_qrs_locs

        # Get QRS locations in current view
        qrs_in_view = [qrs for qrs in qrs_source if start_sample <= qrs < end_sample]

        if len(qrs_in_view) < 2:
            return None, None, None, None, None  # Not enough QRS complexes to calculate HR

        # Calculate RR intervals in seconds
        rr_intervals = np.diff(qrs_in_view) / self.fs

        # Calculate average heart rate
        avg_rr = np.mean(rr_intervals)
        avg_heart_rate = 60 / avg_rr

        # Calculate instantaneous heart rate (from most recent interval if available)
        # Use the last 3 intervals if available for smoother instantaneous HR
        if len(rr_intervals) >= 3:
            recent_rr = np.mean(rr_intervals[-3:])
        else:
            recent_rr = rr_intervals[-1]

        inst_heart_rate = 60 / recent_rr

        # Calculate additional heart rate statistics
        min_heart_rate = 60 / np.max(rr_intervals)  # Longest RR interval = lowest HR
        max_heart_rate = 60 / np.min(rr_intervals)  # Shortest RR interval = highest HR

        # Heart rate variability - standard deviation of NN intervals (SDNN) in milliseconds
        # SDNN is a common HRV metric measuring the standard deviation of normal-to-normal intervals
        hr_variability = np.std(rr_intervals) * 1000  # Convert to milliseconds

        return avg_heart_rate, inst_heart_rate, min_heart_rate, max_heart_rate, hr_variability

    def show(self):
        """Display the ECG viewer"""
        plt.show()


if __name__ == "__main__":
    viewer = ECGViewer()
    viewer.show()
